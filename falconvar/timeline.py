"""The chunk grid, owned by no stage and shared by both.

A chunk is the unit a describer summarises, a transcript is cut into, and
retrieval returns. When a file has both a picture and a soundtrack the two
have to agree about where those boundaries fall -- otherwise one question
returns two different clips and there is no honest way to say which is the
answer.

**One policy governs a run, and it may be derived from either modality.** That
is the knob worth having: what the boundaries should follow is a property of
the footage, not of the pipeline.

    uniform   a grid. Both sides derive it from the same arithmetic, so
              nothing propagates and neither has to run first.
    scene     the video pass finds them, streaming, from frame content.
    vad       the audio pass finds them, in the gaps between speech.
    speaker   the audio pass finds them, where the voice changes.

Propagation is one way per run, and it is always cheap in the same direction.
Whisper returns a timestamp per word, so a finished transcript can be re-cut
to any grid without re-running inference and without losing a word. The video
side has no equivalent -- a sampler's decisions are made during the decode
pass and cannot be revisited. So an audio-derived policy forces audio to run
first, while a video-derived one forces nothing: transcription needs no
boundaries at all, and can run concurrently and be cut afterwards.

This module imports nothing from the project, like `db` and `fanout`, because
both `audio` and `video` depend on it and neither may depend on the other.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

#: Rounding for identity. Boundaries come from float arithmetic over two
#: independent clocks, so comparing them exactly would report drift that is
#: not there. A millisecond is far finer than any boundary decision.
PRECISION = 3

#: A final chunk shorter than this fraction of the nominal chunk length is
#: merged into the one before it, under every policy.
#:
#: A grid divides the media wherever it happens to end, so the tail is
#: uniformly distributed over the chunk length -- 97.99 s at 20 s leaves a
#: usable 17.99 s, but 100.4 s leaves 0.4 s. That stub still costs a describer
#: call per sampler, still keeps a frame because every chunk keeps at least
#: one, and is still a moment retrieval can return and nobody can play. A
#: fraction rather than a fixed number of seconds so the rule holds at any
#: chunk length, and applied to every policy so `min` means the same thing
#: however the boundaries were derived.
MIN_TAIL_FRACTION = 0.25


@dataclass
class Timeline:
    """Where every chunk starts and ends, and what decided it."""

    spans: list[tuple[float, float]]
    policy: str = "uniform"
    params: dict[str, Any] = field(default_factory=dict)
    #: Which pass computed these -- "video", "audio", or "grid" when the policy
    #: is arithmetic and neither had to run first. Recorded because a reader
    #: otherwise cannot tell a propagated boundary from a natively derived one.
    derived_from: str = "grid"

    def __len__(self) -> int:
        return len(self.spans)

    @property
    def duration_s(self) -> float:
        return self.spans[-1][1] if self.spans else 0.0

    def bounds_of(self, chunk_id: int) -> tuple[float, float]:
        return self.spans[chunk_id]

    def index_at(self, ts: float) -> Optional[int]:
        """Which chunk contains ``ts``. The last chunk owns its own end."""
        for index, (start, end) in enumerate(self.spans):
            if start <= ts < end:
                return index
        if self.spans and ts == self.spans[-1][1]:
            return len(self.spans) - 1
        return None

    def fingerprint(self) -> str:
        """A hash of the grid and the policy that produced it.

        Both the manifest and the transcript record this. If they disagree,
        the two documents were cut on different grids and `chunk_id` means two
        different things -- which becomes a comparison a reader can make
        rather than a drift that nothing reports.
        """
        payload = json.dumps({
            "spans": [[round(s, PRECISION), round(e, PRECISION)]
                      for s, e in self.spans],
            "policy": self.policy,
            "params": self.params,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "timeline_version": 1,
            "policy": self.policy,
            "derived_from": self.derived_from,
            "params": self.params,
            "fingerprint": self.fingerprint(),
            "chunks": [{"chunk_id": i, "start_ts": round(s, PRECISION),
                        "end_ts": round(e, PRECISION)}
                       for i, (s, e) in enumerate(self.spans)],
        }

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> "Timeline":
        return cls(
            spans=[(float(c["start_ts"]), float(c["end_ts"]))
                   for c in document["chunks"]],
            policy=document.get("policy", "uniform"),
            params=document.get("params", {}),
            derived_from=document.get("derived_from", "grid"),
        )


def uniform(duration_s: float, chunk_s: float = 20.0) -> Timeline:
    """A fixed grid, with the last chunk truncated to where the media ends.

    The truncation is not tidiness. The pipeline already corrects a final
    chunk to the real end of the video rather than the grid's, because telling
    a model a window is twenty seconds longer than the footage is telling it
    something false about every video.
    """
    if chunk_s <= 0:
        raise ValueError("chunk_s must be positive")
    spans: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_s:
        spans.append((start, min(start + chunk_s, duration_s)))
        start += chunk_s
    spans = merge_tail(spans or [(0.0, duration_s)], chunk_s * MIN_TAIL_FRACTION)
    return Timeline(spans, "uniform", {"duration_s": chunk_s}, "grid")


def merge_tail(spans: list[tuple[float, float]],
               min_tail_s: float) -> list[tuple[float, float]]:
    """Absorb a too-short final chunk into the one before it.

    Only the last one, and only forwards into its predecessor: a short chunk
    anywhere else is a content boundary that something decided on purpose,
    while the last is an artifact of where the media stopped relative to a
    grid that knew nothing about it.
    """
    if len(spans) < 2:
        return spans
    start, end = spans[-1]
    if end - start >= min_tail_s:
        return spans
    return spans[:-2] + [(spans[-2][0], end)]


def from_cuts(cuts: Iterable[float], duration_s: float, policy: str,
              params: Optional[dict[str, Any]] = None,
              derived_from: str = "grid") -> Timeline:
    """Interior boundaries plus a total length, as spans."""
    edges = [0.0] + sorted(t for t in cuts if 0.0 < t < duration_s) + [duration_s]
    spans = [(a, b) for a, b in zip(edges, edges[1:]) if b > a]
    return Timeline(spans or [(0.0, duration_s)], policy, params or {}, derived_from)


def enforce(spans: list[tuple[float, float]], min_s: float = 0.0,
            max_s: Optional[float] = None) -> list[tuple[float, float]]:
    """Apply the two guards every content-derived policy needs.

    Without them a content-derived grid is unusable in both directions. Voice
    activity cuts on every pause, which on conversational audio is every one
    or two seconds -- that would shred the video into chunks too short to
    describe and multiply the describer calls that dominate the cost. And a
    single unbroken monologue produces one chunk covering the whole file,
    which is no grid at all.

    ``min_s`` merges a short chunk into the one before it; ``max_s`` splits a
    long one on a plain grid. The scene chunker already carries the same pair
    (`--scene-min-duration`, `--chunk-duration`) for exactly these reasons.
    """
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and (end - start) < min_s:
            merged[-1] = (merged[-1][0], end)          # absorb into the previous
        else:
            merged.append((start, end))
    # Nothing precedes the first chunk, so a short one absorbs forwards instead.
    if min_s and len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_s:
        merged[:2] = [(merged[0][0], merged[1][1])]

    if not max_s:
        return merged
    # Split evenly rather than taking fixed `max_s` bites. Taking bites leaves
    # a remainder, and the remainder is routinely shorter than `min_s` -- so a
    # guard whose whole job is to prevent tiny chunks would create them. A
    # 62.5 s span at max 30 becomes three of 20.8 rather than 30, 30 and 2.5.
    out: list[tuple[float, float]] = []
    for start, end in merged:
        span = end - start
        if span <= max_s:
            out.append((start, end))
            continue
        pieces = int(-(-span // max_s))          # ceil, without importing math
        step = span / pieces
        edges = [start + step * i for i in range(pieces)] + [end]
        out.extend(zip(edges, edges[1:]))
    return out
