"""Evidence plus a duration, into spans.

The only place boundaries are decided. `uniform` needs no evidence at all --
it is arithmetic over the container duration.

Both guards are mandatory for a content-derived policy. Voice activity cuts on
every pause, which would shred the video into chunks too short to describe; a
monologue yields zero cuts and one chunk covering the file. `min_s` merges,
`max_s` splits.

`max_s` splits evenly, not into fixed bites: 30 s bites off a 62.5 s span leave
a 2.5 s remainder, so a guard against short chunks would create one.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..shared.contracts.documents import Timeline

#: Every policy, and which evidence each needs. The dependency, not a branch:
#: a driver reads this to know what must run first, rather than carrying an
#: ordering rule of its own.
POLICIES: dict[str, Optional[str]] = {
    "uniform": None,           # arithmetic over media.json. Nothing runs first.
    "scene": "video",          # boundaries.scenes -> cuts.json
    "vad": "audio",            # boundaries.speech reading transcript.raw.json
    "speaker": "audio",
}

#: A final chunk shorter than this fraction of the nominal length is merged
#: into the one before it, under every policy.
#:
#: A grid divides the media wherever it happens to end, so the tail is
#: uniformly distributed over the chunk length -- 97.99 s at 20 s leaves a
#: usable 17.99 s, but 100.4 s leaves 0.40 s. That stub is not harmless: it
#: costs a describer call *per sampler*, it keeps a frame because every chunk
#: keeps at least one, and it is a moment retrieval can return that nobody can
#: play. A fraction rather than a number of seconds, so the rule holds at any
#: chunk length.
MIN_TAIL_FRACTION = 0.25


def merge_tail(spans: list[tuple[float, float]],
               min_tail_s: float) -> list[tuple[float, float]]:
    """Absorb a too-short final chunk into the one before it.

    Only the last, and only backwards into its predecessor. A short chunk
    anywhere else is a content boundary something decided on purpose; the last
    is an artifact of where the media stopped relative to a grid that knew
    nothing about it.
    """
    if len(spans) < 2:
        return spans
    start, end = spans[-1]
    if end - start >= min_tail_s:
        return spans
    return spans[:-2] + [(spans[-2][0], end)]


def enforce(spans: list[tuple[float, float]], min_s: float = 0.0,
            max_s: Optional[float] = None) -> list[tuple[float, float]]:
    """The two guards every content-derived policy needs."""
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and (end - start) < min_s:
            merged[-1] = (merged[-1][0], end)      # absorb into the previous
        else:
            merged.append((start, end))
    # Nothing precedes the first chunk, so a short one absorbs forwards.
    if min_s and len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_s:
        merged[:2] = [(merged[0][0], merged[1][1])]

    if not max_s:
        return merged

    out: list[tuple[float, float]] = []
    for start, end in merged:
        span = end - start
        if span <= max_s:
            out.append((start, end))
            continue
        pieces = int(-(-span // max_s))             # ceil, without importing math
        step = span / pieces
        edges = [start + step * i for i in range(pieces)] + [end]
        out.extend(zip(edges, edges[1:]))
    return out


def uniform(video_id: str, duration_s: float, chunk_s: float = 20.0) -> Timeline:
    """A fixed grid, truncated to where the media ends.

    The truncation is not tidiness: telling a describer that a window is twenty
    seconds longer than the footage is telling it something false about every
    video.
    """
    if chunk_s <= 0:
        raise ValueError("chunk_s must be positive")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    spans: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_s:
        spans.append((start, min(start + chunk_s, duration_s)))
        start += chunk_s
    spans = merge_tail(spans or [(0.0, duration_s)], chunk_s * MIN_TAIL_FRACTION)
    return Timeline(video_id, spans, "uniform", {"chunk_s": chunk_s}, "grid")


def from_cuts(video_id: str, cuts: Sequence[float], duration_s: float,
              policy: str, params: Optional[dict[str, Any]] = None,
              derived_from: str = "grid",
              min_s: float = 5.0,
              max_s: Optional[float] = 30.0) -> Timeline:
    """Interior cut times plus a total length, guarded, as a Timeline.

    A policy that finds nothing to cut on returns one chunk covering the file,
    which `max_s` then divides evenly. That is the honest outcome for a
    monologue asked to be split on speaker changes -- there are none, and
    inventing some would be worse than saying so through the grid.

    The last span always reaches ``duration_s``, which is the *container's*
    duration rather than either stream's. A grid built from the audio alone
    leaves the last video frames outside every chunk: the streams differ by
    16 ms on the reference file, and a file is as long as its longest.
    """
    edges = [0.0] + sorted(t for t in cuts if 0.0 < t < duration_s) + [duration_s]
    spans = [(a, b) for a, b in zip(edges, edges[1:]) if b > a]
    spans = enforce(spans or [(0.0, duration_s)], min_s, max_s)
    spans = merge_tail(spans, min_s)
    settings = {"min_s": min_s, "max_s": max_s, **(params or {})}
    return Timeline(video_id, spans, policy, settings, derived_from)


def build(video_id: str, policy: str, duration_s: float,
          cuts: Optional[Sequence[float]] = None,
          derived_from: str = "grid",
          chunk_s: float = 20.0, min_s: float = 5.0,
          max_s: Optional[float] = None,
          params: Optional[dict[str, Any]] = None) -> Timeline:
    """One grid, whatever the policy.

    `max_s` defaults to ``chunk_s`` for a content-derived policy, so "how long
    should a chunk be" is one number rather than two that can disagree.
    """
    if policy not in POLICIES:
        raise KeyError(f"unknown policy {policy!r}; "
                       f"known: {', '.join(POLICIES)}")
    if policy == "uniform":
        return uniform(video_id, duration_s, chunk_s)
    if cuts is None:
        raise ValueError(
            f"policy {policy!r} needs cuts from "
            f"boundaries.{'scenes' if POLICIES[policy] == 'video' else 'speech'}")
    return from_cuts(video_id, cuts, duration_s, policy, params,
                     derived_from, min_s, max_s if max_s is not None else chunk_s)


__all__ = ["POLICIES", "MIN_TAIL_FRACTION", "merge_tail", "enforce", "uniform",
           "from_cuts", "build"]
