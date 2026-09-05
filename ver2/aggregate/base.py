"""What an aggregator is, and what it gets to work with.

Every other stage here answers a question about a *chunk*. This one answers
questions about a *video*: how busy was it, who spoke most, what happens in it,
which twenty seconds are unlike the rest. Retrieval is bad at all of those --
embeddings cannot count, and "the busiest moment" is an exact question that
similarity answers approximately.

**An aggregator reads documents, never modules.** `descriptions.json`,
`transcript.json` and `timeline.json` arrive as parsed JSON, exactly as the
manifest does for `describe`. That is what keeps this stage from depending on
how the video or audio passes work rather than on what they produced, and it
is why re-running the whole set costs nothing but the aggregators' own work.

``depends_on`` names either a **source** -- a sampler id, or `transcript` --
or another aggregator. The first decides whether it can run at all, the second
decides order. An aggregator whose sources are absent is dropped rather than
run against nothing, because a summary of no input is not a summary.

``tier`` is what it costs, and it is metadata rather than structure:

    free    arithmetic over what is already written. No model, no network.
    local   a model on this machine's GPU.
    llm     a paid API call.

A caller can ask for a tier and get everything at or below it, which is how a
free pass over a hundred videos stays free.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

TIERS = ("free", "local", "llm")


@dataclass
class Context:
    """One video's finished output, joined into the shape aggregators want.

    Built once per run and shared by every aggregator in it, so the documents
    are parsed once however many read them.
    """

    video_id: str
    descriptions: Optional[dict[str, Any]] = None
    transcript: Optional[dict[str, Any]] = None
    timeline: Optional[dict[str, Any]] = None
    manifest: Optional[dict[str, Any]] = None
    out_dir: Optional[Path] = None
    #: Set by the reader when an index is available. Only `novelty` reads it.
    index: Any = None
    embedder: Any = None

    @classmethod
    def from_dir(cls, video_id: str, out_dir: Path) -> "Context":
        """Read whatever a video has produced. Missing documents are None."""
        def load(name: str) -> Optional[dict]:
            path = Path(out_dir) / f"{name}.json"
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))

        return cls(video_id=video_id, out_dir=Path(out_dir),
                   descriptions=load("descriptions"), transcript=load("transcript"),
                   timeline=load("timeline"), manifest=load("manifest"))

    # ---------------------------------------------------------------- sources
    @property
    def samplers(self) -> list[str]:
        """Which questions were asked of this video's pictures."""
        seen: list[str] = []
        for chunk in (self.descriptions or {}).get("chunks", []):
            for sampler in chunk.get("samplers", {}):
                if sampler not in seen:
                    seen.append(sampler)
        return seen

    @property
    def sources(self) -> list[str]:
        """Every input an aggregator could name in `depends_on`."""
        out = list(self.samplers)
        if self.has_speech:
            out.append("transcript")
        return out

    @property
    def has_speech(self) -> bool:
        return any((c.get("text") or "").strip()
                   for c in (self.transcript or {}).get("chunks", []))

    def has(self, source: str) -> bool:
        return source in self.sources

    # ----------------------------------------------------------------- chunks
    @property
    def chunks(self) -> list[dict[str, Any]]:
        """One row per chunk of the shared grid, both modalities attached.

        The join every aggregator would otherwise write for itself. `chunk_id`
        means the same thing in both documents by construction -- that is what
        the shared timeline is for -- so this is a lookup rather than a match.
        """
        if self._joined is not None:
            return self._joined

        spoken = {c["chunk_id"]: c
                  for c in (self.transcript or {}).get("chunks", [])}
        described = {c["chunk_id"]: c
                     for c in (self.descriptions or {}).get("chunks", [])}
        grid = {c["chunk_id"]: c for c in (self.timeline or {}).get("chunks", [])}

        rows = []
        for chunk_id in sorted(set(spoken) | set(described) | set(grid)):
            bounds = (grid.get(chunk_id) or described.get(chunk_id)
                      or spoken.get(chunk_id) or {})
            said = spoken.get(chunk_id) or {}
            rows.append({
                "chunk_id": chunk_id,
                "start_ts": float(bounds.get("start_ts", 0.0)),
                "end_ts": float(bounds.get("end_ts", 0.0)),
                "descriptions": (described.get(chunk_id) or {}).get("samplers", {}),
                "transcript": said.get("text") or "",
                "turns": (said.get("structured") or {}).get("turns", []),
                "speakers": (said.get("structured") or {}).get("speakers", []),
                "word_count": said.get("word_count", 0),
            })
        self._joined = rows
        return rows

    _joined: Optional[list] = field(default=None, repr=False, compare=False)

    @property
    def duration(self) -> float:
        rows = self.chunks
        return rows[-1]["end_ts"] if rows else 0.0

    def text_of(self, chunk: dict, sources: Optional[list[str]] = None) -> str:
        """One chunk's text from the named sources, best-effort and in order."""
        parts = []
        for source in sources or (self.samplers + ["transcript"]):
            if source == "transcript":
                if chunk["transcript"]:
                    parts.append(chunk["transcript"])
            else:
                block = chunk["descriptions"].get(source) or {}
                if block.get("description"):
                    parts.append(block["description"])
        return " ".join(parts).strip()

    # ------------------------------------------------------------- staleness
    def fingerprint(self, sources: Optional[list[str]] = None) -> str:
        """A hash of the text an aggregator will actually read.

        Recorded on every result, because an aggregate derived from
        descriptions that have since been rewritten is not wrong in any visible
        way -- a stale summary reads perfectly. This makes it a comparison,
        the way `manifest_fingerprint` does for descriptions and `text_hash`
        for vectors, rather than something a reader has to assume.
        """
        payload = json.dumps(
            [[c["chunk_id"], round(c["start_ts"], 3), round(c["end_ts"], 3),
              self.text_of(c, sources)] for c in self.chunks],
            sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@runtime_checkable
class Aggregator(Protocol):
    """One video-level pass over what the chunk stages wrote."""

    id: str
    tier: str
    depends_on: tuple[str, ...]

    def aggregate(self, ctx: Context) -> Optional[dict[str, Any]]:
        """The video-level result, or None when the inputs cannot support one.

        None is the honest answer for "this video has no speech, so there are
        no speaker statistics" -- distinct from an empty result, which would
        claim the question was asked and answered.
        """
        ...

    def config(self) -> dict[str, Any]:
        """Recorded beside the result, so it says what produced it."""
        ...
