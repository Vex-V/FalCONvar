"""9 · the Aggregator protocol, and what one is given.

**Aggregates answer what retrieval cannot.** Embeddings cannot count, so "the
busiest moment", "who dominated", "how much of this is speech" are exact
questions similarity answers approximately. This reads the finished documents
-- never the video, never another component's modules -- and produces
video-level structure.

**A tier is a cost ceiling, and asking for a dear one still runs the cheap
ones.** `free` is arithmetic, `local` adds GPU models, `llm` adds paid calls.
They run cheapest first, so a run that dies partway has produced the free
results rather than none.

**`depends_on` drops rather than fails.** A dependency naming a source
(`yolo`, `transcript`) is a requirement on the video; one naming another
aggregator orders it first. `speakers` on silent CCTV is not an error, it is a
question that does not apply, and it is reported as skipped *with the reason* --
because "speakers did not run" is only useful beside why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

from ..shared.documents import (Descriptions, Manifest, Timeline, Transcript,
                                fingerprint_of)

#: Cheapest first. A tier is a ceiling, not a selection.
TIERS = ("free", "local", "llm")


@dataclass
class Context:
    """Every finished document, joined by `chunk_id`.

    The join is the point: a chunk is one thing, and what the picture said
    about it and what was said during it are two halves of the same record.
    """

    video_id: str
    timeline: Timeline
    manifest: Optional[Manifest] = None
    descriptions: Optional[Descriptions] = None
    transcript: Optional[Transcript] = None

    @property
    def sources(self) -> set[str]:
        """What this video actually has -- sampler ids, plus `transcript`.

        `depends_on` is checked against this, so a dependency is a question
        about the video rather than about the request.
        """
        found: set[str] = set()
        if self.descriptions is not None:
            for chunk in self.descriptions.chunks:
                found |= set(chunk.get("samplers", {}))
        if self.transcript is not None and any(
                c.get("word_count") for c in self.transcript.chunks):
            found.add("transcript")
        return found

    def chunk_ids(self) -> list[int]:
        return list(range(len(self.timeline)))

    def span_of(self, chunk_id: int) -> tuple[float, float]:
        return self.timeline.bounds_of(chunk_id)

    def text_of(self, chunk_id: int) -> dict[str, str]:
        """Everything said about one chunk, by sampler id."""
        out: dict[str, str] = {}
        if self.descriptions is not None:
            for chunk in self.descriptions.chunks:
                if chunk["chunk_id"] == chunk_id:
                    for sid, block in chunk.get("samplers", {}).items():
                        out[sid] = block.get("description", "")
        if self.transcript is not None:
            text = self.transcript.text_of(chunk_id)
            if text:
                out["transcript"] = text
        return out

    def inputs_fingerprint(self) -> str:
        """A hash of the chunk text actually read.

        A summary of descriptions that have since been rewritten reads
        perfectly, which is precisely why staleness cannot be left to a reader
        to notice.
        """
        return fingerprint_of({
            "timeline": self.timeline.fingerprint(),
            "text": {str(i): self.text_of(i) for i in self.chunk_ids()},
        })


class Aggregator(Protocol):
    name: str
    tier: str
    about: str
    depends_on: Sequence[str]

    def run(self, context: Context) -> dict[str, Any]: ...


def missing(aggregator: Any, context: Context,
            done: Sequence[str] = ()) -> Optional[str]:
    """Why this aggregator cannot run here, or None.

    A message rather than a boolean: "speakers did not run" is only useful
    beside why it did not.
    """
    for need in getattr(aggregator, "depends_on", ()):
        if need in done:
            continue
        if need in context.sources:
            continue
        if need in AGGREGATOR_NAMES:
            return f"needs {need}, which did not run"
        return f"needs {need}, which this video has no output for"
    return None


#: Which aggregators exist, for telling "depends on another aggregator" from
#: "depends on something the video has". Kept as data so `missing` can say
#: which kind a failed dependency was without importing anything.
AGGREGATOR_NAMES = frozenset({
    "stats", "speakers", "coverage", "novelty", "ner", "sentiment",
    "summary", "chapters", "events",
})


def resolve_order(names: Sequence[str], tier_of: dict[str, str]) -> list[str]:
    """Cheapest tier first, then dependencies before dependents.

    Takes a tier map rather than the classes, so ordering a `--tier free` run
    never imports the llm modules it is excluding.
    """
    ordered: list[str] = []
    for tier in TIERS:
        tier_names = [n for n in names if tier_of[n] == tier]
        remaining = list(tier_names)
        while remaining:
            progressed = False
            for name in list(remaining):
                deps = [d for d in _DEPENDS.get(name, ()) if d in names]
                if all(d in ordered or d not in remaining for d in deps):
                    ordered.append(name)
                    remaining.remove(name)
                    progressed = True
            if not progressed:                    # a cycle; run them anyway
                ordered.extend(remaining)
                break
    return ordered


#: Declared here rather than read off the classes, for the same reason as
#: `tier_of`: ordering must not import what it is ordering.
_DEPENDS: dict[str, tuple[str, ...]] = {
    "speakers": ("transcript",),
    "chapters": ("summary",),
}

__all__ = ["AGGREGATOR_NAMES", "TIERS", "Aggregator", "Context", "missing",
           "resolve_order"]
