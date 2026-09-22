"""The Aggregator protocol, and what one is given.

`Context` joins every finished document by `chunk_id` -- what the picture said
about a window and what was said during it are two halves of one record.

A tier is a cost ceiling: `free` is arithmetic, `local` adds GPU models, `llm`
adds paid calls. They run cheapest first, so a run that dies partway has
produced the free results rather than none.

**Two kinds of aggregator.** `stats`, `speakers` and `coverage` count what
extraction produced and take no input. Everything else reads an *input* -- a
selection over the documents, see `inputs` -- and answers once per input:
`ner`, `sentiment`, and every prompt and link profile in `definitions`.

`depends_on` names a source the video must have (`transcript`). A question that
does not apply is reported as skipped with the reason, never as a failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

from ..shared.contracts.documents import (Descriptions, Manifest, Timeline, Transcript,
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
        """What this video actually has -- answer ids, plus `transcript`."""
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
        """Everything said about one chunk, by answer id."""
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
        """A hash of every chunk's text, for the aggregators that take no input.

        An aggregator that reads an input is fingerprinted on what that input
        read instead -- `inputs.Read.fingerprint`.
        """
        return fingerprint_of({
            "timeline": self.timeline.fingerprint(),
            "text": {str(i): self.text_of(i) for i in self.chunk_ids()},
        })


class Aggregator(Protocol):
    """Counts what extraction produced. Takes no input."""

    name: str
    tier: str
    about: str
    depends_on: Sequence[str]

    def run(self, context: Context) -> dict[str, Any]: ...


class Reader(Protocol):
    """Answers once per input.

    `read` is separate from `run` so the driver can fingerprint what would be
    read -- and reuse a stored answer -- before paying for anything.
    """

    name: str
    tier: str
    about: str
    depends_on: Sequence[str]
    takes_inputs: bool
    #: What the answer depends on besides the text read: a prompt's version, a
    #: model's configuration.
    version: str

    def read(self, context: Context, one: Any) -> Any: ...

    def run(self, context: Context, read: Any) -> dict[str, Any]: ...


def missing(aggregator: Any, context: Context) -> Optional[str]:
    """Why this aggregator cannot run here, or None.

    A message rather than a boolean: "speakers did not run" is only useful
    beside why it did not.
    """
    for need in getattr(aggregator, "depends_on", ()):
        if need not in context.sources:
            return f"needs {need}, which this video has no output for"
    return None


__all__ = ["TIERS", "Aggregator", "Context", "Reader", "missing"]
