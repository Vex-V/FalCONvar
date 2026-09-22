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

**`DefinitionRunner` is here rather than beside the kinds that subclass it**,
for the reason `samplers/base.py` holds `Sampler`: a base class in a package's
`__init__` is reached by importing the package, so every kind that inherits it
drags in its siblings. One aggregator folder imports one module here instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

from ..shared.contracts.documents import (Descriptions, Manifest, Timeline, Transcript,
                                          fingerprint_of)
#: Re-exported so the local tier raises a name from its own namespace, while
#: `except ModelUnavailable` covers audio's too. They were separate classes
#: with the same name, so it silently covered only one.
from ..shared.errors import ModelUnavailable

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


#: How many lines go into one fold.
BATCH = 25

#: Lines one `spans` or `items` call reads. Above it, `items` asks per window
#: and `spans` divides folded parts instead of chunks.
WINDOW = 100


def schema(name: str, properties: dict[str, Any]) -> dict[str, Any]:
    """`{name, schema}` for a strict structured call: every key required."""
    return {"name": name.replace(":", "_").replace("~", "_"),
            "schema": {"type": "object", "additionalProperties": False,
                       "required": list(properties), "properties": properties}}


def listing(key: str, item: dict[str, Any]) -> dict[str, Any]:
    """A property holding a list of objects, every key required."""
    return {key: {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": list(item), "properties": item}}}


class DefinitionRunner:
    """The second kind of aggregator: one built from a definition.

    `summary`, `chapters` and `events` are not classes -- they are entries in
    `definitions.json`, and this is what runs one. A subclass is a *kind*
    (`fold`, `spans`, `items`, `link`), so adding a prompt adds no code.

    Everything a kind needs from the definition is resolved here: its fields,
    its version, and who answers it. `_run` is the only thing a kind writes.
    """

    tier = "llm"
    depends_on: tuple[str, ...] = ()
    takes_inputs = True

    def __init__(self, definition_id: str, llm: Optional[str] = None) -> None:
        from ..shared.models.llm import Model
        from . import definitions
        self.name = definition_id
        self.section, self.definition = definitions.locate(definition_id)
        self.entry = definitions.get(self.section, self.definition)
        self.about = self.entry.get("about", "")
        self.version = definitions.version_of(self.section, self.definition)
        self.llm = Model(llm, role="llm")

    @property
    def model_key(self) -> str:
        return self.llm.key

    def properties(self) -> dict[str, Any]:
        """The definition's own fields, compiled by the shared builder -- the
        same one a describe shape compiles through, so an aggregate's answer
        and a description's are built by one set of rules."""
        from ..shared.contracts.fields import compile_fields
        return compile_fields(self.entry["fields"])

    def read(self, context: Any, one: Any) -> Any:
        from .inputs import read as read_input
        return read_input(context, one)

    def run(self, context: Any, read: Any) -> dict[str, Any]:
        import asyncio
        return asyncio.run(self._run(context, read))

    async def _run(self, context: Any, read: Any) -> dict[str, Any]:
        raise NotImplementedError


__all__ = ["BATCH", "TIERS", "WINDOW", "Aggregator", "Context", "DefinitionRunner",
           "ModelUnavailable", "Reader", "listing", "missing", "schema"]
