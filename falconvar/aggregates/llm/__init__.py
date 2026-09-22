"""The llm tier: paid calls over what an input read.

Runs last, after `statistics` and `model`, so a run that dies here has still
produced the arithmetic.

One runner per kind -- `fold`, `spans`, `items`, and `entities` for link
profiles -- each constructed from a definition in `definitions.json` or
`data/aggregates.json`. `summary` is not a class; it is a fold.

**Ask for a word count, not "several sentences".** Once structured fields
arrived, the model sized its summary as one field among many: 105 median words
against 363 in the prose-only era. Saying "at least 150 words", and that the
fields *index* the summary rather than replace it, took it back to 246. The
summary is the only text here that gets embedded, so its length is a retrieval
parameter rather than a style preference.

**Spans are resolved, never trusted.** A model is asked for chunk ids, which it
can copy; times it would invent.

**Nothing is cut silently.** There was a 400-row cap on what any of these read.
A fold batches without one; `items` asks per window, concurrently; `spans`
divides folded parts when the chunks will not fit one call.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from .. import definitions, inputs

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
    """What every kind shares: the definition, its version, and who answers."""

    tier = "llm"
    depends_on: tuple[str, ...] = ()
    takes_inputs = True

    def __init__(self, definition_id: str, llm: Optional[str] = None) -> None:
        from ...shared.models.llm import Model
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
        """The definition's own fields, compiled by describe's builder."""
        from ...video_rag import driver as video_rag
        return video_rag.answer_schema(self.entry["fields"])

    def read(self, context: Any, one: inputs.Input) -> Any:
        return inputs.read(context, one)

    def run(self, context: Any, read: Any) -> dict[str, Any]:
        return asyncio.run(self._run(context, read))

    async def _run(self, context: Any, read: Any) -> dict[str, Any]:
        raise NotImplementedError


__all__ = ["BATCH", "WINDOW", "DefinitionRunner", "listing", "schema"]
