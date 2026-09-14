"""A table of contents: contiguous chapters over the whole video.

Spans are resolved through the timeline, never trusted from the model. It is
asked for chunk ids, which it can copy; times it would invent."""

from __future__ import annotations

from typing import Any, Optional

from ...shared.llm import LLMUnavailable, Model
from ..base import Context
from ..rendering import chunk_rows, resolve_span
from . import SYSTEM

_CHAPTERS_SCHEMA = {
    "name": "chapters",
    "schema": {
        "type": "object", "additionalProperties": False,
        "required": ["chapters"],
        "properties": {"chapters": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["title", "summary", "first_chunk", "last_chunk"],
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    # Ids, not times. See the module docstring.
                    "first_chunk": {"type": "integer"},
                    "last_chunk": {"type": "integer"},
                },
            },
        }},
    },
}


class ChaptersAggregator:
    name = "chapters"
    tier = "llm"
    about = "a table of contents: contiguous chapters over the whole video"
    depends_on = ("summary",)

    def __init__(self, provider: Optional[str] = None,
                 model: Optional[str] = None) -> None:
        self.llm = Model(provider, model, role="llm")

    @property
    def model_key(self) -> str:
        return self.llm.key

    def run(self, context: Context) -> dict[str, Any]:
        rows = chunk_rows(context)
        if not rows:
            raise LLMUnavailable("nothing to divide into chapters")
        body = "\n".join(line for _, line in rows)
        answer = self.llm.complete(
            "Divide this video into chapters. They must be contiguous and "
            "cover every chunk, and you must cite chunk ids from the input:\n\n"
            + body, _CHAPTERS_SCHEMA, SYSTEM)

        chapters = []
        for chapter in answer["chapters"]:
            ids = list(range(int(chapter["first_chunk"]),
                             int(chapter["last_chunk"]) + 1))
            start, end = resolve_span(context, ids)
            chapters.append({"title": chapter["title"],
                             "summary": chapter["summary"],
                             "chunk_ids": ids,
                             "start_ts": round(start, 3),
                             "end_ts": round(end, 3)})
        covered = {c for chapter in chapters for c in chapter["chunk_ids"]}
        return {"chapters": chapters,
                "count": len(chapters),
                # Reported rather than repaired: a gap means the model did not
                # do what it was asked, and hiding that would make the next
                # reader trust a table of contents that skips a minute.
                "covers_all_chunks": covered >= set(context.chunk_ids()),
                "uncovered": sorted(set(context.chunk_ids()) - covered)}


__all__ = ["ChaptersAggregator"]
