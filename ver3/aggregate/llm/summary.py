"""What the whole video is about, in one pass over every chunk.

Every fold is recorded, not just the last. A leaf summary covers a real span
and is the only description at that granularity, between one chunk and the
whole file -- so `layers` keeps them, each carrying its chunk_ids and the span
resolved from the grid."""

from __future__ import annotations

from typing import Any, Optional

from ...shared.llm import DEFAULT_MODEL, LLMUnavailable, complete
from ..base import Context
from ..rendering import batched, chunk_rows, resolve_span
from . import BATCH, SYSTEM

_SUMMARY_SCHEMA = {
    "name": "video_summary",
    "schema": {
        "type": "object", "additionalProperties": False,
        "required": ["summary", "topics", "setting", "notable"],
        "properties": {
            "summary": {
                "type": "string",
                "description": ("What the whole video is about, in at least "
                                "150 words. The other fields index this "
                                "summary rather than replacing it, so do not "
                                "shorten it because they exist."),
            },
            "topics": {"type": "array", "items": {"type": "string"},
                       "description": "Recurring subjects, as short phrases."},
            "setting": {"type": "string",
                        "description": "Where this takes place, in one line."},
            "notable": {"type": "array", "items": {"type": "string"},
                        "description": "Things a viewer would remember."},
        },
    },
}

_FOLD_SCHEMA = {
    "name": "partial_summary",
    "schema": {
        "type": "object", "additionalProperties": False,
        "required": ["summary"],
        "properties": {"summary": {
            "type": "string",
            "description": ("What this stretch of the video covers, in at "
                            "least 80 words, in order."),
        }},
    },
}


class SummaryAggregator:
    name = "summary"
    tier = "llm"
    about = "what the whole video is about, in one pass over every chunk"
    depends_on: tuple[str, ...] = ()

    def __init__(self, model: Optional[str] = None, batch: int = BATCH) -> None:
        self.model = model or DEFAULT_MODEL
        self.batch = batch

    def run(self, context: Context) -> dict[str, Any]:
        rows = chunk_rows(context)
        if not rows:
            raise LLMUnavailable("nothing to summarise: no chunk has any text")

        layers: list[dict[str, Any]] = []
        level = 0
        # `(chunk_ids, text)`. Ids travel with the text through every fold, so
        # a merged span is the union of what it merged.
        parts = [([cid], line) for cid, line in rows]

        while len(parts) > self.batch:
            folded: list[tuple[list[int], str]] = []
            for group in batched(parts, self.batch):
                ids = [c for part in group for c in part[0]]
                body = "\n".join(text for _, text in group)
                answer = complete(
                    f"Summarise this stretch of a video, in order:\n\n{body}",
                    _FOLD_SCHEMA, self.model, SYSTEM)
                folded.append((ids, answer["summary"]))
            start, end = 0.0, 0.0
            layers.append({
                "level": level,
                "parts": [{"chunk_ids": ids, "summary": text,
                           "start_ts": round(resolve_span(context, ids)[0], 3),
                           "end_ts": round(resolve_span(context, ids)[1], 3)}
                          for ids, text in folded],
            })
            parts = folded
            level += 1

        body = "\n".join(text for _, text in parts)
        final = complete(
            f"Summarise this whole video:\n\n{body}",
            _SUMMARY_SCHEMA, self.model, SYSTEM, max_output_tokens=4000)
        return {
            **final,
            "chunks_read": len(rows),
            # How much paraphrase sits between the descriptions and this text.
            # `[]` says truthfully that no intermediate summary existed, rather
            # than that one was discarded.
            "reduction_levels": level,
            "layers": layers,
            "word_count": len(final["summary"].split()),
        }


__all__ = ["SummaryAggregator"]
