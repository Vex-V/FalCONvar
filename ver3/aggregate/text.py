"""The llm tier: what the whole video is about, in chapters, and what happened.

Three aggregators, all reading the same rendering of the video and all paid.
They run last, after `free`, so a run that dies here has still produced the
arithmetic.

**Ask for a word count, not "several sentences".** Once structured fields
arrived, `falconvar`'s model sized its summary as one field among many: 105
median words against 363 in the prose-only era. Saying "at least 150 words",
and that the fields *index* the summary rather than replace it, took it back to
246 median words. The summary is the only text here that gets embedded, so its
length is a retrieval parameter rather than a style preference.

**Every fold is recorded, not just the last.** `summary` folds chunk lines in
batches, then folds the results again while more than a batch remains. Those
intermediate summaries used to be transient, which threw away a coarse account
of the video that had already been paid for -- a leaf summary covers a real
span and is the only description at that granularity, between one chunk and the
whole file. They are kept in `layers`, each part carrying its `chunk_ids` and
the span resolved from the grid.

**Spans are resolved, never trusted.** A model is asked for chunk ids, which it
can copy; times it would invent. `rendering.resolve_span` turns ids into the
grid's own numbers.
"""

from __future__ import annotations

from typing import Any, Optional

from ..shared.llm import DEFAULT_MODEL, LLMUnavailable, complete
from .base import Context
from .rendering import batched, chunk_rows, resolve_span

#: How many chunk lines go into one fold. Above this, `summary` folds twice.
BATCH = 25

SYSTEM = (
    "You summarise video content for a retrieval index. Report only what the "
    "supplied descriptions say. Do not speculate about intent or identity. "
    "Every chunk id you cite must be one that appears in the input."
)

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

_EVENTS_SCHEMA = {
    "name": "events",
    "schema": {
        "type": "object", "additionalProperties": False,
        "required": ["events"],
        "properties": {"events": {
            "type": "array",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["what", "chunk_id", "kind"],
                "properties": {
                    "what": {"type": "string"},
                    "chunk_id": {"type": "integer"},
                    "kind": {"type": "string",
                             "description": "action | arrival | change | speech"},
                },
            },
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


class ChaptersAggregator:
    name = "chapters"
    tier = "llm"
    about = "a table of contents: contiguous chapters over the whole video"
    depends_on = ("summary",)

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or DEFAULT_MODEL

    def run(self, context: Context) -> dict[str, Any]:
        rows = chunk_rows(context)
        if not rows:
            raise LLMUnavailable("nothing to divide into chapters")
        body = "\n".join(line for _, line in rows)
        answer = complete(
            "Divide this video into chapters. They must be contiguous and "
            "cover every chunk, and you must cite chunk ids from the input:\n\n"
            + body, _CHAPTERS_SCHEMA, self.model, SYSTEM)

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


class EventsAggregator:
    name = "events"
    tier = "llm"
    about = "discrete things that happened, each pinned to a chunk"
    depends_on: tuple[str, ...] = ()

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or DEFAULT_MODEL

    def run(self, context: Context) -> dict[str, Any]:
        rows = chunk_rows(context)
        if not rows:
            raise LLMUnavailable("nothing to find events in")
        body = "\n".join(line for _, line in rows)
        answer = complete(
            "List the discrete events in this video. Each must cite the chunk "
            "id it happened in, taken from the input:\n\n" + body,
            _EVENTS_SCHEMA, self.model, SYSTEM)

        events = []
        for event in answer["events"]:
            chunk_id = int(event["chunk_id"])
            start, end = resolve_span(context, [chunk_id])
            events.append({"what": event["what"], "kind": event["kind"],
                           "chunk_id": chunk_id,
                           "start_ts": round(start, 3),
                           "end_ts": round(end, 3)})
        events.sort(key=lambda e: e["start_ts"])
        return {"events": events, "count": len(events)}


__all__ = ["BATCH", "ChaptersAggregator", "EventsAggregator", "SummaryAggregator"]
