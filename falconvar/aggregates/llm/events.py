"""Discrete things that happened, each pinned to a chunk."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from ...shared.models.llm import LLMUnavailable, Model
from ..base import Context
from ..rendering import chunk_rows, resolve_span
from . import SYSTEM

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


class EventsAggregator:
    name = "events"
    tier = "llm"
    about = "discrete things that happened, each pinned to a chunk"
    depends_on: tuple[str, ...] = ()

    def __init__(self, llm: Optional[str] = None) -> None:
        self.llm = Model(llm, role="llm")

    @property
    def model_key(self) -> str:
        return self.llm.key

    def run(self, context: Context) -> dict[str, Any]:
        rows = chunk_rows(context)
        if not rows:
            raise LLMUnavailable("nothing to find events in")
        body = "\n".join(line for _, line in rows)
        answer = asyncio.run(self.llm.complete(
            "List the discrete events in this video. Each must cite the chunk "
            "id it happened in, taken from the input:\n\n" + body,
            _EVENTS_SCHEMA, SYSTEM))

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


__all__ = ["EventsAggregator"]
