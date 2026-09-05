"""The discrete things that happen, and where.

A chapter says what a stretch is about; an event says what *occurred* -- someone
arriving, a transaction completing, an object changing hands, a reactor
exploding. The distinction matters because the two answer different questions:
"what is this part about" against "when did X happen".

**One entry per event, not one per chunk.** A chunk where nothing discrete
happens produces nothing, and a chunk where three things happen produces three.
That is the difference between an event list and a rephrased description, and
it is what the prompt spends most of its words on.

Events are placed at a **chunk**, not a timestamp. The descriptions they are
drawn from cover a whole window and never say which second within it, so a
precise time would be invented. The chunk is the finest position the evidence
actually supports.
"""

from __future__ import annotations

import collections
from typing import Any, Optional

from ver2.llm import DEFAULT_MODEL

from .base import Context
from .llm import SYSTEM, chunk_lines, complete, pick_sources, resolve_span

#: People and objects first: a discrete happening is usually somebody doing
#: something, so the specialists that name actors lead over the scene account.
PREFERENCE = ("yolo", "objects", "overview", "clip", "uniform", "transcript", "text")

SCHEMA = {
    "name": "video_events",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["events"],
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["event", "chunk_id", "actor", "category"],
                    "properties": {
                        "event": {"type": "string",
                                  "description": "What happened, one short sentence."},
                        "chunk_id": {"type": "integer",
                                     "description": "The chunk the event occurs in."},
                        "actor": {"type": "string",
                                  "description": ("Who or what did it, described as the "
                                                  "text describes them. Empty if unclear. "
                                                  "Never a name or an identity.")},
                        "category": {"type": "string",
                                     "description": ("Kind of event: arrival, departure, "
                                                     "transaction, interaction, movement, "
                                                     "speech, incident, or other.")},
                    },
                },
            }
        },
    },
}

PROMPT = """\
Below are consecutive segments of one video, each with its chunk id and \
timecode.

Extract the discrete events: things that occur at a point in time, such as \
someone arriving or leaving, a transaction completing, an object changing \
hands, a notable interaction, or an incident.

Rules:
- Only report events the text below actually supports.
- One entry per event, not one per chunk. Skip chunks where nothing discrete \
happens.
- Use the chunk ids exactly as given; do not invent ids.
- Describe the actor as the text describes them. Never guess a name, an age or \
an identity.

{lines}"""


class EventsAggregator:
    """A timeline of discrete happenings, each anchored to a chunk."""

    id = "events"
    tier = "llm"
    depends_on = ()

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or DEFAULT_MODEL

    def aggregate(self, ctx: Context) -> Optional[dict[str, Any]]:
        sources = pick_sources(ctx, PREFERENCE)
        lines = chunk_lines(ctx, sources, limit=300)
        if not lines:
            return None

        result = complete(PROMPT.format(lines="\n".join(lines)),
                          schema=SCHEMA, model=self.model, system=SYSTEM)

        events, dropped = [], 0
        for event in result.get("events", []):
            span = resolve_span(ctx, event.get("chunk_id"))
            if span is None:
                dropped += 1
                continue
            start, end, _ = span
            events.append({
                "event": event["event"],
                "actor": (event.get("actor") or "").strip() or None,
                "category": (event.get("category") or "other").strip().lower(),
                "chunk_id": event["chunk_id"],
                "start_ts": round(start, 3),
                "end_ts": round(end, 3),
            })
        if not events:
            return None
        events.sort(key=lambda e: (e["start_ts"], e["event"]))

        categories = collections.Counter(e["category"] for e in events)
        return {"events": events, "based_on": sources, "dropped": dropped,
                "count": len(events),
                "categories": dict(categories.most_common()),
                # Where events cluster. A count per chunk is the cheap answer
                # to "when was the most going on", which is a different
                # question from novelty and often a better one.
                "per_chunk": dict(collections.Counter(e["chunk_id"] for e in events))}

    def config(self) -> dict[str, Any]:
        return {"id": self.id, "tier": self.tier, "params": {"model": self.model}}
