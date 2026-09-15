"""The `spans` kind: contiguous ranges over the whole video. `chapters` is one.

Spans are resolved through the timeline, never trusted from the model. It is
asked for chunk ids, which it can copy; times it would invent.

**A video too long for one call is divided by parts, not windows.** Dividing
each window separately would force a chapter break at every window edge. So
the chunks are folded first -- as `summary` folds them -- until the parts fit,
and the model cites part ids, each resolved to the chunks it covers.
"""

from __future__ import annotations

from typing import Any

from .. import definitions
from ..rendering import resolve_span
from . import WINDOW, DefinitionRunner, listing, schema
from .fold import fold


class SpansAggregator(DefinitionRunner):
    async def _run(self, context: Any, read: Any) -> dict[str, Any]:
        text = definitions.kind_text("spans")
        key = self.entry.get("key") or "spans"
        fields = list(self.entry["fields"])
        units = [([r.chunk_id], r.line) for r in read.rows]
        layers: list[dict[str, Any]] = []
        by_parts = len(units) > WINDOW
        if by_parts:
            units, layers = await fold(context, self.llm, units, until=WINDOW)
            lines = []
            for number, (ids, said) in enumerate(units):
                start, end = resolve_span(context, ids)
                lines.append(f"[{number}] {start:.0f}-{end:.0f}s  {said}")
        else:
            lines = [said for _, said in units]

        item = {"first_chunk": {"type": "integer"}, "last_chunk": {"type": "integer"},
                **self.properties()}
        answer = await self.llm.complete(
            f"{self.entry['instruction']} {text['cite_parts' if by_parts else 'cite']}"
            "\n\n" + "\n".join(lines),
            schema(self.name, listing(key, item)), definitions.system())

        spans = []
        for span in answer[key]:
            first, last = int(span["first_chunk"]), int(span["last_chunk"])
            ids = (sorted({c for ids, _ in units[max(first, 0):last + 1] for c in ids})
                   if by_parts else list(range(first, last + 1)))
            start, end = resolve_span(context, ids)
            spans.append({**{f: span.get(f) for f in fields}, "chunk_ids": ids,
                          "start_ts": round(start, 3), "end_ts": round(end, 3)})
        covered = {c for span in spans for c in span["chunk_ids"]}
        return {key: spans,
                "count": len(spans),
                # Reported rather than repaired: a gap means the model did not do
                # what it was asked, and hiding that would make the next reader
                # trust a table of contents that skips a minute.
                "covers_all_chunks": covered >= set(context.chunk_ids()),
                "uncovered": sorted(set(context.chunk_ids()) - covered),
                "divided": "parts" if by_parts else "chunks",
                **({"layers": layers} if layers else {})}


__all__ = ["SpansAggregator"]
