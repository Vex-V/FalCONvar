"""The `items` kind: discrete things, each pinned to a chunk. `events` is one.

Items are independent of each other, so a long video is asked in windows, all
at once, and the answers concatenated in time order.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .. import definitions
from ..rendering import batched, resolve_span
from ..base import WINDOW, DefinitionRunner, listing, schema


class ItemsAggregator(DefinitionRunner):
    async def _run(self, context: Any, read: Any) -> dict[str, Any]:
        key = self.entry.get("key") or "items"
        fields = list(self.entry["fields"])
        ask = f"{self.entry['instruction']} {definitions.kind_text('items')['cite']}"
        shape = schema(self.name, listing(key, {"chunk_id": {"type": "integer"},
                                                **self.properties()}))
        windows = batched(read.rows, WINDOW)
        answers = await asyncio.gather(*(
            self.llm.complete(f"{ask}\n\n" + "\n".join(r.line for r in window),
                              shape, definitions.system())
            for window in windows))

        found = []
        for answer in answers:
            for item in answer[key]:
                chunk_id = int(item["chunk_id"])
                start, end = resolve_span(context, [chunk_id])
                found.append({**{f: item.get(f) for f in fields}, "chunk_id": chunk_id,
                              "start_ts": round(start, 3), "end_ts": round(end, 3)})
        found.sort(key=lambda e: e["start_ts"])
        return {key: found, "count": len(found), "windows": len(windows)}


__all__ = ["ItemsAggregator"]
