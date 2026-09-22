"""The `fold` kind: batch the chunks, summarise each batch, summarise the result.

`summary` is one. Every fold is recorded, not just the last: a leaf summary
covers a real span and is the only description at that granularity, between
one chunk and the whole file -- so `layers` keeps them, each with its chunk ids
and the span resolved from the grid.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ...shared.models.llm import Model
from .. import definitions
from ..rendering import batched, resolve_span
from ..base import BATCH, DefinitionRunner, schema


async def fold(context: Any, llm: Model, parts: list[tuple[list[int], str]],
               until: int, batch: int = BATCH, instruction: str = "",
               ) -> tuple[list[tuple[list[int], str]], list[dict[str, Any]]]:
    """Fold `(chunk_ids, text)` parts in batches until at most `until` remain.

    The folds of one layer are independent, so they are asked at once; the
    next layer waits for all of them. Ids travel with the text, so a merged
    part covers the union of what it merged.
    """
    text = definitions.kind_text("fold")
    ask = instruction or text["fold_instruction"]
    shape = schema("partial_summary", {"summary": {
        "type": "string", "description": text["fold_about"]}})
    layers: list[dict[str, Any]] = []
    while len(parts) > until:
        groups = batched(parts, batch)
        answers = await asyncio.gather(*(
            llm.complete(f"{ask}\n\n" + "\n".join(t for _, t in group), shape,
                         definitions.system())
            for group in groups))
        parts = [([c for part in group for c in part[0]], answer["summary"])
                 for group, answer in zip(groups, answers)]
        layers.append({"level": len(layers), "parts": [
            {"chunk_ids": ids, "summary": said,
             "start_ts": round(resolve_span(context, ids)[0], 3),
             "end_ts": round(resolve_span(context, ids)[1], 3)}
            for ids, said in parts]})
    return parts, layers


class FoldAggregator(DefinitionRunner):
    async def _run(self, context: Any, read: Any) -> dict[str, Any]:
        rows = read.rows
        parts, layers = await fold(
            context, self.llm, [([r.chunk_id], r.line) for r in rows],
            until=BATCH, instruction=self.entry.get("fold_instruction") or "")
        body = "\n".join(said for _, said in parts)
        final = await self.llm.complete(
            f"{self.entry['instruction']}\n\n{body}",
            schema(self.name, self.properties()), definitions.system(),
            max_output_tokens=4000)
        payload = {
            **final,
            "chunks_read": len(rows),
            # How much paraphrase sits between the descriptions and this text.
            # `[]` says truthfully that no intermediate summary existed, rather
            # than that one was discarded.
            "reduction_levels": len(layers),
            "layers": layers,
        }
        if isinstance(final.get("summary"), str):
            payload["word_count"] = len(final["summary"].split())
        return payload


__all__ = ["FoldAggregator", "fold"]
