"""Chunk ids in, times out; and batching. Shared by the llm kinds.

Spans are resolved through the timeline rather than trusted from the model --
it is asked for chunk ids, which it can copy; times it would invent. Ids travel
with the text through every fold, so a merged span is the union of what it
merged.
"""

from __future__ import annotations

from typing import Any, Sequence

from .base import Context


def batched(items: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def resolve_span(context: Context, chunk_ids: Sequence[int]
                 ) -> tuple[float, float]:
    """The real span covering these chunks, from the grid.

    Never the model's own numbers: it is asked for chunk ids because those it
    can copy, and times it would invent.
    """
    valid = [c for c in chunk_ids if 0 <= c < len(context.timeline)]
    if not valid:
        return 0.0, 0.0
    return (min(context.span_of(c)[0] for c in valid),
            max(context.span_of(c)[1] for c in valid))


__all__ = ["batched", "resolve_span"]
