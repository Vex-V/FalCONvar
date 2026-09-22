"""How what `inputs` read reaches an aggregator: times, batches and pieces.

Spans are resolved through the timeline rather than trusted from the model --
it is asked for chunk ids, which it can copy; times it would invent. Ids travel
with the text through every fold, so a merged span is the union of what it
merged.

`batched` divides for the llm kinds, `pieces` for the local models, and the
difference is what the limit means: a batch is how much one paid call should
carry, a piece is how much a checkpoint can physically read.
"""

from __future__ import annotations

import re
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


def plain(row: Any) -> str:
    """A row's text without labels or times: what a model should read."""
    return " ".join(said for _, said in row.parts)


def pieces(text: str, limit: int) -> list[str]:
    """Text cut at sentence ends -- then at spaces -- into pieces of at most
    `limit` characters. Nothing is dropped.

    **Long text is cut, never truncated.** Each local model has a length it
    reads, and past it the rest is silently ignored: a sentiment model scoring
    a whole 20-second chunk reported the tone of its first clause and called it
    the chunk's. The answer is taken over the pieces instead.
    """
    out: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if current:
                out.append(current)
                current = ""
            out.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= limit:
            current = f"{current} {sentence}"
        else:
            out.append(current)
            current = sentence
    if current:
        out.append(current)
    return [p for p in out if p]


__all__ = ["batched", "pieces", "plain", "resolve_span"]
