"""One rendering of the video, shared by every text aggregator.

Each of `summary`, `chapters` and `events` needs the same thing: the video as
lines of text, one per chunk, in order, with the chunk ids attached. Written
once here so three aggregators cannot each format it slightly differently and
then disagree about which chunk a sentence came from.

**Ids travel with the text.** `chunk_rows` returns `(chunk_id, line)` rather
than a formatted string, so a span is the union of the ids that went into it
and is never reconstructed afterwards by parsing a string this module formatted
itself. `falconvar` learned this on the summary folds: a merge summary's span
has to be the union of what it merged.

**Spans are resolved through the timeline, not trusted from the model.** A
model asked for a chapter boundary returns a plausible number; the timeline
knows the real one. So a model names chunk ids and this converts them.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .base import Context

#: The order accounts are *listed* in, not a choice between them. Speech first
#: because it is usually the most informative thing about a chunk, then prose
#: about the picture, then whatever else described it.
PREFERENCE = ("transcript", "clip", "uniform")


def pick_sources(context: Context,
                 preference: Sequence[str] = PREFERENCE) -> list[str]:
    """Which sampler ids to read, best first, falling back to whatever exists.

    A video with only `yolo` still gets summarised -- from the people
    descriptions, which is worse than a scene account and much better than
    refusing.
    """
    have = context.sources
    chosen = [name for name in preference if name in have]
    # A paired sampler asking a preferred question counts too: `uniform:overview`
    # is prose about the whole frame, whatever strategy chose the frames.
    chosen += sorted(s for s in have
                     if s not in chosen and ":" in s
                     and s.split(":", 1)[1] in preference)
    return chosen or sorted(have)


def chunk_rows(context: Context, sources: Optional[Sequence[str]] = None,
               limit: int = 400) -> list[tuple[int, str]]:
    """`(chunk_id, line)` for every chunk that has anything to say.

    **Every account of a chunk, not the first one found.** Taking only the
    best-ranked source throws the other modality away: on a narrated video the
    picture and the soundtrack are two independent accounts of the same
    seconds, and a summary built from one of them is missing half of what is
    known. Measured the hard way -- a run here picked stub `clip` text over 428
    words of real narration and the model correctly reported that it had been
    given nothing to summarise.

    Each account is labelled with its sampler id, so the model can tell what
    was seen from what was said.
    """
    names = list(sources) if sources is not None else pick_sources(context)
    rows: list[tuple[int, str]] = []
    for chunk_id in context.chunk_ids():
        said = context.text_of(chunk_id)
        parts = [f"{n}: {said[n].strip()}" for n in names if said.get(n)]
        text = "  ".join(parts)
        if not text:
            continue
        start, end = context.span_of(chunk_id)
        rows.append((chunk_id, f"[{chunk_id}] {start:.0f}-{end:.0f}s  "
                               f"{text.strip()}"))
        if len(rows) >= limit:
            break
    return rows


def batched(items: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def resolve_span(context: Context, chunk_ids: Sequence[int]
                 ) -> tuple[float, float]:
    """The real span covering these chunks, from the grid.

    Never the model's own numbers: it is asked for chunk ids because those it
    can copy, and times it would invent.
    """
    if not chunk_ids:
        return 0.0, 0.0
    valid = [c for c in chunk_ids if 0 <= c < len(context.timeline)]
    if not valid:
        return 0.0, 0.0
    starts = [context.span_of(c)[0] for c in valid]
    ends = [context.span_of(c)[1] for c in valid]
    return min(starts), max(ends)


__all__ = ["PREFERENCE", "batched", "chunk_rows", "pick_sources", "resolve_span"]
