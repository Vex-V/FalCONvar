"""What the text aggregators share: one rendering of the video, and how to pick.

`summary`, `chapters` and `events` all reduce over the *same* thing -- the
video as one line per chunk -- so that rendering lives here rather than being
invented three times with three subtly different truncations. What differs
between them is the question and the schema, which is what each module is.

**Which sources to read is a preference, not a fixed list.** A run may have any
combination of samplers plus a transcript, and the useful ordering is not the
same for every question: chapters and summaries want the broad scene account
first, events want people and objects, because a discrete happening is usually
somebody doing something. Each module states its own order and takes what the
video actually has.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ver2.llm import complete  # noqa: F401  (re-exported for the aggregators)
from .base import Context

SYSTEM = (
    "You are summarising machine-generated descriptions of one video: what a "
    "vision model saw in sampled frames, and what a speech model transcribed. "
    "Work only from the text you are given. Do not invent detail, do not "
    "speculate about intent or identity, and say plainly when the material is "
    "thin rather than padding it."
)


def pick_sources(ctx: Context, preference: Sequence[str]) -> list[str]:
    """The sources worth reading here, best first, filtered to what exists."""
    have = set(ctx.sources)
    chosen = [name for name in preference if name in have]
    # Anything the preference did not anticipate still gets read: a run may
    # carry `uniform:<custom prompt>` under an id no module knows about, and
    # ignoring it would silently drop the only account of the video.
    chosen += [name for name in ctx.sources if name not in chosen]
    return chosen


def chunk_rows(ctx: Context, sources: Sequence[str], limit: int = 300,
               include_speech: bool = True) -> list[tuple[int, str]]:
    """One `(chunk_id, line)` per chunk that has anything to say.

    The chunk id is included in the line because every aggregator that returns
    positions returns *these* ids -- a chapter naming a chunk range or an event
    naming a chunk is only resolvable back to a time span because the model was
    given the ids to use.

    It is also returned *beside* the line, because a caller that batches these
    needs to know which chunks a batch covered without parsing the string it
    just built. `summary` does exactly that, to give each of its intermediate
    summaries a real span.
    """
    rows = []
    for chunk in ctx.chunks:
        parts = []
        for source in sources:
            if source == "transcript":
                continue
            block = chunk["descriptions"].get(source) or {}
            if block.get("description"):
                parts.append(f"{source}: {block['description']}")
        if include_speech and chunk["transcript"]:
            parts.append(f"said: {chunk['transcript']}")
        body = "  ".join(parts).strip()
        if body:
            rows.append((chunk["chunk_id"],
                         f"[{chunk['chunk_id']}] "
                         f"{chunk['start_ts']:.1f}-{chunk['end_ts']:.1f}s: "
                         f"{body[:limit]}"))
    return rows


def chunk_lines(ctx: Context, sources: Sequence[str], limit: int = 300,
                include_speech: bool = True) -> list[str]:
    """The lines alone, for a caller that does not batch them."""
    return [line for _, line in chunk_rows(ctx, sources, limit, include_speech)]


def batched(items: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def resolve_span(ctx: Context, first: int, last: Optional[int] = None
                 ) -> Optional[tuple[float, float, list[int]]]:
    """Turn chunk ids back into a time span, or None if they are not real.

    A model asked for ids sometimes returns one that does not exist. Dropping
    the entry is better than clamping it to the nearest chunk, which would
    invent a position and make a wrong answer look like a right one.
    """
    by_id = {c["chunk_id"]: c for c in ctx.chunks}
    last = first if last is None else last
    if first not in by_id or last not in by_id or last < first:
        return None
    covered = [i for i in range(first, last + 1) if i in by_id]
    return by_id[first]["start_ts"], by_id[last]["end_ts"], covered
