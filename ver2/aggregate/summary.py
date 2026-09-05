"""One summary of the whole video, reduced hierarchically.

**Not one call over everything.** A long video's chunk lines run past any
sensible context, and a model handed all of them writes about the beginning and
the end. Summarising batches, then summarising the summaries, keeps every call
looking at material it can actually hold -- and the middle of a two-hour
recording survives into the answer.

Short videos take the single-call path, because a tree over four lines adds a
round trip and a layer of paraphrase for nothing.

**Every layer is recorded.** The intermediate summaries used to be thrown away
once the final call had read them, which discarded a coarse account of the
video that had already been paid for -- a leaf summary covers a real span and
sits exactly between "one chunk" and "the whole video" in granularity. They are
kept in `layers`, each part carrying the chunk ids and time span it covers.
They are still not embedded: they paraphrase material the chunk vectors already
hold, so indexing them would return the same moment several times over.

This is the one aggregate that is **embedded**. A summary is text about a
video, so "which video is about nuclear reactors" is a question it can answer
and no per-chunk vector can: a chunk that mentions reactors is not the same as
a video that is about them. Everything else here is statistical, and a vector
of a count is not a useful thing.
"""

from __future__ import annotations

from typing import Any, Optional

from ver2.llm import DEFAULT_MODEL

from .base import Context
from .llm import SYSTEM, batched, chunk_rows, complete, pick_sources

PREFERENCE = ("overview", "clip", "uniform", "transcript", "yolo", "objects", "text")

#: How many chunk lines one leaf call sees. Sized so a leaf stays well inside
#: the context with room for the instruction, rather than to a token budget
#: that would need recomputing per model.
BATCH = 25

SCHEMA = {
    "name": "video_summary",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "key_points", "topics"],
        "properties": {
            "summary": {
                "type": "string",
                "description": ("What this video is and what happens in it, "
                                "in 4 to 8 sentences of prose. This is the text "
                                "a search index is built from, so it must "
                                "contain what someone would search for."),
            },
            "key_points": {
                "type": "array", "items": {"type": "string"},
                "description": "The things someone would most need to know, one per entry.",
            },
            "topics": {
                "type": "array", "items": {"type": "string"},
                "description": "Short lowercase labels for what the video covers.",
            },
        },
    },
}

LEAF = """\
These are consecutive segments of one video, each with its chunk id and \
timecode. Summarise what happens across them in 2 to 3 sentences. Stay factual \
and keep any detail someone searching this footage later would need.

{lines}"""

MERGE = """\
These are summaries of consecutive parts of one video, in order. Combine them \
into a single summary of the whole stretch, in 2 to 3 sentences. Keep what \
matters, drop repetition, and add nothing that is not below.

{parts}"""

FINAL = """\
These summaries cover consecutive stretches of a single video, in order. \
Produce an overall summary, the key points someone would need, and short topic \
labels.

{parts}"""


class SummaryAggregator:
    """A whole-video summary, key points and topics."""

    id = "summary"
    tier = "llm"
    depends_on = ()

    def __init__(self, model: Optional[str] = None, batch: int = BATCH) -> None:
        self.model = model or DEFAULT_MODEL
        self.batch = batch

    def aggregate(self, ctx: Context) -> Optional[dict[str, Any]]:
        sources = pick_sources(ctx, PREFERENCE)
        rows = chunk_rows(ctx, sources, limit=320)
        if not rows:
            return None
        by_id = {c["chunk_id"]: c for c in ctx.chunks}

        # A part is one piece of text plus the chunk ids it covers. The ids
        # travel with the text through every fold, so a merge summary's span is
        # the union of what it merged, and nothing has to be reconstructed
        # afterwards by parsing a string this module formatted itself.
        parts = [{"text": line, "ids": [chunk_id]} for chunk_id, line in rows]
        layers: list[dict[str, Any]] = []
        levels = 0

        if len(parts) > self.batch:
            parts = self._fold(parts, LEAF, "lines", "\n")
            layers.append(self._layer(1, "leaf", parts, by_id))
            levels = 1
            # Keep folding until one call can hold what is left. A video long
            # enough to need three levels is rare; the loop costs nothing when
            # it is not.
            while len(parts) > self.batch:
                parts = self._fold(parts, MERGE, "parts", "\n\n")
                levels += 1
                layers.append(self._layer(levels, "merge", parts, by_id))

        joined = "\n\n".join(part["text"] for part in parts)
        result = complete(FINAL.format(parts=joined),
                          schema=SCHEMA, model=self.model, system=SYSTEM)
        summary = (result.get("summary") or "").strip()
        if not summary:
            return None
        return {
            "summary": summary,
            "key_points": result.get("key_points") or [],
            "topics": [t.lower() for t in (result.get("topics") or [])],
            "based_on": sources,
            "chunks": len(rows),
            # How much paraphrase sits between the descriptions and this text.
            # Zero means one call saw everything; two means it saw summaries of
            # summaries, which is worth knowing when judging the wording.
            "reduction_levels": levels,
            # Every intermediate summary, in the order it was produced. These
            # were transient once, which threw away a coarse account of the
            # video that had already been paid for: a leaf summary covers a
            # real span, and that is exactly the granularity between "one
            # chunk" and "the whole video". Kept, and still not embedded --
            # they paraphrase material the chunk vectors already hold, so
            # indexing them would return the same moment several times over.
            "layers": layers,
        }

    def _fold(self, parts: list[dict[str, Any]], template: str, field: str,
              joiner: str) -> list[dict[str, Any]]:
        """One round of summarising: N parts in, ceil(N / batch) parts out."""
        folded = []
        for group in batched(parts, self.batch):
            body = joiner.join(part["text"] for part in group)
            folded.append({
                "text": complete(template.format(**{field: body}),
                                 model=self.model, system=SYSTEM),
                "ids": [i for part in group for i in part["ids"]],
            })
        return folded

    @staticmethod
    def _layer(level: int, kind: str, parts: list[dict[str, Any]],
               by_id: dict[int, Any]) -> dict[str, Any]:
        """One recorded layer: each part with the span it actually covers.

        The span is resolved from the chunk ids rather than carried along as
        numbers, so it agrees with the timeline by construction -- the same
        reason `chapters` resolves its spans instead of trusting the model's.
        """
        entries = []
        for part in parts:
            ids = sorted(part["ids"])
            first, last = by_id.get(ids[0]), by_id.get(ids[-1])
            entries.append({
                "text": part["text"],
                "chunk_ids": [ids[0], ids[-1]],
                "start_ts": round(first["start_ts"], 3) if first else None,
                "end_ts": round(last["end_ts"], 3) if last else None,
            })
        return {"level": level, "kind": kind, "parts": entries}

    def config(self) -> dict[str, Any]:
        return {"id": self.id, "tier": self.tier,
                "params": {"model": self.model, "batch": self.batch}}
