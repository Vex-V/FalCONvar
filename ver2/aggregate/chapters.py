"""Consecutive chunks grouped into named sections.

Gives the video a navigable structure: a timeline a person can click, and a
coarse index an agent can drill into before it knows what to search for. That
is what "topics" is actually useful for at single-video scale, where a
statistical topic model has far too little text to work with.

Chapters are **contiguous and non-overlapping** by construction -- they are
runs of the shared chunk grid, so a chapter is always a real span of media time
and always maps back to the same chunk ids the manifest and the transcript use.
Any chapter naming ids that do not exist is dropped rather than clamped to the
nearest real one, because a fabricated position looks exactly like a correct
one once it is written down.
"""

from __future__ import annotations

from typing import Any, Optional

from ver2.llm import DEFAULT_MODEL

from .base import Context
from .llm import SYSTEM, chunk_lines, complete, pick_sources, resolve_span

PREFERENCE = ("overview", "clip", "uniform", "transcript", "yolo", "objects", "text")

SCHEMA = {
    "name": "video_chapters",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["chapters"],
        "properties": {
            "chapters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["title", "first_chunk", "last_chunk", "summary"],
                    "properties": {
                        "title": {"type": "string",
                                  "description": "A short label, a few words."},
                        "first_chunk": {"type": "integer",
                                        "description": "Id of the first chunk in this chapter."},
                        "last_chunk": {"type": "integer",
                                       "description": "Id of the last chunk in this chapter."},
                        "summary": {"type": "string",
                                    "description": "One sentence on what happens here."},
                    },
                },
            }
        },
    },
}

PROMPT = """\
Below are consecutive segments of one video, each with its chunk id and \
timecode.

Group them into chapters: runs of consecutive chunks that belong together \
because the same thing is going on.

Rules:
- Chapters must be consecutive and must not overlap.
- Every chunk must fall inside exactly one chapter.
- Use the chunk ids exactly as given; do not invent ids.
- Prefer a handful of meaningful chapters over many tiny ones.

{lines}"""


class ChaptersAggregator:
    """The video divided into titled, contiguous sections."""

    id = "chapters"
    tier = "llm"
    depends_on = ()

    def __init__(self, model: Optional[str] = None, min_chunks: int = 3) -> None:
        self.model = model or DEFAULT_MODEL
        # Below this there is nothing to divide: a three-chunk video chaptered
        # into three chapters has learned nothing the grid did not already say.
        self.min_chunks = min_chunks

    def aggregate(self, ctx: Context) -> Optional[dict[str, Any]]:
        sources = pick_sources(ctx, PREFERENCE)
        lines = chunk_lines(ctx, sources, limit=260)
        if len(lines) < self.min_chunks:
            return None

        result = complete(PROMPT.format(lines="\n".join(lines)),
                          schema=SCHEMA, model=self.model, system=SYSTEM)

        chapters, dropped = [], 0
        for chapter in result.get("chapters", []):
            span = resolve_span(ctx, chapter.get("first_chunk"),
                                chapter.get("last_chunk"))
            if span is None:
                dropped += 1
                continue
            start, end, covered = span
            chapters.append({
                "title": chapter["title"],
                "summary": chapter["summary"],
                "start_ts": round(start, 3),
                "end_ts": round(end, 3),
                "first_chunk": chapter["first_chunk"],
                "last_chunk": chapter["last_chunk"],
                "chunk_ids": covered,
            })
        if not chapters:
            return None
        chapters.sort(key=lambda c: c["start_ts"])

        # Whether the chapters actually tile the video. Reported rather than
        # enforced: a gap means the model left chunks out, and knowing that is
        # more useful than silently stretching a neighbour over them.
        covered = {i for c in chapters for i in c["chunk_ids"]}
        missing = [c["chunk_id"] for c in ctx.chunks if c["chunk_id"] not in covered]
        return {"chapters": chapters, "based_on": sources,
                "dropped": dropped, "uncovered_chunks": missing,
                "covers_whole_video": not missing}

    def config(self) -> dict[str, Any]:
        return {"id": self.id, "tier": self.tier, "params": {"model": self.model}}
