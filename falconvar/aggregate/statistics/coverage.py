"""Which chunks have an account, and from which modality.

A chunk nothing described is a hole in the index, and worth naming rather
than inferring from a count."""

from __future__ import annotations

from typing import Any

from ..base import Context

class CoverageAggregator:
    name = "coverage"
    tier = "free"
    about = "which chunks have an account, from which modality"
    depends_on: tuple[str, ...] = ()

    def run(self, context: Context) -> dict[str, Any]:
        rows = []
        both = picture_only = sound_only = neither = 0
        for chunk_id in context.chunk_ids():
            said = context.text_of(chunk_id)
            has_sound = bool(said.get("transcript"))
            has_picture = bool(set(said) - {"transcript"})
            if has_picture and has_sound:
                both += 1
            elif has_picture:
                picture_only += 1
            elif has_sound:
                sound_only += 1
            else:
                neither += 1
            start, end = context.span_of(chunk_id)
            rows.append({"chunk_id": chunk_id,
                         "start_ts": round(start, 3), "end_ts": round(end, 3),
                         "sources": sorted(said)})
        return {
            "both": both, "picture_only": picture_only,
            "sound_only": sound_only, "neither": neither,
            # A chunk nothing described is a hole in the index, and it is worth
            # naming rather than inferring from a count.
            "silent_chunks": [r["chunk_id"] for r in rows if not r["sources"]],
            "chunks": rows,
        }


__all__ = ["CoverageAggregator"]
