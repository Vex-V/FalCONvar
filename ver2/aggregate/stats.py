"""Counts and densities over time.

Pure arithmetic on what the describers and the transcriber already wrote. It
exists because **embeddings cannot count.** "The busiest moment", "when was
nobody around", "how much of this is speech" are exact questions, and
similarity answers them approximately at best -- a search for "a crowded shop"
returns chunks that *read* crowded, which is not the same as the chunk with the
most people in it.

Everything here is derived from `structured`, never from the prose. The prose
is what got embedded; the fields are what a filter or a count can use, and this
is the clearest case of that split paying off.
"""

from __future__ import annotations

import collections
from typing import Any, Optional

from .base import Context


def _extreme(series: list[dict], key: str, pick) -> Optional[dict]:
    if not series:
        return None
    row = pick(series, key=lambda r: r[key])
    return {"chunk_id": row["chunk_id"], "start_ts": row["start_ts"],
            "end_ts": row["end_ts"], key: row[key]}


class StatsAggregator:
    """Counts, densities and the extremes of each, over the whole video."""

    id = "stats"
    tier = "free"
    depends_on = ()

    def aggregate(self, ctx: Context) -> Optional[dict[str, Any]]:
        chunks = ctx.chunks
        if not chunks:
            return None

        out: dict[str, Any] = {
            "duration_s": round(ctx.duration, 2),
            "chunks": len(chunks),
            "sources": ctx.sources,
        }

        # People, from whichever sampler owns them on this run: the `yolo`
        # specialist returns one bound object per person, and the scene
        # question returns plain phrases when no specialist ran.
        people = []
        for chunk in chunks:
            count = None
            for sampler, block in chunk["descriptions"].items():
                entries = (block.get("structured") or {}).get("people")
                if isinstance(entries, list):
                    count = len(entries)
                    break
            if count is not None:
                people.append({**{k: chunk[k] for k in
                                  ("chunk_id", "start_ts", "end_ts")},
                               "people": count})
        if people:
            counts = [p["people"] for p in people]
            out["people"] = {
                "series": people,
                "min": min(counts), "max": max(counts),
                "mean": round(sum(counts) / len(counts), 2),
                "total_observations": sum(counts),
                "busiest": _extreme(people, "people", max),
                "quietest": _extreme(people, "people", min),
            }

        # Objects and visible text, counted by how many chunks mention each.
        for field, label in (("objects", "objects"), ("visible_text", "visible_text")):
            tally: collections.Counter = collections.Counter()
            for chunk in chunks:
                for block in chunk["descriptions"].values():
                    for entry in (block.get("structured") or {}).get(field, []) or []:
                        name = (entry.get(field[:-1] if field.endswith("s") else "text")
                                or entry.get("text") or entry.get("object")
                                if isinstance(entry, dict) else entry)
                        if name:
                            tally[str(name).strip().lower()] += 1
            if tally:
                out[label] = {"distinct": len(tally),
                              "most_common": tally.most_common(15)}

        if ctx.has_speech:
            spoken = sum(t["end"] - t["start"]
                         for c in chunks for t in c["turns"])
            words = sum(c["word_count"] for c in chunks)
            with_speech = sum(1 for c in chunks if c["word_count"])
            out["speech"] = {
                "chunks_with_speech": with_speech,
                "chunks_silent": len(chunks) - with_speech,
                "words": words,
                "spoken_seconds": round(spoken, 2),
                # What fraction of the running time is somebody talking. A
                # number retrieval cannot produce at all.
                "speech_ratio": (round(spoken / ctx.duration, 3)
                                 if ctx.duration else None),
                "words_per_minute": (round(words / (ctx.duration / 60), 1)
                                     if ctx.duration else None),
            }
        return out

    def config(self) -> dict[str, Any]:
        return {"id": self.id, "tier": self.tier}
