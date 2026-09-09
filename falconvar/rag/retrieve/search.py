"""A query, into ranked moments.

RRF twice: an index fuses vector and text rankings per unit, and this fuses the
units of one chunk into a moment.

    score = 1/(k + best) + 0.5/(k + second),  k = 10

A chunk scores as its best unit plus a discounted second, never a sum. Summing
over every unit a chunk contributed applies RRF to the wrong problem -- it
fuses several rankings of the *same* items, where the term count is constant,
while a chunk contributes one term per sampler that described it. Count then
overwhelms rank.

The bounded form keeps the agreement property on purpose: two accounts at ranks
2 and 3 still beat one at rank 1, which is why the per-sampler split exists.
A third and fourth account add nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

#: Small on purpose. At k=60 the reciprocal barely varies across a short
#: candidate list, which is what let count beat rank.
MOMENT_K = 10

#: What a second, independent account of the same window is worth.
SECOND_WEIGHT = 0.5


@dataclass
class Moment:
    """One chunk, and the units that spoke for it."""

    video_id: str
    chunk_id: int
    start_ts: float
    end_ts: float
    score: float
    hits: list[dict[str, Any]] = field(default_factory=list)

    @property
    def samplers(self) -> list[str]:
        return [h["sampler_id"] for h in self.hits]

    def as_dict(self) -> dict[str, Any]:
        return {"video_id": self.video_id, "chunk_id": self.chunk_id,
                "start_ts": round(self.start_ts, 3),
                "end_ts": round(self.end_ts, 3),
                "score": round(self.score, 6),
                "samplers": self.samplers,
                # Keyed by the pairing, with the two halves beside it, so a
                # caller can group by sampler or by question without parsing
                # an id whose separator is optional.
                "questions": {h["sampler_id"]: h.get("question", "")
                              for h in self.hits},
                "descriptions": {h["sampler_id"]: h["content"] for h in self.hits}}


def to_moments(hits: Sequence[dict[str, Any]], video_id: str,
               spans: Sequence[tuple[float, float]],
               limit: int = 5) -> list[Moment]:
    """Fuse per-unit hits into per-chunk moments."""
    ranked = sorted(hits, key=lambda h: -h["score"])
    by_chunk: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for rank, hit in enumerate(ranked, start=1):
        by_chunk.setdefault(hit["chunk_id"], []).append((rank, hit))

    moments: list[Moment] = []
    for chunk_id, entries in by_chunk.items():
        entries.sort(key=lambda p: p[0])
        best = entries[0][0]
        score = 1.0 / (MOMENT_K + best)
        if len(entries) > 1:
            score += SECOND_WEIGHT / (MOMENT_K + entries[1][0])
        start, end = (spans[chunk_id] if chunk_id < len(spans) else (0.0, 0.0))
        moments.append(Moment(video_id, chunk_id, start, end, score,
                              [hit for _, hit in entries]))
    moments.sort(key=lambda m: -m.score)
    return moments[:limit]


__all__ = ["MOMENT_K", "SECOND_WEIGHT", "Moment", "to_moments"]
