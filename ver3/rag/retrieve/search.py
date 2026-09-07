"""10 · a query, into ranked moments.

**RRF twice, never a weighted score.** Vector rank and text rank are fused per
unit inside the index; here the units of one chunk are fused into a moment.
Cosine distance and a lexical score have no common scale, and any weight
between them would be invented -- RRF reads only the orderings, so it needs no
calibration.

**A chunk scores as its best unit plus a discounted second, never a sum.**

    score = 1/(k + best) + 0.5/(k + second),  k = 10

Summing `1/(k+rank)` over every unit a chunk contributed is RRF applied to the
wrong problem. RRF fuses several rankings of the *same* items, where the term
count is constant; a chunk contributes one term per sampler that described it.
At k=60 over ~20 candidates `1/(k+rank)` spans only 1.31x, so count overwhelms
rank -- measured on `falconvar`, a chunk whose best description ranked 13th beat
one whose best ranked 1st, on three mediocre terms against one excellent one,
and the shipped ranking got the *video* right 57.7% of the time.

    aggregation              video ok   literal MRR   paraphrase MRR
    sum, k=60                  0.577       0.421          0.341
    max, k=60                  1.000       0.668          0.442
    max + 0.5*second, k=10     1.000       0.682          0.446

The bounded form keeps the agreement property on purpose -- two accounts at
ranks 2 and 3 still beat one at rank 1, which is why the per-sampler split
exists -- but a third and fourth account add nothing. It cannot regress however
many samplers run, where `sum` only survives while chunks have few units.

**This is invisible on a single video with a uniform sampler set**, where every
chunk contributes the same number of terms and the bias cancels. It appears the
moment an index holds more than one video, which is the normal case.
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
