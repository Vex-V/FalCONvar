"""From ranked descriptions to moments you can play.

The index ranks **descriptions**; what a person wants back is a **moment** -- a
window of media time with an in point, an out point, and the frames that prove
it. Those are different units, and this is where the gap is closed.

Aggregating by Reciprocal Rank Fusion again, for the same reason it is used
inside the SQL: the scores coming back are on scales that do not survive being
added together (a cosine similarity and an RRF score are not commensurable),
while the *ordering* always means the same thing. A chunk's score is the sum
of 1/(k + rank) over each of its descriptions that ranked.

The consequence is deliberate. A chunk whose clip description ranked 2nd and
whose yolo description ranked 5th beats a chunk with a single 1st-place hit,
because two independently-written accounts of the same twenty seconds both
matching the question is stronger evidence than one. But it takes real
agreement: 1/(k+1) dwarfs 1/(k+40), so a pile of weak matches cannot outvote
one strong one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .index.base import Hit

RRF_K = 60


@dataclass
class Moment:
    """One chunk of one video, and why it matched."""

    video_id: str
    chunk_id: int
    score: float
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    hits: list[Hit] = field(default_factory=list)

    @property
    def samplers(self) -> list[str]:
        return [h.sampler for h in self.hits]

    @property
    def frame_indexes(self) -> list[int]:
        """Every frame any matching description covered, deduplicated.

        The evidence for the moment: these come straight out of the frame
        store, no seeking and no re-decoding.
        """
        seen: list[int] = []
        for hit in self.hits:
            for index in hit.frame_indexes or []:
                if index not in seen:
                    seen.append(index)
        return sorted(seen)

    @property
    def span(self) -> str:
        if self.start_ts is None or self.end_ts is None:
            return "?"
        return f"{self.start_ts:.1f}-{self.end_ts:.1f}s"

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "samplers": self.samplers,
            "frame_indexes": self.frame_indexes,
            "descriptions": {h.sampler: h.content for h in self.hits},
        }


def to_moments(hits: list[Hit], rrf_k: int = RRF_K,
               limit: Optional[int] = None) -> list[Moment]:
    """Group ranked descriptions into ranked moments."""
    moments: dict[tuple[str, int], Moment] = {}
    for rank, hit in enumerate(hits, start=1):
        key = (hit.video_id, hit.chunk_id)
        moment = moments.get(key)
        if moment is None:
            moment = Moment(video_id=hit.video_id, chunk_id=hit.chunk_id, score=0.0,
                            start_ts=hit.start_ts, end_ts=hit.end_ts)
            moments[key] = moment
        moment.score += 1.0 / (rrf_k + rank)
        moment.hits.append(hit)
        if moment.start_ts is None:
            moment.start_ts, moment.end_ts = hit.start_ts, hit.end_ts

    ordered = sorted(moments.values(), key=lambda m: m.score, reverse=True)
    for moment in ordered:
        moment.hits.sort(key=lambda h: h.score, reverse=True)
    return ordered[:limit] if limit else ordered


def search(query: str, embedder, index, video_id: Optional[str] = None,
           limit: int = 20, moments: int = 5,
           sampler: Optional[str] = None) -> list[Moment]:
    """Embed the question, rank descriptions, fold them into moments.

    ``limit`` is how many descriptions to rank, ``moments`` how many windows to
    return. The first should be comfortably larger than the second: a moment
    can only benefit from agreement between its descriptions if both of them
    made it into the ranked list.

    ``sampler`` restricts the search to one question's answers -- ask only what
    the person detector saw, or only what was written on screen. It is a filter
    over one shared space, since every vector in the index came from the same
    embedder.

    **Filtering costs the agreement signal, and that is worth knowing.** With
    one sampler a chunk can contribute at most one description, so the RRF
    aggregation below has nothing to fuse and degenerates to plain rank order.
    Unfiltered search is the only mode where two independent accounts of the
    same twenty seconds can vote together.
    """
    vector = embedder.embed_query(query)
    hits = index.search(vector, embedder.config(), query_text=query,
                        video_id=video_id, limit=limit, sampler=sampler)
    return to_moments(hits, limit=moments)
