"""From ranked descriptions to moments you can play.

The index ranks **descriptions**; what a person wants back is a **moment** -- a
window of media time with an in point, an out point, and the frames that prove
it. Those are different units, and this is where the gap is closed.

Ranks, not scores, for the same reason the SQL uses them: a cosine similarity
and an RRF score are not commensurable, while an *ordering* always means the
same thing.

**A chunk is not scored by summing its descriptions' ranks, and summing was
measured to be badly wrong.** RRF is defined for fusing several rankings of
the *same* items, where every item appears once per ranking, so the number of
terms is constant. A chunk contributes a variable number -- three if three
samplers described it, one if one did -- and at k=60 over ~20 candidates
1/(k+rank) spans only 1.31x, so count overwhelms rank. Measured on a mixed
index of two videos: all eleven Chernobyl descriptions ranked ahead of every
test1 description, and a test1 chunk whose best description ranked *13th* won
anyway, on three mediocre terms against one excellent one. Video-level
accuracy was 0.577. This docstring used to claim "a pile of weak matches
cannot outvote one strong one", which is false at k=60 -- two rank-40 hits
beat one rank-1 hit.

A chunk therefore scores as its **best** description plus a discounted
**second** best:

    score = 1/(k + best) + 0.5 * 1/(k + second)

Two terms at most, whatever the sampler count, so nothing wins by being
described more often. The second term keeps the property that splitting a
chunk into per-sampler descriptions exists for: a chunk whose clip description
ranked 2nd and whose yolo description ranked 5th can still beat a chunk with a
single 1st-place hit, because two independently-written accounts of the same
twenty seconds both matching is stronger evidence than one. A third and fourth
account add nothing, which is what stops count creeping back in.

Measured over 52 query pairs on two videos with different sampler counts:
video-level accuracy 0.577 -> 1.000, literal MRR 0.421 -> 0.682, paraphrase
0.341 -> 0.446. `eval/aggregation.py` reproduces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# One of the two names this module takes from `embed`. A Hit is what an
# index returns; a Moment is what this turns them into.
from ver2.embed.index import Hit

#: Rank decay. Small deliberately: at k=60 the gap between rank 1 and rank 20
#: is only 1.31x, which is what let a chunk win on count rather than on rank.
RRF_K = 10

#: What a second, independently-written account of the same window is worth
#: beside the best one. Applied to exactly one extra description, so a score is
#: bounded at two terms however many samplers ran.
AGREEMENT_WEIGHT = 0.5


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
               limit: Optional[int] = None,
               agreement: float = AGREEMENT_WEIGHT) -> list[Moment]:
    """Group ranked descriptions into ranked moments."""
    moments: dict[tuple[str, int], Moment] = {}
    terms: dict[tuple[str, int], list[float]] = {}
    for rank, hit in enumerate(hits, start=1):
        key = (hit.video_id, hit.chunk_id)
        moment = moments.get(key)
        if moment is None:
            moment = Moment(video_id=hit.video_id, chunk_id=hit.chunk_id, score=0.0,
                            start_ts=hit.start_ts, end_ts=hit.end_ts)
            moments[key] = moment
            terms[key] = []
        terms[key].append(1.0 / (rrf_k + rank))
        moment.hits.append(hit)
        if moment.start_ts is None:
            moment.start_ts, moment.end_ts = hit.start_ts, hit.end_ts

    # Best, plus a discounted second best -- never a sum over all of them,
    # which made a chunk's score a function of how many samplers described it
    # rather than of how well any single one matched.
    for key, moment in moments.items():
        ranked = sorted(terms[key], reverse=True)
        moment.score = ranked[0] + (agreement * ranked[1] if len(ranked) > 1 else 0.0)

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
