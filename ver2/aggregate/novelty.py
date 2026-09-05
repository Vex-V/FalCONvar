"""Which chunks are least like the rest of the video.

Reuses the vectors already in the index -- no model runs, no API call, nothing
new stored. For surveillance this is often the whole question ("show me the
unusual bit"), and it gives an agent a ranked place to start instead of
guessing search terms blind: you cannot search for the anomaly you have not
thought of, but you can rank by distance from the ordinary.

Distance is cosine from the video's own centroid, so "unusual" means unusual
*for this video*. A quiet shop where one chunk has a delivery, and a busy shop
where one chunk is empty, both surface their odd twenty seconds; neither is
compared against footage it has nothing to do with.

**Scored per sampler, then fused.** A chunk's `yolo` description can be an
outlier while its `clip` description is ordinary -- the people changed, the
room did not -- and averaging the two vectors would hide exactly that. Each
question is ranked in its own space and a chunk takes its highest novelty,
with the sampler that produced it recorded, so the answer says *what* was
unusual rather than only that something was.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from .base import Context

#: Below three points per sampler a centroid is not a centre of anything, and
#: every distance is an artifact of how few there are.
MIN_POINTS = 3


def _cosine_to_centroid(vectors: list[list[float]]) -> list[float]:
    """Distance of each vector from the mean direction, in [0, 2]."""
    dims = len(vectors[0])
    centroid = [sum(v[i] for v in vectors) / len(vectors) for i in range(dims)]
    norm = math.sqrt(sum(c * c for c in centroid)) or 1.0
    centroid = [c / norm for c in centroid]
    out = []
    for vector in vectors:
        length = math.sqrt(sum(x * x for x in vector)) or 1.0
        out.append(1.0 - sum(a * b for a, b in zip(vector, centroid)) / length)
    return out


class NoveltyAggregator:
    """Ranks a video's chunks by how unlike the rest of it they are."""

    id = "novelty"
    tier = "free"
    depends_on = ()

    def __init__(self, min_points: int = MIN_POINTS) -> None:
        self.min_points = min_points

    def aggregate(self, ctx: Context) -> Optional[dict[str, Any]]:
        # The vectors are the input, so an unembedded video has no answer here
        # rather than a poor one.
        if ctx.index is None or ctx.embedder is None:
            return None
        rows = _vectors_for(ctx)
        if not rows:
            return None

        by_sampler: dict[str, list[dict]] = {}
        for row in rows:
            by_sampler.setdefault(row["sampler"], []).append(row)

        best: dict[int, dict] = {}
        bases = {}
        for sampler, group in sorted(by_sampler.items()):
            if len(group) < self.min_points:
                continue
            distances = _cosine_to_centroid([r["vector"] for r in group])
            mean = sum(distances) / len(distances)
            spread = math.sqrt(sum((d - mean) ** 2 for d in distances) / len(distances))
            bases[sampler] = {"points": len(group), "mean_distance": round(mean, 4),
                              "stdev": round(spread, 4)}
            for row, distance in zip(group, distances):
                entry = {"chunk_id": row["chunk_id"], "sampler": sampler,
                         "novelty": round(distance, 4),
                         "outlier": distance > mean + 2 * spread,
                         "start_ts": row["start_ts"], "end_ts": row["end_ts"],
                         "text": (row["content"] or "")[:220]}
                if (row["chunk_id"] not in best
                        or entry["novelty"] > best[row["chunk_id"]]["novelty"]):
                    best[row["chunk_id"]] = entry

        if not best:
            return None
        ranked = sorted(best.values(), key=lambda r: -r["novelty"])
        return {"bases": bases, "ranked": ranked,
                "outliers": [r for r in ranked if r["outlier"]],
                "most_unusual": ranked[0]}

    def config(self) -> dict[str, Any]:
        return {"id": self.id, "tier": self.tier, "min_points": self.min_points}


def _vectors_for(ctx: Context) -> list[dict[str, Any]]:
    """Every stored vector for this video, whichever index holds them.

    Read through the index rather than recomputed: these are the same vectors
    retrieval ranks against, so novelty and search agree about what a chunk
    means. Recomputing would embed the text a second time and could disagree
    with the index if the text had since changed.
    """
    index = ctx.index
    if hasattr(index, "client") and hasattr(index.client, "table"):
        rows = (index.client.table("chunk_embeddings")
                .select("chunk_id,sampler,content,start_ts,end_ts,embedding")
                .eq("video_id", ctx.video_id)
                .eq("embedder", _key(ctx)).execute().data)
        return [{"chunk_id": r["chunk_id"], "sampler": r["sampler"],
                 "content": r["content"],
                 "start_ts": float(r["start_ts"]) if r["start_ts"] is not None else 0.0,
                 "end_ts": float(r["end_ts"]) if r["end_ts"] is not None else 0.0,
                 "vector": _as_vector(r["embedding"])} for r in rows]

    # Qdrant: scroll the collection with vectors attached.
    from ver2.embed.units import collection_name, embedder_key

    config = ctx.embedder.config()
    name = collection_name(embedder_key(config))
    if not index.client.collection_exists(name):
        return []
    from qdrant_client import models

    out, offset = [], None
    while True:
        points, offset = index.client.scroll(
            collection_name=name,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="video_id", match=models.MatchValue(value=ctx.video_id))]),
            with_payload=True, with_vectors=True, limit=256, offset=offset)
        for point in points:
            payload = point.payload or {}
            out.append({"chunk_id": payload["chunk_id"],
                        "sampler": payload["sampler"],
                        "content": payload.get("content", ""),
                        "start_ts": payload.get("start_ts") or 0.0,
                        "end_ts": payload.get("end_ts") or 0.0,
                        "vector": list(point.vector)})
        if offset is None:
            return out


def _key(ctx: Context) -> str:
    from ver2.embed.units import embedder_key

    return embedder_key(ctx.embedder.config())


def _as_vector(value: Any) -> list[float]:
    """pgvector arrives as its text form through PostgREST: `[0.1,0.2,…]`."""
    if isinstance(value, str):
        return [float(x) for x in value.strip("[]").split(",") if x]
    return [float(x) for x in value]
