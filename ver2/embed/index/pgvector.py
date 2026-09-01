"""pgvector, beside the descriptions it indexes.

The shared copy. Its advantage over the local one is not speed -- at this scale
neither is slow -- but that it sits in the same database as `descriptions`,
`chunks` and `videos`, so a search can be filtered and joined with SQL rather
than reconciled in Python afterwards.

It is also the half of the hybrid that can do full text. `search_descriptions`
runs the vector ranking and a `tsvector` ranking and fuses them with RRF,
inside one query. See schema.sql for why RRF rather than a weighted sum.

Vectors go in as a bracketed string because that is pgvector's text input form
and PostgREST speaks JSON, not the binary protocol.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ver2 import db

from ..units import Unit, embedder_key
from .base import Hit


def _literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.7g}" for v in vector) + "]"


class PgVectorIndex:
    """The `chunk_embeddings` table. A ``VectorIndex``."""

    name = "pgvector"

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None,
                 client: Any = None) -> None:
        self.client = client or db.client_from_env(url, key, write=True)

    def ensure(self, embedder: dict[str, Any]) -> None:
        # DDL is schema.sql's job: the table is shared by every embedder and
        # creating it from application code would mean each run asserting a
        # shape the others also assert. A missing table surfaces on first write.
        return None

    def existing(self, video_id: str, embedder: dict[str, Any]) -> dict[tuple[int, str], str]:
        rows = (self.client.table("chunk_embeddings")
                .select("chunk_id,sampler,text_hash")
                .eq("video_id", video_id)
                .eq("embedder", embedder_key(embedder))
                .execute().data)
        return {(r["chunk_id"], r["sampler"]): r["text_hash"] for r in rows}

    def upsert(self, units: Sequence[Unit], vectors: Sequence[Sequence[float]],
               embedder: dict[str, Any]) -> None:
        key = embedder_key(embedder)
        rows = []
        for unit, vector in zip(units, vectors):
            payload = unit.payload()
            rows.append({
                "video_id": unit.video_id,
                "chunk_id": unit.chunk_id,
                "sampler": unit.sampler,
                "embedder": key,
                "dims": len(vector),
                "embedding": _literal(vector),
                "content": payload["content"],
                "structured": payload["structured"],
                "text_hash": payload["text_hash"],
                "manifest_fingerprint": payload["manifest_fingerprint"],
                "start_ts": payload["start_ts"],
                "end_ts": payload["end_ts"],
                "frame_indexes": payload["frame_indexes"],
            })
        if rows:
            self.client.table("chunk_embeddings").upsert(
                rows, on_conflict="video_id,chunk_id,sampler,embedder").execute()

    def search(self, vector: Sequence[float], embedder: dict[str, Any],
               query_text: Optional[str] = None, video_id: Optional[str] = None,
               limit: int = 20, sampler: Optional[str] = None) -> list[Hit]:
        rows = self.client.rpc("search_embeddings", {
            "p_embedder": embedder_key(embedder),
            "p_query_vector": _literal(vector),
            "p_query_text": query_text,
            "p_video_id": video_id,
            "p_sampler": sampler,
            "p_limit": limit,
        }).execute().data or []
        return [Hit(
            video_id=r["video_id"], chunk_id=r["chunk_id"], sampler=r["sampler"],
            score=float(r["score"]), content=r["content"],
            start_ts=float(r["start_ts"]) if r.get("start_ts") is not None else None,
            end_ts=float(r["end_ts"]) if r.get("end_ts") is not None else None,
            frame_indexes=r.get("frame_indexes"),
            vector_rank=r.get("vector_rank"), text_rank=r.get("text_rank"),
        ) for r in rows]
