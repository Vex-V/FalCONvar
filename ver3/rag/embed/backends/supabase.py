"""Postgres with pgvector, through the Supabase REST client.

**The default backend, and the reason is the lexical half.** `embeddings.fts`
is a generated tsvector over the content *and* the structured values, indexed
with GIN, so this is the only backend that can rank by text at all. Measured on
`falconvar`: dense-only 0.429 top-1 against 0.714 for dense + BM25 fused with
RRF, better on 4 of 7 queries and never worse.

**Ranking happens in the database, via an RPC.** Fetching every row to rank in
Python would move a video's whole index over the wire per query, and the two
halves have to be fused where both are cheap to produce. `db/supabase/rpc.sql`
holds `search_embeddings`; without it this falls back to a dense-only query and
says so rather than pretending.

Written through `shared/db.py` so the key names live in one place.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ....shared import db
from ..units import Unit

TABLE = "embeddings"


class SupabaseIndex:
    """A `VectorIndex` over the `embeddings` table."""

    name = "supabase"

    def __init__(self, video_id: str, embedder_key: str,
                 api: Any = None) -> None:
        self.video_id = video_id
        self.embedder_key = embedder_key
        self._api = api or db.client()
        self.degraded: Optional[str] = None

    # -- writing ---------------------------------------------------------
    def stored_hashes(self) -> dict[str, str]:
        response = (self._api.table(TABLE)
                    .select("chunk_id,sampler_id,text_hash")
                    .eq("video_id", self.video_id)
                    .eq("embedder", self.embedder_key).execute())
        return {f"{self.video_id}:{r['chunk_id']}:{r['sampler_id']}":
                r.get("text_hash", "")
                for r in (response.data or [])}

    def upsert(self, units: Sequence[Unit]) -> int:
        rows = [{
            "video_id": u.video_id, "chunk_id": u.chunk_id,
            "sampler_id": u.sampler_id, "embedder": self.embedder_key,
            "text_hash": u.text_hash, "content": u.content,
            "structured": u.structured, "embedding": list(u.vector or []),
        } for u in units if u.vector]
        return db.upsert(TABLE, rows, self._api)

    def prune(self, live_keys: set[str]) -> int:
        """Drop rows for chunks that no longer exist.

        An upserting table keeps whatever it was never told to remove, so a
        grid that shrank leaves rows naming a chunk nobody can play. The file
        backend rewrites its whole document and never has this problem.
        """
        stale = [k for k in self.stored_hashes() if k not in live_keys]
        for key in stale:
            _, chunk_id, sampler_id = key.split(":", 2)
            (self._api.table(TABLE).delete()
                .eq("video_id", self.video_id)
                .eq("chunk_id", int(chunk_id))
                .eq("sampler_id", sampler_id)
                .eq("embedder", self.embedder_key).execute())
        return len(stale)

    def save(self) -> str:
        return f"supabase://{TABLE}"

    # -- reading ---------------------------------------------------------
    def search(self, vector: Sequence[float], query: str, limit: int = 20,
               sampler: Optional[str] = None) -> list[dict[str, Any]]:
        try:
            response = self._api.rpc("search_embeddings", {
                "p_query_vector": list(vector),
                "p_query_text": query,
                "p_video_id": self.video_id,
                "p_embedder": self.embedder_key,
                "p_sampler": sampler,
                "p_limit": limit,
            }).execute()
        except Exception as exc:                         # noqa: BLE001
            # Said out loud rather than silently degraded: a dense-only result
            # and a fused one are indistinguishable on sight.
            self.degraded = (f"search_embeddings RPC unavailable ({exc}); "
                             "ranking has no lexical half")
            return self._dense_only(vector, limit, sampler)

        return [{
            "chunk_id": r["chunk_id"], "sampler_id": r["sampler_id"],
            "content": r.get("content", ""),
            "structured": r.get("structured", {}),
            "score": float(r.get("score", 0.0)),
            "dense_rank": r.get("vector_rank"),
            "text_rank": r.get("text_rank"),
        } for r in (response.data or [])]

    def _dense_only(self, vector: Sequence[float], limit: int,
                    sampler: Optional[str]) -> list[dict[str, Any]]:
        """Every row for this video, ranked in Python. The fallback.

        Correct but not scalable -- it moves the whole index over the wire. It
        exists so a database without the RPC still answers, and `degraded` says
        why the answer is worse.
        """
        import math

        query = (self._api.table(TABLE)
                 .select("chunk_id,sampler_id,content,structured,embedding")
                 .eq("video_id", self.video_id)
                 .eq("embedder", self.embedder_key))
        if sampler is not None:
            query = query.eq("sampler_id", sampler)
        rows = query.execute().data or []

        def cosine(a: Sequence[float], b: Sequence[float]) -> float:
            if not a or not b or len(a) != len(b):
                return -1.0
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        scored = sorted(rows, key=lambda r: -cosine(vector, r.get("embedding") or []))
        return [{
            "chunk_id": r["chunk_id"], "sampler_id": r["sampler_id"],
            "content": r.get("content", ""),
            "structured": r.get("structured", {}),
            "score": 1.0 / (60 + rank), "dense_rank": rank, "text_rank": None,
        } for rank, r in enumerate(scored[:limit], start=1)]


__all__ = ["SupabaseIndex", "TABLE"]
