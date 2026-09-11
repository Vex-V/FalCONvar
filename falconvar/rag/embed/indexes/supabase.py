"""Postgres with pgvector, through the Supabase REST client.

Ranking happens in the database via the `search_embeddings` RPC, which fuses a
vector ranking and a `ts_rank_cd` text ranking with RRF. Fetching every row to
rank in Python would move a video's whole index over the wire per query.

Without the RPC this falls back to a dense-only query and sets `degraded` to
say why, because a dense-only result and a fused one are indistinguishable on
sight.

`db/supabase/install.sql` holds the schema and the RPC.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ....shared import db
from ..units import Unit

TABLE = "embeddings"


def as_vector(value: Any) -> list[float]:
    """A pgvector column as floats, whatever PostgREST handed back.

    **It hands back a string.** `vector(1536)` arrives as the text
    `"[-0.0342,0.0450,...]"`, not a list -- so a cosine written against a list
    silently compared nothing, returned its "these are not comparable"
    sentinel for every row, and left `sorted` to preserve the order the rows
    happened to arrive in. The fallback then reported a ranking it had not
    computed, with a `degraded` note saying only that the lexical half was
    missing. Measured: every similarity -1.0 across a 4-row table.
    """
    if isinstance(value, str):
        try:
            return [float(x) for x in value.strip("[]").split(",") if x]
        except ValueError:
            return []
    return [float(x) for x in (value or [])]

#: The whole-video half. Its own table because `embeddings` answers *which
#: twenty seconds* and this answers *which video* -- and a video is not a moment
#: you can play, so a shared table would return one beside real moments.
VIDEO_TABLE = "video_embeddings"


def write_video_unit(unit: Any, embedder_key: str, api: Any = None) -> int:
    """Upsert one whole-video vector. Keyed (video_id, kind, embedder).

    Separate from `SupabaseIndex` because it is not an index over moments: it
    has no chunk, no sampler and no ranking of its own yet. `install.sql` has
    held the table since before anything wrote it; this is what fills it.
    """
    from ....shared import db
    if not unit or not unit.vector:
        return 0
    return db.upsert(VIDEO_TABLE, [{
        "video_id": unit.video_id, "kind": "summary",
        "embedder": embedder_key, "text_hash": unit.text_hash,
        "content": unit.content, "embedding": list(unit.vector),
    }], api or db.client())


def search_videos(vector: Sequence[float], embedder_key: str,
                  limit: int = 5, api: Any = None) -> list[dict[str, Any]]:
    """Which video is this about. Ranked in Python, deliberately.

    One row per video, so the whole table is a handful of vectors even on a
    large deployment -- the objection to `_dense_only` (it moves a video's whole
    index over the wire) does not apply when the table IS one row per video.
    """
    import math
    from ....shared import db

    rows = (api or db.client(write=False)).table(VIDEO_TABLE).select(
        "video_id,kind,content,embedding").eq("embedder", embedder_key
                                              ).execute().data or []

    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    scored = sorted(rows, key=lambda r: -cosine(vector, as_vector(r.get("embedding"))))
    return [{"video_id": r["video_id"], "kind": r.get("kind", "summary"),
             "content": r.get("content", ""),
             "similarity": round(cosine(vector, as_vector(r.get("embedding"))), 6)}
            for r in scored[:limit]]


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
            # Both halves as columns, so narrowing to one question across
            # samplers is an equality rather than a suffix match on the id.
            "sampler": u.sampler, "question": u.question,
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
               sampler: Optional[str] = None,
               question: Optional[str] = None,
               strategy: Optional[str] = None,
               chunk_ids: Optional[Sequence[int]] = None,
               structured: Optional[dict[str, Any]] = None,
               video_ids: Optional[Sequence[str]] = None
               ) -> list[dict[str, Any]]:
        # `None` means every video. The RPC takes an array for the same reason
        # this does: one video and three are the same question over a
        # different set.
        scope = list(video_ids) if video_ids is not None else (
            [self.video_id] if self.video_id else None)
        try:
            response = self._api.rpc("search_embeddings", {
                "p_query_vector": list(vector),
                "p_query_text": query,
                "p_video_ids": scope,
                "p_embedder": self.embedder_key,
                "p_sampler": sampler,
                "p_question": question,
                "p_strategy": strategy,
                "p_chunk_ids": list(chunk_ids) if chunk_ids else None,
                "p_structured": structured or None,
                "p_limit": limit,
            }).execute()
        except Exception as exc:                         # noqa: BLE001
            # Said out loud rather than silently degraded: a dense-only result
            # and a fused one are indistinguishable on sight.
            self.degraded = (f"search_embeddings RPC unavailable ({exc}); "
                             "ranking has no lexical half")
            return self._dense_only(vector, limit, sampler, question,
                                    strategy, chunk_ids, structured, scope)

        return [{
            "video_id": r.get("video_id", self.video_id),
            "chunk_id": r["chunk_id"], "sampler_id": r["sampler_id"],
            "sampler": r.get("sampler", ""), "question": r.get("question", ""),
            "content": r.get("content", ""),
            "structured": r.get("structured", {}),
            "score": float(r.get("score", 0.0)),
            "dense_rank": r.get("vector_rank"),
            "text_rank": r.get("text_rank"),
            # The RPC already joins `chunks` for these. Dropping them meant the
            # caller re-read a local `timeline.json` to learn what the database
            # had just told it -- which made search fail outright on a
            # deployment that has the rows and no output directory.
            "start_ts": r.get("start_ts"), "end_ts": r.get("end_ts"),
        } for r in (response.data or [])]

    def _dense_only(self, vector: Sequence[float], limit: int,
                    sampler: Optional[str],
                    question: Optional[str] = None,
                    strategy: Optional[str] = None,
                    chunk_ids: Optional[Sequence[int]] = None,
                    structured: Optional[dict[str, Any]] = None,
                    video_ids: Optional[Sequence[str]] = None
                    ) -> list[dict[str, Any]]:
        """Every row for this video, ranked in Python. The fallback.

        Correct but not scalable -- it moves the whole index over the wire. It
        exists so a database without the RPC still answers, and `degraded` says
        why the answer is worse.
        """
        import math

        query = (self._api.table(TABLE)
                 .select("video_id,chunk_id,sampler_id,sampler,question,"
                         "content,structured,embedding")
                 .eq("embedder", self.embedder_key))
        if video_ids:
            query = query.in_("video_id", list(video_ids))
        if sampler is not None:
            query = query.eq("sampler_id", sampler)
        if question is not None:
            query = query.eq("question", question)
        # Every filter the RPC applies, applied here too. A fallback that
        # quietly ignored one would answer a different question from the path
        # it stands in for, and `degraded` only warns about the ranking.
        if strategy is not None:
            query = query.eq("sampler", strategy)
        if chunk_ids:
            query = query.in_("chunk_id", list(chunk_ids))
        if structured:
            query = query.contains("structured", structured)
        rows = query.execute().data or []

        def cosine(a: Sequence[float], b: Sequence[float]) -> float:
            if not a or not b or len(a) != len(b):
                return -1.0
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        scored = sorted(rows, key=lambda r: -cosine(vector, as_vector(r.get("embedding"))))
        return [{
            "video_id": r.get("video_id", self.video_id),
            "chunk_id": r["chunk_id"], "sampler_id": r["sampler_id"],
            "content": r.get("content", ""),
            "structured": r.get("structured", {}),
            "sampler": r.get("sampler", ""), "question": r.get("question", ""),
            "score": 1.0 / (60 + rank), "dense_rank": rank, "text_rank": None,
        } for rank, r in enumerate(scored[:limit], start=1)]


__all__ = ["SupabaseIndex", "TABLE"]
