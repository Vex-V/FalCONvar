"""Qdrant, embedded or served, with both halves of the hybrid.

A dense vector and a sparse one in one collection, fused server-side by
`Fusion.RRF`.

The sparse half sends raw term frequencies with hashed term ids;
`Modifier.IDF` has Qdrant compute inverse document frequency from the
collection itself. IDF is a property of the corpus, and the corpus is what the
database holds.

One collection per embedder, named for it, so 768-wide vectors cannot be ranked
against 1536-wide ones.

Two things this sparse half does not do and Postgres does: no stemming, and no
length normalisation or TF saturation. `supabase` is the better lexical half;
this one is the cheaper one.

A stored sparse vector is built from tokens, so changing `indexes.tokenize`
makes it stale without changing `text_hash`. Delete the collection after
touching the tokeniser -- it is a derived cache and one re-upsert refills it.
"""

from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from ....shared import paths
from ..units import Unit
from . import tokenize

#: Named, because a collection carrying two kinds of vector needs to say which
#: is which. The names are also what `Prefetch(using=...)` selects.
DENSE = "dense"
SPARSE = "sparse"

#: RRF's constant, matching the other backends so a score means the same thing
#: whichever index produced it.
FUSE_K = 60


def _depth(limit: int) -> int:
    """How deep each half ranks before they are fused.

    Wider than the final limit, and matching the Postgres RPC's
    `greatest(p_limit * 4, 40)`. Prefetching only `limit` truncates each
    ranking before fusion, so a row the lexical half placed 25th cannot reach
    the result even when the dense half agrees -- measured as this backend
    firing on 12/20 rows where Postgres, ranking 80, fired on 20/20.
    """
    return max(limit * 4, 40)


def collection_name(embedder_key: str) -> str:
    return "falconvar_" + embedder_key.replace(":", "_").replace("/", "_")


def _point_id(key: str) -> str:
    """A stable uuid5 of the unit key. Qdrant wants an int or a uuid."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _term_id(token: str) -> int:
    """A stable id for a token. Unsigned 31-bit, which Qdrant accepts."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4)
                          .digest(), "big") & 0x7FFFFFFF


def sparse_of(text: str) -> Optional[Any]:
    """Term frequencies as a `SparseVector`. None when there are no terms.

    Raw counts, deliberately: `Modifier.IDF` on the collection applies the
    weighting, so anything done here would be applied twice.
    """
    from qdrant_client.models import SparseVector

    counts = Counter(_term_id(t) for t in tokenize(text))
    if not counts:
        return None
    indices = sorted(counts)
    return SparseVector(indices=indices,
                        values=[float(counts[i]) for i in indices])


class QdrantIndex:
    """A `VectorIndex` over one collection, dense and sparse together."""

    name = "qdrant"

    def __init__(self, video_id: str, embedder_key: str,
                 url: Optional[str] = None, path: Optional[Path] = None,
                 client: Any = None) -> None:
        self.video_id = video_id
        self.embedder_key = embedder_key
        self.collection = collection_name(embedder_key)

        if client is not None:
            self._client = client
            return
        from qdrant_client import QdrantClient
        if url:
            self._client = QdrantClient(url=url)
        else:
            # Shared across videos, so beside them rather than inside one
            # video's directory -- but still under OUT_ROOT, so `rm -rf
            # data/falconvar` removes everything this tree wrote. DATA_ROOT would
            # put it outside the namespace that exists to keep falconvar and
            # falconvar from colliding.
            store = Path(path or (paths.OUT_ROOT / "_qdrant"))
            store.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(store))
            # Only a client this object opened is one this object may close.
            self._owned = True

    #: Whether `close` should release the client. False for a client handed in
    #: or a served one, which the caller owns.
    _owned = False

    def close(self) -> None:
        """Release the storage lock. Idempotent.

        **Embedded Qdrant takes an exclusive lock on its folder**, and a client
        that is never closed never gives it back. A CLI run does not notice --
        the process exits and the lock goes with it, which is why this looked
        like a two-process problem. In a server it is a one-process problem:
        the first search leaks the lock and every later one fails with
        `Storage folder ... is already accessed by another instance`, on a
        route that worked a minute earlier.

        Served mode (`url=`) holds no lock, so there is nothing to release.
        """
        client, self._client = getattr(self, "_client", None), None
        if client is not None and self._owned:
            try:
                client.close()
            except Exception:                            # noqa: BLE001
                pass                                     # already closed

    def __enter__(self) -> "QdrantIndex":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- schema ----------------------------------------------------------
    def _has_sparse(self) -> bool:
        info = self._client.get_collection(self.collection)
        config = info.config.params
        return bool(getattr(config, "sparse_vectors", None))

    def _ensure(self, dims: int) -> None:
        from qdrant_client.models import (Distance, Modifier, SparseIndexParams,
                                          SparseVectorParams, VectorParams)

        if self._client.collection_exists(self.collection):
            if self._has_sparse():
                return
            # Written by an earlier, dense-only version of this file. The
            # collection is a derived cache and `embed` refills whatever an
            # index is missing, so recreating costs one re-upsert rather than
            # a migration -- and leaving it would mean the sparse half silently
            # never contributing.
            self._client.delete_collection(self.collection)

        self._client.create_collection(
            self.collection,
            vectors_config={DENSE: VectorParams(size=dims,
                                                distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE: SparseVectorParams(
                index=SparseIndexParams(),
                # The server computes IDF from the collection. See the module
                # docstring for why that is not the client's job.
                modifier=Modifier.IDF)},
        )

    # -- writing ---------------------------------------------------------
    def stored_hashes(self) -> dict[str, str]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not self._client.collection_exists(self.collection):
            return {}
        if not self._has_sparse():
            # About to be recreated by `_ensure`, so nothing in it counts as
            # stored. Saying it holds units would make `embed` skip the
            # re-upsert that refills it.
            return {}
        found: dict[str, str] = {}
        offset = None
        while True:
            points, offset = self._client.scroll(
                self.collection,
                scroll_filter=Filter(must=[FieldCondition(
                    key="video_id", match=MatchValue(value=self.video_id))]),
                limit=256, offset=offset, with_payload=True, with_vectors=False)
            for point in points:
                payload = point.payload or {}
                found[payload.get("key", "")] = payload.get("text_hash", "")
            if offset is None:
                break
        return found

    def upsert(self, units: Sequence[Unit]) -> int:
        from qdrant_client.models import PointStruct

        vectors = [u for u in units if u.vector]
        if not vectors:
            return 0
        self._ensure(len(vectors[0].vector))

        points = []
        for unit in vectors:
            payload_vectors: dict[str, Any] = {DENSE: list(unit.vector)}
            sparse = sparse_of(unit.content)
            if sparse is not None:
                payload_vectors[SPARSE] = sparse
            points.append(PointStruct(id=_point_id(unit.key),
                                      vector=payload_vectors, payload={
                "key": unit.key, "video_id": unit.video_id,
                "chunk_id": unit.chunk_id, "sampler_id": unit.sampler_id,
                # Both halves, so a filter can narrow to one pairing, to one
                # question across samplers, or to one sampler's whole output.
                "sampler": unit.sampler, "question": unit.question,
                "text_hash": unit.text_hash, "content": unit.content,
                "structured": unit.structured,
            }))
        self._client.upsert(self.collection, points=points)
        return len(points)

    def prune(self, live_keys: set[str]) -> int:
        """Drop points for chunks that no longer exist.

        An upserting store keeps whatever it was never told to remove, so a
        grid that shrank leaves points naming a chunk nobody can play.
        """
        from qdrant_client.models import PointIdsList

        stale = [k for k in self.stored_hashes() if k and k not in live_keys]
        if not stale:
            return 0
        self._client.delete(self.collection, points_selector=PointIdsList(
            points=[_point_id(k) for k in stale]))
        return len(stale)

    def save(self) -> str:
        return f"qdrant://{self.collection}"

    # -- reading ---------------------------------------------------------
    def _filter(self, sampler: Optional[str],
                question: Optional[str] = None,
                strategy: Optional[str] = None,
                chunk_ids: Optional[Sequence[int]] = None,
                structured: Optional[dict[str, Any]] = None,
                video_ids: Optional[Sequence[str]] = None) -> Any:
        """Every narrowing, as equalities on payload the unit already carries.

        `sampler` matches `sampler_id` -- the pairing, `clip:text` -- because
        that is the id every recorded measurement and every stored result uses.
        `question` matches across samplers, which is the query a person makes:
        "the text on screen", not "what the CLIP sampler said". `strategy`
        matches one sampler's whole output. None of the three is a prefix or
        suffix of the id: a bare `clip` means the question *is* the strategy
        name, so the two halves are carried as their own keys.

        `chunk_ids` is the drill-down -- search, read the ids back, ask for
        more about those -- and it is also how a *time* window is expressed,
        because the caller resolves seconds to ids through the grid rather than
        a span being stored a second time beside every vector.

        `structured` is exact values, and is only meaningful where a shape
        fixed the vocabulary with `one_of`. On free text one video produced
        `cashier`, `customer` and `cashier or customer near checkout`, and a
        filter for the first matched all three.
        """
        from qdrant_client.models import (FieldCondition, Filter, MatchAny,
                                          MatchValue)

        # Scope is a SET of videos: one, three, or every video is the same
        # question asked over a different set. `None` means every video, which
        # is why it adds no condition at all rather than defaulting to this
        # index's own id.
        must: list[Any] = []
        scope = list(video_ids) if video_ids is not None else (
            [self.video_id] if self.video_id else [])
        if scope:
            must.append(FieldCondition(key="video_id",
                                       match=MatchAny(any=scope)))
        if sampler is not None:
            must.append(FieldCondition(key="sampler_id",
                                       match=MatchValue(value=sampler)))
        if question is not None:
            must.append(FieldCondition(key="question",
                                       match=MatchValue(value=question)))
        if strategy is not None:
            must.append(FieldCondition(key="sampler",
                                       match=MatchValue(value=strategy)))
        if chunk_ids:
            must.append(FieldCondition(key="chunk_id",
                                       match=MatchAny(any=list(chunk_ids))))
        for field, value in (structured or {}).items():
            # Nested payload is addressed by path, so a shape's field is
            # filterable without a schema change -- the payload already holds
            # the whole structured answer.
            #
            # A list is one condition PER element, not `MatchAny`. Qdrant
            # matches an array field when *any* element equals the value, so N
            # separate must-conditions mean all N are present -- which is what
            # Postgres `structured @> {...}` means. `MatchAny` would mean *any*
            # of them, so the two backends would answer different questions
            # from the same request, and `speakers: [A, B]` is exactly where
            # that shows.
            for one in (value if isinstance(value, list) else [value]):
                must.append(FieldCondition(key=f"structured.{field}",
                                           match=MatchValue(value=one)))
        return Filter(must=must)

    def _ranks(self, query: Any, using: str, limit: int,
               where: Any) -> dict[str, int]:
        """point id -> rank for one half, so a hit can say which matched.

        The fused response carries a score but not its component ranks, and
        which half found something is exactly what a reader needs: a `t` marker
        distinguishes "the words matched too" from "dense answered alone".
        Two id-only queries against an embedded store cost almost nothing.
        """
        if query is None:
            return {}
        found = self._client.query_points(
            self.collection, query=query, using=using, limit=limit,
            query_filter=where, with_payload=False).points
        return {str(p.id): rank for rank, p in enumerate(found, start=1)}

    def search(self, vector: Sequence[float], query: str, limit: int = 20,
               sampler: Optional[str] = None,
               question: Optional[str] = None,
               strategy: Optional[str] = None,
               chunk_ids: Optional[Sequence[int]] = None,
               structured: Optional[dict[str, Any]] = None,
               video_ids: Optional[Sequence[str]] = None
               ) -> list[dict[str, Any]]:
        """Dense and sparse, fused by the server with RRF."""
        from qdrant_client.models import Fusion, FusionQuery, Prefetch

        if not self._client.collection_exists(self.collection):
            return []
        where = self._filter(sampler, question, strategy, chunk_ids,
                             structured, video_ids)
        dense = list(vector)
        sparse = sparse_of(query)

        depth = _depth(limit)
        prefetch = [Prefetch(query=dense, using=DENSE, limit=depth,
                             filter=where)]
        if sparse is not None:
            prefetch.append(Prefetch(query=sparse, using=SPARSE, limit=depth,
                                     filter=where))

        found = self._client.query_points(
            self.collection, prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit, with_payload=True).points

        # At the same depth the fusion used, or a hit fused from rank 30 would
        # come back with a blank marker and read as "this half said nothing".
        dense_rank = self._ranks(dense, DENSE, depth, where)
        text_rank = self._ranks(sparse, SPARSE, depth, where)

        hits = []
        for point in found:
            payload = point.payload or {}
            pid = str(point.id)
            hits.append({
                # Carried, not assumed from the caller: with a scope of several
                # videos a hit is only identifiable with it, and `to_moments`
                # groups on (video_id, chunk_id).
                "video_id": payload.get("video_id"),
                "chunk_id": payload.get("chunk_id"),
                "sampler_id": payload.get("sampler_id"),
                "sampler": payload.get("sampler", ""),
                "question": payload.get("question", ""),
                "content": payload.get("content", ""),
                "structured": payload.get("structured", {}),
                "score": float(point.score),
                "dense_rank": dense_rank.get(pid),
                "text_rank": text_rank.get(pid),
            })
        return hits


__all__ = ["DENSE", "FUSE_K", "SPARSE", "QdrantIndex", "collection_name", "sparse_of"]
