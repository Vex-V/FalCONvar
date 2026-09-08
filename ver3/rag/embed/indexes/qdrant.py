"""Qdrant, embedded or served, with both halves of the hybrid.

**A dense vector and a sparse one, fused server-side by `Fusion.RRF`.** Qdrant
has had sparse vectors since 1.7 and native fusion since 1.10; an earlier note
in this package called it dense-only, which was inherited from `falconvar` and
was a claim about the implementation rather than the database.

**The sparse half is IDF-weighted by the server, not by us.** The client sends
term frequencies with stable ids; `Modifier.IDF` on the sparse vector makes
Qdrant compute inverse document frequency from the collection itself. That is
what turns a bag of counts into a BM25-shaped score, and it is better than
computing IDF here for a reason worth stating: IDF is a property of the corpus,
and the corpus is what the database holds. A client that computed it would be
working from whatever subset it happened to have loaded.

Terms are hashed to ids rather than kept in a vocabulary. A vocabulary would
have to be built before the first insert, shared with every writer, and
migrated whenever it grew -- and with the server doing IDF there is nothing a
vocabulary would buy. Collisions are possible and harmless at this scale: two
unrelated words sharing an id costs one spurious weak match, not a wrong
ranking.

**One collection per embedder, named for it.** `name:model:dims` becomes the
collection, so 768-wide vectors cannot be ranked against 1536-wide ones, and a
query embedded by the wrong model searches a collection that does not exist
rather than the wrong vectors.

**A stored sparse vector is built from tokens, so changing the tokeniser makes
it stale.** `text_hash` is over the content, not over the terms, so nothing
notices: `embed` reports every unit current and skips the re-upsert. Delete the
collection after touching `indexes.tokenize` -- it is a derived cache and one
re-upsert refills it.

Embedded by default -- a local path, no service to run -- because the point of
a second backend is developing against the local stack.
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


class QdrantUnavailable(RuntimeError):
    """No client, or a server that will not answer."""


def collection_name(embedder_key: str) -> str:
    return "ver3_" + embedder_key.replace(":", "_").replace("/", "_")


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
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:                       # pragma: no cover
            raise QdrantUnavailable(
                "qdrant-client is not installed: pip install qdrant-client"
            ) from exc
        if url:
            self._client = QdrantClient(url=url)
        else:
            # Shared across videos, so beside them rather than inside one
            # video's directory -- but still under OUT_ROOT, so `rm -rf
            # data/ver3` removes everything this tree wrote. DATA_ROOT would
            # put it outside the namespace that exists to keep ver3 and
            # falconvar from colliding.
            store = Path(path or (paths.OUT_ROOT / "_qdrant"))
            store.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(store))

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
    def _filter(self, sampler: Optional[str]) -> Any:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = [FieldCondition(key="video_id",
                               match=MatchValue(value=self.video_id))]
        if sampler is not None:
            must.append(FieldCondition(key="sampler_id",
                                       match=MatchValue(value=sampler)))
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
               sampler: Optional[str] = None) -> list[dict[str, Any]]:
        """Dense and sparse, fused by the server with RRF."""
        from qdrant_client.models import Fusion, FusionQuery, Prefetch

        if not self._client.collection_exists(self.collection):
            return []
        where = self._filter(sampler)
        dense = list(vector)
        sparse = sparse_of(query)

        prefetch = [Prefetch(query=dense, using=DENSE, limit=limit,
                             filter=where)]
        if sparse is not None:
            prefetch.append(Prefetch(query=sparse, using=SPARSE, limit=limit,
                                     filter=where))

        found = self._client.query_points(
            self.collection, prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit, with_payload=True).points

        dense_rank = self._ranks(dense, DENSE, limit, where)
        text_rank = self._ranks(sparse, SPARSE, limit, where)

        hits = []
        for point in found:
            payload = point.payload or {}
            pid = str(point.id)
            hits.append({
                "chunk_id": payload.get("chunk_id"),
                "sampler_id": payload.get("sampler_id"),
                "content": payload.get("content", ""),
                "structured": payload.get("structured", {}),
                "score": float(point.score),
                "dense_rank": dense_rank.get(pid),
                "text_rank": text_rank.get(pid),
            })
        return hits


__all__ = ["DENSE", "FUSE_K", "SPARSE", "QdrantIndex", "QdrantUnavailable",
           "collection_name", "sparse_of"]
