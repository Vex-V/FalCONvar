"""Qdrant, embedded or served.

**Dense only, and it says so.** Qdrant has no lexical half here, so a search
against it drops the component measured at 0.429 against 0.714 top-1 on the
same corpus. `backends.has_lexical("qdrant")` is False and the retrieve driver
prints a note, because the two results are indistinguishable on sight: every
hit simply lacks a `t` marker.

**One collection per embedder, named for it.** `name:model:dims` becomes the
collection, so 768-wide vectors cannot be ranked against 1536-wide ones, and a
query embedded by the wrong model searches a collection that does not exist
rather than the wrong vectors.

Embedded by default -- a local path, no service to run -- because the point of
having a second backend is to develop against the local stack, and a container
requirement would defeat that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from ....shared import paths
from ..units import Unit


class QdrantUnavailable(RuntimeError):
    """No client, or a server that will not answer."""


def collection_name(embedder_key: str) -> str:
    return "ver3_" + embedder_key.replace(":", "_").replace("/", "_")


def _point_id(unit: Unit) -> str:
    """A stable uuid5 of the unit key. Qdrant wants an int or a uuid."""
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, unit.key))


class QdrantIndex:
    """A `VectorIndex` over one collection."""

    name = "qdrant"

    def __init__(self, video_id: str, embedder_key: str,
                 url: Optional[str] = None, path: Optional[Path] = None,
                 client: Any = None) -> None:
        self.video_id = video_id
        self.embedder_key = embedder_key
        self.collection = collection_name(embedder_key)
        self._pending: list[Unit] = []

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
            # Embedded. Shared across videos, so it lives beside the data root
            # rather than inside one video's directory.
            store = Path(path or (paths.DATA_ROOT / "qdrant"))
            store.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(store))

    def _ensure(self, dims: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self._client.collection_exists(self.collection):
            return
        self._client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
        )

    # -- writing ---------------------------------------------------------
    def stored_hashes(self) -> dict[str, str]:
        """key -> text_hash for this video, so a re-run embeds what changed."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not self._client.collection_exists(self.collection):
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
        self._client.upsert(self.collection, points=[
            PointStruct(id=_point_id(u), vector=list(u.vector), payload={
                "key": u.key, "video_id": u.video_id, "chunk_id": u.chunk_id,
                "sampler_id": u.sampler_id, "text_hash": u.text_hash,
                "content": u.content, "structured": u.structured,
            }) for u in vectors])
        return len(vectors)

    def prune(self, live_keys: set[str]) -> int:
        """Drop points for chunks that no longer exist.

        A grid that shrank leaves points naming a chunk nobody can play. Unlike
        the file backend, which rewrites its whole document, an upserting store
        keeps whatever it was never told to remove.
        """
        from qdrant_client.models import PointIdsList

        stale = [k for k in self.stored_hashes() if k and k not in live_keys]
        if not stale:
            return 0
        import uuid
        self._client.delete(self.collection, points_selector=PointIdsList(
            points=[str(uuid.uuid5(uuid.NAMESPACE_URL, k)) for k in stale]))
        return len(stale)

    def save(self) -> str:
        # Written on upsert; nothing is buffered.
        return f"qdrant://{self.collection}"

    # -- reading ---------------------------------------------------------
    def search(self, vector: Sequence[float], query: str, limit: int = 20,
               sampler: Optional[str] = None) -> list[dict[str, Any]]:
        """Dense only. `text_rank` is always None, and that is the point."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not self._client.collection_exists(self.collection):
            return []
        must = [FieldCondition(key="video_id",
                              match=MatchValue(value=self.video_id))]
        if sampler is not None:
            must.append(FieldCondition(key="sampler_id",
                                       match=MatchValue(value=sampler)))
        found = self._client.query_points(
            self.collection, query=list(vector), limit=limit,
            query_filter=Filter(must=must), with_payload=True).points

        hits = []
        for rank, point in enumerate(found, start=1):
            payload = point.payload or {}
            hits.append({
                "chunk_id": payload.get("chunk_id"),
                "sampler_id": payload.get("sampler_id"),
                "content": payload.get("content", ""),
                "structured": payload.get("structured", {}),
                # No fusing: one ranking, so its own reciprocal is the score.
                "score": 1.0 / (60 + rank),
                "dense_rank": rank,
                "text_rank": None,
            })
        return hits


__all__ = ["QdrantIndex", "QdrantUnavailable", "collection_name"]
