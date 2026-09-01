"""Qdrant, embedded -- a directory, not a server.

``QdrantClient(path=...)`` runs the engine in-process against a local
directory. No Docker, no daemon, nothing to start before a search works, and
the index sits beside the manifest and the frame store under `out/`. For a
corpus this size that is not a compromise: the same library, the same query
semantics, one fewer moving part.

One collection per embedder, named after it. Two embedders' vectors are not
comparable, and a collection has a fixed width, so the name carrying the
embedder key is what stops a 768-wide vector being searched against 1536-wide
neighbours.

Dense only. Qdrant can carry sparse vectors too, and `bge-m3` emits them, but
the text half of the hybrid currently lives in Postgres. What that means in
practice is written down in `search.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from ..units import Unit, collection_name, embedder_key
from .base import Hit

DEFAULT_PATH = Path("out") / "qdrant"


class QdrantIndex:
    """A local on-disk Qdrant collection. A ``VectorIndex``."""

    name = "qdrant"

    def __init__(self, path: str | Path = DEFAULT_PATH, client: Any = None) -> None:
        self.path = Path(path)
        if client is None:
            from qdrant_client import QdrantClient

            self.path.mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=str(self.path))
        self.client = client

    def _collection(self, embedder: dict[str, Any]) -> str:
        return collection_name(embedder_key(embedder))

    def ensure(self, embedder: dict[str, Any]) -> None:
        from qdrant_client import models

        name = self._collection(embedder)
        if self.client.collection_exists(name):
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=int(embedder["dimensions"]),
                distance=models.Distance.COSINE,
            ),
        )

    def existing(self, video_id: str, embedder: dict[str, Any]) -> dict[tuple[int, str], str]:
        from qdrant_client import models

        name = self._collection(embedder)
        if not self.client.collection_exists(name):
            return {}
        found: dict[tuple[int, str], str] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=name,
                scroll_filter=models.Filter(must=[models.FieldCondition(
                    key="video_id", match=models.MatchValue(value=video_id))]),
                with_payload=True, with_vectors=False, limit=256, offset=offset,
            )
            for point in points:
                p = point.payload or {}
                found[(p["chunk_id"], p["sampler"])] = p.get("text_hash", "")
            if offset is None:
                return found

    def upsert(self, units: Sequence[Unit], vectors: Sequence[Sequence[float]],
               embedder: dict[str, Any]) -> None:
        from qdrant_client import models

        self.ensure(embedder)
        self.client.upsert(
            collection_name=self._collection(embedder),
            points=[
                models.PointStruct(id=unit.point_id, vector=list(vector),
                                   payload=unit.payload())
                for unit, vector in zip(units, vectors)
            ],
        )

    def search(self, vector: Sequence[float], embedder: dict[str, Any],
               query_text: Optional[str] = None, video_id: Optional[str] = None,
               limit: int = 20, sampler: Optional[str] = None) -> list[Hit]:
        from qdrant_client import models

        name = self._collection(embedder)
        if not self.client.collection_exists(name):
            return []
        # Filtering, not a separate space: every vector here came from the same
        # model and is comparable to every other. The sampler narrows which
        # question's answers are searched, it does not change the geometry.
        must = []
        if video_id:
            must.append(models.FieldCondition(
                key="video_id", match=models.MatchValue(value=video_id)))
        if sampler:
            must.append(models.FieldCondition(
                key="sampler", match=models.MatchValue(value=sampler)))
        flt = models.Filter(must=must) if must else None
        found = self.client.query_points(
            collection_name=name, query=list(vector), query_filter=flt,
            limit=limit, with_payload=True,
        ).points
        hits: list[Hit] = []
        for rank, point in enumerate(found, start=1):
            p = point.payload or {}
            hits.append(Hit(
                video_id=p["video_id"], chunk_id=p["chunk_id"], sampler=p["sampler"],
                score=float(point.score), content=p.get("content", ""),
                start_ts=p.get("start_ts"), end_ts=p.get("end_ts"),
                frame_indexes=p.get("frame_indexes"), vector_rank=rank,
            ))
        return hits
