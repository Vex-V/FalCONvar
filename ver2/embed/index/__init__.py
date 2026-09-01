"""Where vectors live: locally in Qdrant, shared in Postgres, or both."""

from .base import Hit, VectorIndex
from .multi import MultiIndex


def __getattr__(name: str):
    if name == "QdrantIndex":
        from .qdrant import QdrantIndex
        return QdrantIndex
    if name == "PgVectorIndex":
        from .pgvector import PgVectorIndex
        return PgVectorIndex
    raise AttributeError(name)


__all__ = ["Hit", "MultiIndex", "PgVectorIndex", "QdrantIndex", "VectorIndex"]
