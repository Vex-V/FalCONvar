"""Where vectors live: locally in Qdrant, shared in Postgres, or both.

``build`` is here rather than in a driver because both CLIs construct the same
thing. `embed` writes to an index and `retrieve` reads from one, and if they
disagreed about what ``qdrant,pgvector`` names, a search would query somewhere
the vectors were never written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .base import Hit, VectorIndex
from .multi import MultiIndex
from .qdrant import DEFAULT_PATH as DEFAULT_QDRANT_PATH

#: Every backend a name may refer to. The first one named is the primary.
BACKENDS = ("qdrant", "pgvector")


def build(names: Iterable[str],
          qdrant_path: str | Path = DEFAULT_QDRANT_PATH) -> Any:
    """One index, or a fan-out over several with the first named as primary.

    Imported per name rather than up front, so a qdrant-only run never loads
    the Supabase client and a pgvector-only run never loads qdrant_client.
    """
    built = []
    for name in names:
        if name == "qdrant":
            from .qdrant import QdrantIndex

            built.append(QdrantIndex(qdrant_path))
        elif name == "pgvector":
            from .pgvector import PgVectorIndex

            built.append(PgVectorIndex())
        else:
            raise ValueError(
                f"unknown index backend {name!r}; expected "
                + " and/or ".join(BACKENDS))
    if not built:
        raise ValueError("no index named; expected " + " and/or ".join(BACKENDS))
    return built[0] if len(built) == 1 else MultiIndex(*built)


def __getattr__(name: str):
    if name == "QdrantIndex":
        from .qdrant import QdrantIndex
        return QdrantIndex
    if name == "PgVectorIndex":
        from .pgvector import PgVectorIndex
        return PgVectorIndex
    raise AttributeError(name)


__all__ = ["BACKENDS", "DEFAULT_QDRANT_PATH", "Hit", "MultiIndex",
           "PgVectorIndex", "QdrantIndex", "VectorIndex", "build"]
