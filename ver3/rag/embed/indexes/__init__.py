"""Where vectors live. One protocol, three places to put them.

    local      one JSON file per (video, embedder). No service to run, and
               the only one that works with nothing installed.
    qdrant     embedded or served. Dense only.
    supabase   Postgres with pgvector, through the REST client.

**A backend is chosen out loud when it is the weaker one.** Qdrant is
dense-only, so a search against it silently drops a component measured at
0.429 against 0.714 top-1 on the same corpus. Both are real options, but
`--index qdrant` prints a note saying the ranking has no lexical half --
because a dense-only result and a hybrid one are indistinguishable on sight:
every hit merely lacks a `t` marker, which reads as "no lexical match for this
query" rather than "this index cannot have one".

**The embedder key is in the collection name, in every backend.** A mismatch
across widths fails loudly; a mismatch between two models of the *same* width
returns a well-formed ranking that means nothing. Putting the key in the name
turns the second into the first.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional, Protocol, Sequence

from ..units import Unit

#: name -> ("module:Class", has_lexical_half). Resolved on first use, so a
#: local run imports neither a Qdrant client nor a Postgres one.
_BACKENDS: dict[str, tuple[str, bool]] = {
    "local": ("local:LocalIndex", True),
    "qdrant": ("qdrant:QdrantIndex", False),
    "supabase": ("supabase:SupabaseIndex", True),
}


class VectorIndex(Protocol):
    """What every backend must do."""

    def stored_hashes(self) -> dict[str, str]: ...
    def upsert(self, units: Sequence[Unit]) -> int: ...
    def prune(self, live_keys: set[str]) -> int: ...
    def save(self) -> Any: ...
    def search(self, vector: Sequence[float], query: str, limit: int = 20,
               sampler: Optional[str] = None) -> list[dict[str, Any]]: ...


def build(name: str, video_id: str, embedder_key: str, **kwargs) -> VectorIndex:
    if name not in _BACKENDS:
        raise KeyError(f"unknown index {name!r}; known: {', '.join(available())}")
    module_name, class_name = _BACKENDS[name][0].split(":")
    module = importlib.import_module(f".{module_name}", __package__)
    return getattr(module, class_name)(video_id, embedder_key, **kwargs)


def has_lexical(name: str) -> bool:
    """Whether this backend can contribute a text ranking at all."""
    return _BACKENDS[name][1]


def available() -> list[str]:
    return sorted(_BACKENDS)


__all__ = ["VectorIndex", "available", "build", "has_lexical"]
