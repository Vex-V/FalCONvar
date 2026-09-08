"""Where vectors live.

    qdrant     embedded (a path) or served (a url). The default.
    supabase   Postgres with pgvector, through the REST client.

Both fuse a dense and a lexical ranking. `has_lexical` describes what a backend
was built to do, so a False there is a gap in this code rather than in the
store.

`tokenize` lives here rather than in one backend: a query has to be split the
same way the documents were, whichever index answers it. It keeps `£1.85`, `9p`
and `1:23` whole -- the literal strings the lexical half is best at -- and drops
stopwords, which Postgres gets free from `to_tsvector('english', ...)`.
"""

from __future__ import annotations

import importlib
import re
from typing import Any, Optional, Protocol, Sequence

from ..units import Unit

#: How text is split, for every backend that matches on words. A query has to
#: be tokenised the same way the documents were, so this lives with the package
#: rather than with whichever index happens to define it first.
#:
#: Keeps `£1.85`, `9p` and `1:23` whole -- those are exactly the literal strings
#: the lexical half is best at, and a tokeniser that split them would throw away
#: the advantage it is there for.
_WORD = re.compile(r"[a-z0-9£$€%.:'-]+")

#: Dropped before matching. Postgres gets this free from
#: `to_tsvector('english', ...)`; the other two backends have to say it.
#:
#: Not an optimisation. IDF already makes a stopword's *score* negligible, so
#: the ranking barely moves -- what it fixes is the `t` marker. Without it, the
#: query "youngsters fleeing a poisoned town" matched a chunk on the word "a"
#: and reported a text rank, so a marker meant to say "the words matched too"
#: fired on every query and distinguished nothing.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to from by
for with without into onto over under again further is are was were be been
being am do does did doing have has had having i you he she it we they them
his her its our their as so such no nor not only own same too very can will
just should now there here when where why how all any both each few more most
other some what which who whom
""".split())


def tokenize(text: str) -> list[str]:
    """Words, lowercased, stopwords removed.

    Keeps `£1.85`, `9p` and `1:23` whole -- those are exactly the literal
    strings the lexical half is best at, and splitting them would throw away
    the advantage it exists for.
    """
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]

#: name -> ("module:Class", has_lexical_half). Resolved on first use, so a
#: local run imports neither a Qdrant client nor a Postgres one.
_BACKENDS: dict[str, tuple[str, bool]] = {
    "qdrant": ("qdrant:QdrantIndex", True),
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


__all__ = ["STOPWORDS", "VectorIndex", "available", "build", "has_lexical",
           "tokenize"]
