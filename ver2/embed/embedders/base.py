"""What an embedder is.

Two methods, not one, and the difference is load-bearing. Several strong
retrieval models are trained asymmetrically: a passage and a question are
embedded with different prefixes (``search_document:`` against
``search_query:`` for nomic, ``passage:`` against ``query:`` for e5). Calling
one method for both silently costs a large chunk of the model's accuracy, and
nothing about the output looks wrong -- the vectors still have the right shape
and the search still returns results, just worse ones.

``dimensions`` is fixed per model and is not a detail. A pgvector column is
declared ``vector(N)``; changing N means a new column and re-embedding
everything, so the index records which embedder produced it and refuses to mix.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence


class Embedder(Protocol):
    name: str
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Vectors for text being stored. Batched: callers pass many."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """A vector for a question, which is not the same operation."""
        ...

    def config(self) -> dict[str, Any]:
        """Recorded beside every vector, so an index says what made it."""
        ...
