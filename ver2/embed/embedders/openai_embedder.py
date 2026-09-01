"""Embedding through the OpenAI API.

The same account and key the describer uses. One request carries many texts,
so indexing a whole video is a single call rather than one per description.

``text-embedding-3-small`` at 1536 dimensions is the default for a specific
reason beyond quality: pgvector's HNSW index does not accept a ``vector``
column wider than 2000 dimensions, so ``-3-large`` at 3072 cannot be indexed
without moving to ``halfvec``. 1536 stays inside every constraint and costs
about two cents per million tokens -- all of test1 is a fraction of a cent.

Symmetric by design: OpenAI embeddings need no prefix on either side, so
``embed_query`` and ``embed_documents`` differ only in batching.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

DEFAULT_MODEL = "text-embedding-3-small"
DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
KEY_VARS = ("OPENAI_API_KEY", "OPENAI_API")


class EmbedderUnavailable(Exception):
    """No key, no SDK, or the API refused in a way retrying will not fix."""


class OpenAIEmbedder:
    """Batched OpenAI embeddings. An ``Embedder``."""

    name = "openai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        dimensions: Optional[int] = None,
        batch_size: int = 64,
        api_key: Optional[str] = None,
        client: Any = None,
    ) -> None:
        self.model_name = model
        # The API can project to fewer dimensions on request; whatever is asked
        # for is what the index column has to be declared as.
        self.dimensions = dimensions or DIMENSIONS.get(model, 1536)
        self.batch_size = batch_size
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:            # pragma: no cover
                raise EmbedderUnavailable(
                    "the openai package is not installed: pip install openai"
                ) from exc
            key = api_key or next((os.environ[v] for v in KEY_VARS
                                   if os.environ.get(v)), None)
            if not key:
                raise EmbedderUnavailable(
                    "no OpenAI key: set " + " or ".join(KEY_VARS) + " in .env"
                )
            client = OpenAI(api_key=key)
        self.client = client

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            try:
                response = self.client.embeddings.create(
                    model=self.model_name, input=batch,
                    dimensions=self.dimensions,
                )
            except Exception as exc:               # noqa: BLE001
                raise EmbedderUnavailable(
                    f"embedding call failed ({self.model_name}): {exc}"
                ) from None
            # Order is guaranteed by index, not by position in the response.
            ordered = sorted(response.data, key=lambda d: d.index)
            vectors.extend([list(d.embedding) for d in ordered])
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model_name,
            "dimensions": self.dimensions,
            "params": {},
        }
