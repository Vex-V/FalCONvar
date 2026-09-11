"""OpenAI embeddings. Imported only when asked for."""

from __future__ import annotations

from typing import Any, Sequence

# The key's names live in one place. Two modules each spelling out their own
# list is how `OPENAI_API` worked for the describer and not for this.
from ...shared.llm import LLMUnavailable, api_key
from .embedders import EmbedderUnavailable

DEFAULT_MODEL = "text-embedding-3-small"
DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str = DEFAULT_MODEL, client: Any = None) -> None:
        self.model = model
        self.dims = DIMS.get(model, 1536)
        if client is not None:
            self._client = client
            return
        try:
            key = api_key()
        except LLMUnavailable as exc:
            raise EmbedderUnavailable(str(exc)) from None
        try:
            from openai import OpenAI
        except ImportError as exc:                       # pragma: no cover
            raise EmbedderUnavailable("openai is not installed") from exc
        self._client = OpenAI(api_key=key)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model,
                                                  input=list(texts))
        return [d.embedding for d in response.data]

    @property
    def key(self) -> str:
        return f"{self.name}:{self.model}:{self.dims}"

    def config(self) -> dict[str, Any]:
        return {"embedder": self.name, "model": self.model, "dims": self.dims}
