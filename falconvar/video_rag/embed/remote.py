"""Vectors from an API: OpenAI, or anything serving OpenAI's /embeddings.

Ollama, LM Studio, llama.cpp, vLLM, Gemini, Mistral, Together and Voyage all
answer the same request, so one class covers them and a provider differs only
in its base URL, its key, and two flags -- whether it takes `dimensions`, and
whether it wants `input_type`.

**The width is part of the key, so it has to be known before the index is
opened.** OpenAI's are known; anything else is found by embedding one short
string, once per process. Every later batch is checked against it, because a
server swapping the model behind a name would otherwise write vectors of a new
width into a space keyed by the old one.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ...shared.models import providers
from .embedders import EmbedderUnavailable, key_for, prefixes_for

#: Widths that need no probe.
KNOWN_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072,
              "text-embedding-ada-002": 1536}

_PROBED: dict[tuple[str, str, str, int], int] = {}


class RemoteEmbedder:
    def __init__(self, provider: str = "openai", model: Optional[str] = None,
                 dims: Optional[int] = None,
                 query_prefix: Optional[str] = None,
                 document_prefix: Optional[str] = None,
                 client: Any = None) -> None:
        self.provider = providers.get(provider)
        self.name = self.provider.name
        self.model = model or self.provider.embed_model or ""
        if not self.model:
            raise EmbedderUnavailable(f"{self.name} has no default embedding model")
        auto_query, auto_document = prefixes_for(self.model)
        self.query_prefix = auto_query if query_prefix is None else query_prefix
        self.document_prefix = auto_document if document_prefix is None else document_prefix
        #: Asked for, where the provider can make it; otherwise only checked.
        self.requested = int(dims) if dims else None
        self._dims = self.requested or (KNOWN_DIMS.get(self.model)
                                        if self.provider.protocol == "openai" else None)
        self._client = client if client is not None else self._connect()

    def _connect(self) -> Any:
        from openai import OpenAI
        try:
            key = providers.api_key(self.provider)
        except providers.ProviderUnavailable as exc:
            raise EmbedderUnavailable(str(exc)) from None
        options: dict[str, Any] = {"api_key": key or "not-needed"}
        url = providers.base_url(self.provider)
        if url:
            options["base_url"] = url
        return OpenAI(**options)

    def _call(self, texts: Sequence[str], side: str) -> list[list[float]]:
        extra: dict[str, Any] = {}
        if self.requested and self.provider.dimensions:
            extra["dimensions"] = self.requested
        if self.provider.input_type:
            extra["extra_body"] = {"input_type": side}
        try:
            response = self._client.embeddings.create(model=self.model,
                                                      input=list(texts), **extra)
        except Exception as exc:                            # noqa: BLE001
            where = providers.base_url(self.provider) or self.name
            raise EmbedderUnavailable(
                f"{self.name} could not embed with {self.model!r} ({where}): {exc}") from None
        data = sorted(response.data, key=lambda d: getattr(d, "index", 0))
        vectors = [list(d.embedding) for d in data]
        if len(vectors) != len(texts):
            raise EmbedderUnavailable(
                f"{self.name} returned {len(vectors)} vectors for {len(texts)} texts")
        return vectors

    def _checked(self, vectors: list[list[float]]) -> list[list[float]]:
        for vector in vectors:
            if self._dims is None:
                self._dims = len(vector)
            elif len(vector) != self._dims:
                raise EmbedderUnavailable(
                    f"{self.name}:{self.model} returned {len(vector)} dimensions "
                    f"where {self._dims} were expected -- a different model behind "
                    "the same name writes into a space it does not belong to")
        return vectors

    @property
    def dims(self) -> int:
        if self._dims is None:
            probe = (self.name, providers.base_url(self.provider) or "", self.model,
                     self.requested or 0)
            if probe not in _PROBED:
                _PROBED[probe] = len(self._call(["width probe"], "document")[0])
            self._dims = _PROBED[probe]
        return self._dims

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._checked(self._call([self.document_prefix + t for t in texts],
                                        "document"))

    def embed_query(self, text: str) -> list[float]:
        return self._checked(self._call([self.query_prefix + text], "query"))[0]

    @property
    def key(self) -> str:
        return key_for(self.name, self.model, self.dims,
                       self.query_prefix, self.document_prefix)

    def config(self) -> dict[str, Any]:
        out: dict[str, Any] = {"embedder": self.name, "model": self.model,
                               "dims": self.dims}
        if self.query_prefix or self.document_prefix:
            out.update(query_prefix=self.query_prefix,
                       document_prefix=self.document_prefix)
        return out


__all__ = ["KNOWN_DIMS", "RemoteEmbedder"]
