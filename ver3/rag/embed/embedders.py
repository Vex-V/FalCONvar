"""The Embedder protocol, and a lazy registry.

**One vector space per embedder, never per sampler.** A space is defined by the
model, not by which prompt produced the text, so everything one embedder writes
is directly comparable. The sampler is payload and querying one is a *filter*;
separate spaces per sampler would partition one space while making
cross-sampler queries impossible.

Different embedders *do* get separate spaces, and the key carries
`name:model:dims` -- which is what stops 768-wide vectors being ranked against
1536-wide ones. A mismatch across widths fails loudly; a mismatch between two
models of the same width returns a well-formed ranking that means nothing,
which is the reason the key is in the collection name at all.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from typing import Any, Protocol, Sequence


class EmbedderUnavailable(Exception):
    """No client, no key, or a model this account cannot reach."""


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    @property
    def key(self) -> str: ...
    def config(self) -> dict[str, Any]: ...


class HashEmbedder:
    """Deterministic vectors from a hash. No model, no network.

    Not a semantic embedder and never pretends to be: it exists so the index,
    the ranking and the retrieval path can be exercised without a key. Its
    `key` says `hash`, so vectors it wrote can never be searched by a real
    embedder's query -- the collection name keeps them apart.
    """

    name = "hash"

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vector = [0.0] * self.dims
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dims
                vector[index] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            out.append([v / norm for v in vector])
        return out

    @property
    def key(self) -> str:
        return f"{self.name}:none:{self.dims}"

    def config(self) -> dict[str, Any]:
        return {"embedder": self.name, "dims": self.dims}


_LAZY = {"openai": "openai_embedder:OpenAIEmbedder"}
_REGISTRY: dict[str, Any] = {"hash": HashEmbedder}


def build(name: str, **kwargs) -> Embedder:
    if name in _REGISTRY:
        return _REGISTRY[name](**kwargs)
    if name not in _LAZY:
        raise KeyError(f"unknown embedder {name!r}; known: {', '.join(available())}")
    module_name, class_name = _LAZY[name].split(":")
    module = importlib.import_module(f".{module_name}", __package__)
    cls = getattr(module, class_name)
    _REGISTRY[name] = cls
    return cls(**kwargs)


def available() -> list[str]:
    return sorted(set(_REGISTRY) | set(_LAZY))


__all__ = ["Embedder", "EmbedderUnavailable", "HashEmbedder", "available", "build"]
