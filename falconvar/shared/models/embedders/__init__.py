"""The Embedder protocol, and resolving a name to one.

**One vector space per embedder, never per sampler.** A space is defined by the
model, not by which prompt produced the text, so everything one embedder writes
is directly comparable. The sampler is payload and querying one is a *filter*;
separate spaces per sampler would partition one space while making
cross-sampler queries impossible.

Different embedders *do* get separate spaces, and the key carries
`provider:model:dims` -- which is what stops 768-wide vectors being ranked
against 1536-wide ones. A mismatch across widths fails loudly; a mismatch
between two models of the same width returns a well-formed ranking that means
nothing, which is the reason the key is in the collection name at all.

**A query and a document are embedded differently where the model says so.**
e5, nomic and bge were trained with a prefix on one side or both, and embedding
a query as a document costs them recall with no error anywhere. The prefixes
are a property of the model, so they are looked up rather than configured, and
a prefix set by hand changes the key -- the vectors it makes are not
comparable to the defaults'.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Optional, Protocol, Sequence
from ...errors import Unavailable


class EmbedderUnavailable(Unavailable):
    """No client, no key, or a model this account cannot reach."""


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    @property
    def key(self) -> str: ...
    def config(self) -> dict[str, Any]: ...


def query_vector(embedder: Any, text: str) -> list[float]:
    """The vector for a search, through the query side where there is one."""
    side = getattr(embedder, "embed_query", None)
    return side(text) if side else embedder.embed([text])[0]


_BGE = "Represent this sentence for searching relevant passages: "

#: model id fragment -> (query prefix, document prefix). Matched lowercase,
#: first hit wins. From each family's model card.
PREFIXES: tuple[tuple[str, str, str], ...] = (
    ("multilingual-e5", "query: ", "passage: "),
    ("e5-", "query: ", "passage: "),
    ("nomic-embed-text", "search_query: ", "search_document: "),
    ("bge-small-en", _BGE, ""),
    ("bge-base-en", _BGE, ""),
    ("bge-large-en", _BGE, ""),
    ("mxbai-embed-large", _BGE, ""),
)


def prefixes_for(model: str) -> tuple[str, str]:
    lowered = model.lower()
    for fragment, query, document in PREFIXES:
        if fragment in lowered:
            return query, document
    return "", ""


def key_for(name: str, model: str, dims: int,
            query_prefix: str, document_prefix: str) -> str:
    """`name:model:dims`, plus a suffix only when the prefixes are not the
    model's own -- so every key written before prefixes existed is unchanged."""
    key = f"{name}:{model}:{dims}"
    if (query_prefix, document_prefix) != prefixes_for(model):
        digest = hashlib.sha1(f"{query_prefix}\0{document_prefix}".encode()).hexdigest()
        key += f":p{digest[:6]}"
    return key


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

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def key(self) -> str:
        return f"{self.name}:none:{self.dims}"

    def config(self) -> dict[str, Any]:
        return {"embedder": self.name, "dims": self.dims}


def build(name: Optional[str] = None, **kwargs) -> Embedder:
    """A provider name, `provider/model`, `hash`, or None for the default.

    Every provider resolves through `shared.models.providers`, so `embed` and
    `retrieve` reading the same environment cannot disagree about which space
    a search belongs to.
    """
    from .. import providers

    chosen, model = providers.choose("embed", name)
    if chosen == providers.OFFLINE["embed"]:
        return HashEmbedder(**({"dims": kwargs["dims"]} if kwargs.get("dims") else {}))
    provider = providers.get(chosen)
    if provider.protocol == "local":
        from .local import LocalEmbedder
        return LocalEmbedder(model, **kwargs)
    from .remote import RemoteEmbedder
    return RemoteEmbedder(chosen, model, **kwargs)


def available() -> list[str]:
    from .. import providers
    return providers.names("embed")


__all__ = ["Embedder", "EmbedderUnavailable", "HashEmbedder", "PREFIXES",
           "available", "build", "key_for", "prefixes_for", "query_vector"]
