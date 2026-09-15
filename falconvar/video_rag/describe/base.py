"""The Describer protocol, and a registry.

A describer takes frames and a context and returns a summary plus whatever
structured fields its question owns. Two kinds exist: a stub that loads nothing,
and `backends.model.ModelDescriber`, which any provider in `shared.models.providers`
answers through.

Resolution is lazy: importing this must not pull in a client, so a stub run
pays for no SDK and no key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence

from .frames import LoadedFrame


class DescriberUnavailable(Exception):
    """No client, no key, or a model this account cannot reach."""


@dataclass
class Description:
    """One answer about one (chunk, sampler)."""

    summary: str
    fields: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class Describer(Protocol):
    #: How many (chunk, sampler) runs `reader.describe` works on at once.
    concurrency: int

    async def describe(self, images: Sequence[LoadedFrame],
                       context: dict[str, Any]) -> Description: ...
    def config(self) -> dict[str, Any]: ...


#: Describers that are not a model provider: the stub, which loads nothing.
_REGISTRY: dict[str, Any] = {}


def register(cls) -> Any:
    _REGISTRY[cls.name] = cls
    return cls


def build(name: Optional[str] = None, **kwargs) -> Describer:
    """A provider, `provider/model`, `stub`, or None for the default.

    Every provider is one class; which wire format it speaks is `shared.models.llm`'s
    concern, so adding a provider adds no describer.
    """
    from ...shared.models import providers

    chosen, _ = providers.choose("describe", name)
    if chosen == providers.OFFLINE["describe"]:
        from .backends import stub  # noqa: F401  -- self-registers
    if chosen in _REGISTRY:
        return _REGISTRY[chosen]()
    from .backends.model import ModelDescriber
    return ModelDescriber(name, **kwargs)


def available() -> list[str]:
    from ...shared.models import providers
    return sorted(set(_REGISTRY) | set(providers.names("describe")))


__all__ = ["Describer", "DescriberUnavailable", "Description", "available",
           "build", "register"]
