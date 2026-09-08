"""The Describer protocol, and a registry.

A describer takes frames and a context and returns a summary plus whatever
structured fields its question owns. Two exist: a stub that loads nothing, and
the OpenAI call.

The registry is lazy: importing this must not pull in `openai`, so a stub run
pays for no client and no key.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

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
    def describe(self, images: Sequence[LoadedFrame],
                 context: dict[str, Any]) -> Description: ...
    def config(self) -> dict[str, Any]: ...


_LAZY = {"openai": "backends.openai_client:OpenAIDescriber"}
_REGISTRY: dict[str, Any] = {}


def register(cls) -> Any:
    _REGISTRY[cls.name] = cls
    return cls


def build(name: str, **kwargs) -> Describer:
    if name in _REGISTRY:
        return _REGISTRY[name](**kwargs)
    if name not in _LAZY:
        raise KeyError(f"unknown describer {name!r}; known: {', '.join(available())}")
    module_name, class_name = _LAZY[name].split(":")
    module = importlib.import_module(f".{module_name}", __package__)
    return register(getattr(module, class_name))(**kwargs)


def available() -> list[str]:
    return sorted(set(_REGISTRY) | set(_LAZY))


__all__ = ["Describer", "DescriberUnavailable", "Description", "available",
           "build", "register"]
