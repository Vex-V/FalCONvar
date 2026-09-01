"""Embedders, addressable by name.

Same shape as `ingest.samplers` and `describe.describers`: a name maps to a
class, heavy ones are imported lazily, and the CLI can build only what is
registered. Nothing here imports torch or the OpenAI SDK until asked to.

Which one is best for this data is a question to measure, not to assume, and
that is the point of the interface. See `local.KNOWN` for the shortlist and
why context length rules out the popular 512-token models.
"""

from __future__ import annotations

from typing import Type

from .base import Embedder

_LAZY: dict[str, str] = {
    "openai": "openai_embedder:OpenAIEmbedder",
    "local": "local:LocalEmbedder",
}
_REGISTRY: dict[str, Type] = {}


def register(cls: Type) -> Type:
    _REGISTRY[cls.name] = cls
    return cls


def _resolve(name: str) -> Type:
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name not in _LAZY:
        raise KeyError(f"unknown embedder {name!r}; known: {', '.join(available())}")
    import importlib

    module_name, class_name = _LAZY[name].split(":")
    module = importlib.import_module(f".{module_name}", __package__)
    return register(getattr(module, class_name))


def build(name: str, **kwargs) -> Embedder:
    return _resolve(name)(**kwargs)


def available() -> list[str]:
    return sorted(set(_REGISTRY) | set(_LAZY))


__all__ = ["Embedder", "available", "build", "register"]
