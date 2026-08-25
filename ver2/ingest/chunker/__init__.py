"""Chunking strategies, addressable by name.

SceneChunker is imported lazily so this package does not pull in
PySceneDetect for a run that only needs uniform windows.
"""

from __future__ import annotations

from typing import Type

from .base import Chunker
from .uniform import UniformChunker

_LAZY: dict[str, str] = {"scene": "scene:SceneChunker"}
_REGISTRY: dict[str, Type[Chunker]] = {}


def register(cls: Type[Chunker]) -> Type[Chunker]:
    _REGISTRY[cls.name] = cls
    return cls


register(UniformChunker)


def _resolve(name: str) -> Type[Chunker]:
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name not in _LAZY:
        known = ", ".join(available())
        raise KeyError(f"unknown chunker {name!r}; known: {known}") from None
    import importlib

    module_name, class_name = _LAZY[name].split(":")
    module = importlib.import_module(f".{module_name}", __package__)
    return register(getattr(module, class_name))


def build(name: str, **kwargs) -> Chunker:
    """Instantiate a registered chunker by name."""
    return _resolve(name)(**kwargs)


def available() -> list[str]:
    return sorted(set(_REGISTRY) | set(_LAZY))


__all__ = ["Chunker", "UniformChunker", "available", "build", "register"]
