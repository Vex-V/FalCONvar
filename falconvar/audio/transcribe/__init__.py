"""Transcribers, addressable by name.

The same shape as `describers` and `embedders`: a name maps to a class,
model-backed ones are imported lazily, and the CLI can build only what is
registered. The stub needs nothing installed and must stay that way.
"""

from __future__ import annotations

from typing import Type

from .base import Segment, Transcriber, Transcript, Word
from .stub import StubTranscriber

_LAZY: dict[str, str] = {"whisper": "whisper:WhisperTranscriber"}
_REGISTRY: dict[str, Type] = {}


def register(cls: Type) -> Type:
    _REGISTRY[cls.name] = cls
    return cls


register(StubTranscriber)


def _resolve(name: str) -> Type:
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name not in _LAZY:
        raise KeyError(f"unknown transcriber {name!r}; known: {', '.join(available())}")
    import importlib

    module_name, class_name = _LAZY[name].split(":")
    module = importlib.import_module(f".{module_name}", __package__)
    return register(getattr(module, class_name))


def build(name: str, **kwargs) -> Transcriber:
    return _resolve(name)(**kwargs)


def available() -> list[str]:
    return sorted(set(_REGISTRY) | set(_LAZY))


__all__ = ["Segment", "Transcriber", "Transcript", "Word", "StubTranscriber",
           "available", "build", "register"]
