"""Diarizers, addressable by name.

`none` is a real choice rather than a placeholder: a single-speaker recording,
or one where only the words matter, should not pay for a second model or a
gated download. It returns no turns, and `align` then leaves every word
unattributed, which is a truthful answer.
"""

from __future__ import annotations

from typing import Any, Type

from ..source import Track
from .base import Diarization, Diarizer, Turn


class NoDiarizer:
    """Answers 'nobody was identified'. A ``Diarizer``."""

    name = "none"

    def diarize(self, track: Track) -> Diarization:
        return Diarization(turns=[], embeddings=None, model=self.config())

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "params": {}}


_LAZY: dict[str, str] = {"pyannote": "pyannote_diarizer:PyannoteDiarizer"}
_REGISTRY: dict[str, Type] = {}


def register(cls: Type) -> Type:
    _REGISTRY[cls.name] = cls
    return cls


register(NoDiarizer)


def _resolve(name: str) -> Type:
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name not in _LAZY:
        raise KeyError(f"unknown diarizer {name!r}; known: {', '.join(available())}")
    import importlib

    module_name, class_name = _LAZY[name].split(":")
    module = importlib.import_module(f".{module_name}", __package__)
    return register(getattr(module, class_name))


def build(name: str, **kwargs) -> Diarizer:
    return _resolve(name)(**kwargs)


def available() -> list[str]:
    return sorted(set(_REGISTRY) | set(_LAZY))


__all__ = ["Diarization", "Diarizer", "NoDiarizer", "Turn",
           "available", "build", "register"]
