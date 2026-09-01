"""Describers, addressable by name.

The same shape as `ingest.samplers`: a name maps to a class, model-backed ones
are imported lazily, and the CLI can only build what is registered here. The
stub needs nothing installed and must stay that way -- it is how the stage is
exercised without a model.

A VLM lands here as one more entry in `_LAZY`, and nothing else changes.
"""

from __future__ import annotations

from typing import Type

from .base import Describer, Description
from .stub import StubDescriber

# name -> "module:ClassName", resolved on first use, so importing this package
# never pulls in a model runtime or a network client. A dotted path starting
# with `ver2.` is imported absolutely; anything else is relative to here.
_LAZY: dict[str, str] = {
    "openai": "ver2.video.describe.vlm.openai_client:OpenAIDescriber",
}
_REGISTRY: dict[str, Type] = {}


def register(cls: Type) -> Type:
    _REGISTRY[cls.name] = cls
    return cls


register(StubDescriber)


def _resolve(name: str) -> Type:
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name not in _LAZY:
        raise KeyError(f"unknown describer {name!r}; known: {', '.join(available())}")
    import importlib

    module_name, class_name = _LAZY[name].split(":")
    module = (importlib.import_module(module_name) if module_name.startswith("ver2.")
              else importlib.import_module(f".{module_name}", __package__))
    return register(getattr(module, class_name))


def build(name: str, **kwargs):
    """Instantiate a registered describer by name."""
    return _resolve(name)(**kwargs)


def available() -> list[str]:
    return sorted(set(_REGISTRY) | set(_LAZY))


__all__ = ["Describer", "Description", "StubDescriber", "available", "build", "register"]
