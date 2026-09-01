"""Sampling strategies, addressable by name.

A new sampler subclasses Sampler, implements propose(), and is registered
here; it then becomes available to build() and to the driver's CLI.

Model-backed samplers are imported lazily. Importing this package must not
pull in torch, ultralytics or easyocr -- the uniform baseline has to stay
usable, and testable, on a machine with none of them installed.
"""

from __future__ import annotations

from typing import Type

from .base import Sampler
from .uniform import UniformSampler

# name -> "module:ClassName", resolved on first use.
# The registry name is what a manifest, a description row and an embedding are
# all keyed by, so it is stable even when the file it lives in is renamed. The
# file says what the sampler is about; the name says what runs.
_LAZY: dict[str, str] = {
    "clip": "scene:ClipChangeSampler",
    "yolo": "people:PersonChangeSampler",
    "objects": "objects:ObjectChangeSampler",
    "text": "ocr:TextChangeSampler",
}
_REGISTRY: dict[str, Type[Sampler]] = {}


def register(cls: Type[Sampler]) -> Type[Sampler]:
    _REGISTRY[cls.name] = cls
    return cls


register(UniformSampler)


def _resolve(name: str) -> Type[Sampler]:
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name not in _LAZY:
        known = ", ".join(available())
        raise KeyError(f"unknown sampler {name!r}; known: {known}") from None
    import importlib

    module_name, class_name = _LAZY[name].split(":")
    module = importlib.import_module(f".{module_name}", __package__)
    return register(getattr(module, class_name))


def build(name: str, **kwargs) -> Sampler:
    """Instantiate a registered sampler by name."""
    return _resolve(name)(**kwargs)


def available() -> list[str]:
    return sorted(set(_REGISTRY) | set(_LAZY))


__all__ = ["Sampler", "UniformSampler", "available", "build", "register"]
