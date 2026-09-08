"""One model answer per (chunk, sampler).

Reads the frame store and nothing else. There is no seek-the-video fallback:
the store exists so this stage has its frames in hand, and a fallback would do
its job while leaving it broken, silently and ~40x slower.

The question asked is the sampler's `prompt`, falling back to its name. Which
keys a call's schema may fill is narrowed by the other *questions* on the same
chunk, so exactly one call answers each key and merging is a plain union.

Resume is keyed on the manifest, the describer and a hash of `prompts.py`
together: without all three, switching describers skips every pair and reports
success having done nothing.
"""

from __future__ import annotations

from .base import Describer, DescriberUnavailable, Description, available, build
from .driver import load, main, run
from .frames import FrameSource, LoadedFrame, StoreUnavailable
from .reader import describe

__all__ = ["Describer", "DescriberUnavailable", "Description", "FrameSource",
           "LoadedFrame", "StoreUnavailable", "available", "build", "describe",
           "load", "main", "run"]
