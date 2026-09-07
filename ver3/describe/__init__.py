"""7 · describe -- one VLM answer per (chunk, sampler).

Reads the frame store and nothing else. The question is the sampler's `prompt`,
falling back to its name; which keys a call may fill is narrowed by the other
*questions* on the chunk, so exactly one call answers each key.
"""

from __future__ import annotations

from .base import Describer, DescriberUnavailable, Description, available, build
from .driver import load, main, run
from .frames import FrameSource, LoadedFrame, StoreUnavailable
from .reader import describe

__all__ = ["Describer", "DescriberUnavailable", "Description", "FrameSource",
           "LoadedFrame", "StoreUnavailable", "available", "build", "describe",
           "load", "main", "run"]
