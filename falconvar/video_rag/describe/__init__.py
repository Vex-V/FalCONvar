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
#: `run` is the only public spelling. The function in `driver.py` is named for
#: its component so a traceback frame says which one failed -- eight frames
#: called `run` carry no information -- but exporting both names would give the
#: library two ways to say the same thing, and `load`, `build` and `available`
#: collide across components anyway, so a bare-name style needs aliases the
#: moment a caller wants a second thing from the same module.
from .driver import load, main, run
from .frames import FrameSource, LoadedFrame, StoreUnavailable
from .reader import answer

__all__ = ["answer", "Describer", "DescriberUnavailable", "Description", "FrameSource",
           "LoadedFrame", "StoreUnavailable", "available", "build",
           "load", "main", "run"]
