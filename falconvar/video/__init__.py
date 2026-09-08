"""5 · ingest -- which frames are worth describing, and why.

One decode pass, feeding every sampler from it. The grid arrives as data and is
never edited: this asks `timeline.nearest(ts)` and nothing else.

**Pixels are converted only for frames that survive decimation** -- 4% of them
at 1/s from 25 fps -- which is the difference between 3 minutes and 37 on a
three-hour file. See `reader.py`.
"""

from __future__ import annotations

from .driver import build_samplers, load, main, run
from .pipeline import ingest
from .reader import Frame, UnreadableSource
from .store import FrameStore

__all__ = ["Frame", "FrameStore", "UnreadableSource", "build_samplers",
           "ingest", "load", "main", "run"]
