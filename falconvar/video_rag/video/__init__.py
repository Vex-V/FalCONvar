"""5 · ingest -- which frames are worth describing, and why.

One decode pass, feeding every sampler from it. The grid arrives as data and is
never edited: this asks `timeline.nearest(ts)` and nothing else.

**Pixels are converted only for frames that survive decimation** -- 4% of them
at 1/s from 25 fps -- which is the difference between 3 minutes and 37 on a
three-hour file. See `reader.py`.
"""

from __future__ import annotations

#: `run` is the only public spelling. The function in `driver.py` is named for
#: its component so a traceback frame says which one failed -- eight frames
#: called `run` carry no information -- but exporting both names would give the
#: library two ways to say the same thing, and `load`, `build` and `available`
#: collide across components anyway, so a bare-name style needs aliases the
#: moment a caller wants a second thing from the same module.
from .driver import SAMPLER_SETTINGS, build_samplers, load, main, run
from .pipeline import ingest
from .reader import Frame, UnreadableSource
from .store import FrameStore

__all__ = ["SAMPLER_SETTINGS", "Frame", "FrameStore", "UnreadableSource", "build_samplers",
           "ingest", "load", "main", "run"]
