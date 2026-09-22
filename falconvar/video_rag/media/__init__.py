"""1 · split -- one file in, two stream descriptions out.

The fork, and only the fork: it says what streams the file carries and what
each half needs to open its own decoder. No pixels, no waveform, no model.
"""

from __future__ import annotations

#: `run` is the only public spelling. The function in `driver.py` is named for
#: its component so a traceback frame says which one failed -- eight frames
#: called `run` carry no information -- but exporting both names would give the
#: library two ways to say the same thing, and `load`, `build` and `available`
#: collide across components anyway, so a bare-name style needs aliases the
#: moment a caller wants a second thing from the same module.
from .driver import load, main, run
from .split import UnusableMedia, split

__all__ = ["UnusableMedia", "load", "main", "run", "split"]
