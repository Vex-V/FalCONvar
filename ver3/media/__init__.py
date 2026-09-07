"""1 · split -- one file in, two stream descriptions out.

The fork, and only the fork: it says what streams the file carries and what
each half needs to open its own decoder. No pixels, no waveform, no model.
"""

from __future__ import annotations

from .driver import load, main, run
from .split import UnusableMedia, split

__all__ = ["UnusableMedia", "load", "main", "run", "split"]
