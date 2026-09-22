"""The raw transcript, onto the grid.

Cheap and repeatable: Whisper timestamps every word, so a transcript can be
re-cut to any grid without touching a model.

A word belongs to the chunk containing its midpoint -- a word straddling a
boundary belongs to whichever side holds more of it, and every word must land
in exactly one chunk or the text is duplicated or dropped.

Chunks with no speech are kept with empty text: `chunk_id` is shared with the
manifest, so dropping the quiet ones renumbers everything after them.

A chunk carries `turns`, one bound record per contiguous run of one voice, so a
window holding three speakers still says who said what.
"""

from __future__ import annotations

#: `run` is the only public spelling. The function in `driver.py` is named for
#: its component so a traceback frame says which one failed -- eight frames
#: called `run` carry no information -- but exporting both names would give the
#: library two ways to say the same thing, and `load`, `build` and `available`
#: collide across components anyway, so a bare-name style needs aliases the
#: moment a caller wants a second thing from the same module.
from .driver import load, main, run
from .cutter import to_chunks

__all__ = ["load", "main", "run", "to_chunks"]
