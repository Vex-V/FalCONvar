"""Where the chunk boundaries fall.

Everything that decides a boundary, from both modalities.

    scenes.py   the picture: decodes video, scores frame-to-frame difference,
                thresholds it into cuts
    speech.py   the soundtrack: reads a finished transcript for silences or
                speaker changes
    grid.py     cuts + a duration -> spans, with the guards applied

`speech.py` lives here rather than in `audio/` so that package never learns
chunking exists. It reads `transcript.raw.json` as a file, never by importing
`audio` -- which is what keeps the import graph acyclic while the run order
flips between policies.
"""

from __future__ import annotations

from .driver import evidence, load, load_cuts, main, retune, run
from .grid import POLICIES, build, enforce, from_cuts, merge_tail, uniform

__all__ = ["POLICIES", "build", "enforce", "evidence", "from_cuts", "load",
           "load_cuts", "main", "merge_tail", "retune", "run", "uniform"]
