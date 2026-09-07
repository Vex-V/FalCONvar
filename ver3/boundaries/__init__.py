"""3 + 4 · where the chunk boundaries fall.

Everything that decides a boundary lives here, from both modalities. That is
the one departure from `falconvar` that the rest of the design rests on: there,
the grid is produced *by* whichever pass ran first and read back out of its
results, which is why there are four ordering cases in `orchestrate.process`, a
`Chunker` protocol and a `Timeline` class describing the same idea, and a
`FixedChunker` bridging them.

Two evidence sources with nothing in common but their output:

    scenes.py   the picture. Decodes video, scores frame-to-frame content
                difference, thresholds it into cuts. The expensive one.
    speech.py   the soundtrack. Reads a finished transcript and finds silences
                or speaker changes. Arithmetic.

    grid.py     cuts + a duration -> spans, with the guards applied. The only
                thing here that decides a *grid*, as opposed to evidence.

`speech.py` lives here rather than in `listen/` on purpose: deriving cuts from
pauses is boundary logic that happens to read a transcript, and putting it in
the audio module would make that module know chunking exists. It reads
`transcript.raw.json` as a **file**, never by importing `listen` -- which is
what keeps the import graph acyclic while the *run* order flips between
policies. On a `vad` run `listen` precedes this package; on a `scene` run it
does not; neither imports the other in either case.

The name is `boundaries` rather than `chunker` because `falconvar` used
"chunker" for the streaming protocol this design deletes.
"""

from __future__ import annotations

from .driver import evidence, load, load_cuts, main, retune, run
from .grid import POLICIES, build, enforce, from_cuts, merge_tail, uniform

__all__ = ["POLICIES", "build", "enforce", "evidence", "from_cuts", "load",
           "load_cuts", "main", "merge_tail", "retune", "run", "uniform"]
