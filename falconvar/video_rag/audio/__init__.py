"""What was said, and by whom. No chunks.

Decode the whole waveform, transcribe, diarize, attribute each word to a
speaker. The output is `transcript.raw.json` -- words, segments and turns, with
no chunk ids at all.

Whole-file is not an optimisation. Whisper carries context across an utterance,
and speaker labels come from clustering over the entire recording, so a
windowed run produces speakers that are not misaligned but unnameable.

This package does not know chunking exists. Boundaries are applied afterwards
by `cut`, and deriving boundaries from speech is `boundaries/speech.py`'s job.
"""

from __future__ import annotations

#: `run` is the only public spelling. The function in `driver.py` is named for
#: its component so a traceback frame says which one failed -- eight frames
#: called `run` carry no information -- but exporting both names would give the
#: library two ways to say the same thing, and `load`, `build` and `available`
#: collide across components anyway, so a bare-name style needs aliases the
#: moment a caller wants a second thing from the same module.
from .driver import load, main, run
from .reader import listen

__all__ = ["listen", "load", "main", "run"]
