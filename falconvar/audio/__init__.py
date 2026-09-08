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

from .driver import load, main, run
from .reader import listen

__all__ = ["listen", "load", "main", "run"]
