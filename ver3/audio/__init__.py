"""2 · listen -- what was said, and by whom. No chunks.

Decode the whole waveform, transcribe it, diarize it, attribute each word to a
speaker. The output is `transcript.raw.json`, which carries words, segments and
turns and **no chunk ids at all**.

**The whole file at once is not an optimisation.** Whisper carries context
across an utterance and detects language from the opening seconds, so a
twenty-second window loses the sentence that straddles its edge. Diarization is
worse: speaker labels come from clustering embeddings over the *entire*
recording, so `SPEAKER_00` in one window bears no relation to `SPEAKER_00` in
the next -- chunk first and the speakers are not misaligned, they are
unnameable.

**This module does not know that chunking exists.** Boundaries are applied
afterwards by `cut`, and the cuts a `vad` or `speaker` policy derives are
`boundaries/speech.py`'s job. Both read the file this writes; neither is
imported here. That asymmetry -- audio conforms to a grid decided anywhere,
because Whisper timestamps every word, while a sampler's decisions are made
during a decode pass and cannot be revisited -- is what lets either modality
own the grid.

**Transcription and diarization stay separate passes, joined by `align`.**
Whisper does not know who spoke and pyannote does not know what was said;
keeping them apart is what lets either be swapped.
"""

from __future__ import annotations

from .driver import load, main, run
from .reader import listen

__all__ = ["listen", "load", "main", "run"]
