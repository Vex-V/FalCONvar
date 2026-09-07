"""6 · cut -- the raw transcript, onto the grid.

The cheap half of the audio side, and the step the whole ordering argument
rests on. Transcription and diarization ran over the whole file and know
nothing about chunks; the grid arrives afterwards and may have come from the
picture, the soundtrack, or arithmetic. Cutting here is exact and costs
nothing because Whisper timestamps every word -- so a transcript can be re-cut
any number of times, to any grid, with no word lost and no model reloaded.

**A word belongs to the chunk containing its midpoint.** The same rule that
attributes a word to a speaker, for the same reason: a word straddling a
boundary belongs to whichever side holds more of it, and every word must land
in exactly one chunk or the text is duplicated or dropped.

**Speaker turns survive a coarse grid rather than being flattened by it.** A
chunk carries `turns` -- one bound record per contiguous run of one voice, with
its own span and its own words -- so a twenty-second window holding three
speakers still says who said what, in order. Parallel lists of speakers and
sentences cannot say which went with which, and cannot be made to afterwards.
"""

from __future__ import annotations

from .driver import load, main, run
from .cutter import to_chunks

__all__ = ["load", "main", "run", "to_chunks"]
