"""Where a transcript goes.

**One call, not three.** The manifest and description sinks hear
`begin`/`chunk_closed`/`finish` because those stages produce results a chunk at
a time and a reader has to be able to see a partial run. Audio does not work
that way: transcription and diarization are whole-file passes that produce
nothing until they produce everything, so a sink with a streaming shape would
be three methods where two of them are always called back to back with nothing
in between. Pretending otherwise would suggest a progress signal that does not
exist.

What is written is two views of one pass. `segments` is the record -- every
word with its own timestamp and speaker -- and `chunks` is the current grid's
view of it, which is derived and can be rebuilt for any other grid without
re-running a model. Keeping both is what makes the chunk boundaries a decision
that can be revisited rather than baked in at inference time.
"""

from __future__ import annotations

from typing import Any, Protocol


class TranscriptSink(Protocol):
    def write(self, document: dict[str, Any]) -> dict[str, Any]:
        """Store one finished transcript document, and return what was stored."""
        ...
