"""One transcript, several destinations.

The same policy every stage here uses: the first sink is authoritative and its
failures are real, the rest are best-effort and a failing one is reported once
and dropped. See `ver2.fanout.FanOut`.

There is no `existing()` to intersect, unlike the description sinks. Audio has
nothing to resume: a transcription either ran over the whole file or it did
not, so a partial copy is not a state this stage can be in.
"""

from __future__ import annotations

from typing import Any

from ver2.fanout import FanOut


class MultiTranscriptSink(FanOut):
    """Fans a transcript out to several sinks. A ``TranscriptSink`` itself."""

    def write(self, document: dict[str, Any]) -> dict[str, Any]:
        written = self.primary.write(document)
        for sink in list(self.secondary):
            self._try(sink, lambda s: s.write(document), "write")
        return written
