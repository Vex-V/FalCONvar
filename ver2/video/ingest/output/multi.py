"""One manifest, several destinations.

Writing to a file *and* to Postgres is not redundancy for its own sake. The
sink holds the only copy of work that is expensive to redo -- minutes of GPU
inference on a long video -- and the two destinations fail in unrelated ways:
a disk is not reachable over a network, and a network outage does not corrupt
a local file.

So the first sink is the primary and its failures are real failures; the rest
are best-effort. A remote sink that raises is reported once and then dropped
for the remainder of the run, rather than raising 137 times or taking an
otherwise good ingest down with it.

The consequence is deliberate and must be understood by anything reading the
remote copy: a run can end with a complete manifest on disk and a partial one
in Postgres. That is exactly what the ``complete`` flag is for -- it stays
false, so a consumer polling the rows can tell "no more chunks yet" from
"no more chunks ever", and never mistakes a dropped sink for a finished run.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ver2.fanout import FanOut

from .base import ManifestSink


class MultiSink(FanOut):
    """Fans a manifest out to several sinks. A ``ManifestSink`` itself."""

    def begin(
        self, video_id: str, source: dict[str, Any], config: dict[str, Any]
    ) -> None:
        self.primary.begin(video_id, source, config)
        for sink in list(self.secondary):
            self._try(sink, lambda s: s.begin(video_id, source, config), "begin")

    def chunk_closed(self, chunk: dict[str, Any], stats: Optional[dict] = None) -> None:
        self.primary.chunk_closed(chunk, stats)
        for sink in list(self.secondary):
            self._try(sink, lambda s: s.chunk_closed(chunk, stats),
                      f"chunk {chunk.get('chunk_id')}")

    def finish(
        self,
        stats: Optional[dict] = None,
        chunks: Optional[Sequence[dict[str, Any]]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        for sink in list(self.secondary):
            self._try(sink, lambda s: s.finish(stats, chunks, config), "finish")
        # Last, and unguarded: the document the run returns is the primary's.
        return self.primary.finish(stats, chunks, config)
