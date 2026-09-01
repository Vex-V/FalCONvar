"""The manifest, written as chunks close rather than at the end.

A downstream describer should be able to start on chunk 0 while chunk 12 is
still being read, and a run that dies at minute 40 of an hour should leave the
first 40 minutes usable. Both need the file to be current, not final.

Each rewrite goes to a temporary file and is moved into place with
``os.replace``, which is atomic on POSIX and Windows alike. A reader therefore
sees either the previous complete manifest or the new one, never a torn file.

``complete`` says whether ingestion finished. A consumer polling the file uses
it to tell "no more chunks yet" from "no more chunks ever".
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

MANIFEST_VERSION = 1


class ManifestWriter:
    """Accumulates chunks and keeps a valid manifest on disk throughout."""

    def __init__(
        self,
        path: str | Path,
        video_id: str,
        source: dict[str, Any],
        config: dict[str, Any],
        flush_every: int = 1,
    ) -> None:
        self.path = Path(path)
        self.video_id = video_id
        self.source = source
        self.config = config
        # Rewriting costs O(chunks^2) writes over a run. At a few KB per chunk
        # that is nothing for a 15-chunk file; a multi-hour ingest with
        # hundreds of chunks can raise this to trade freshness for writes.
        self.flush_every = max(1, flush_every)
        self.chunks: list[dict[str, Any]] = []
        self.stats: dict[str, Any] = {}
        self.complete = False
        self._since_flush = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()

    def document(self) -> dict[str, Any]:
        return {
            "manifest_version": MANIFEST_VERSION,
            "video_id": self.video_id,
            "complete": self.complete,
            "source": self.source,
            "config": self.config,
            "stats": self.stats,
            "chunks": self.chunks,
        }

    def chunk_closed(self, chunk: dict[str, Any], stats: Optional[dict] = None) -> None:
        """Record a finished chunk; flush if enough have accumulated."""
        self.chunks.append(chunk)
        if stats is not None:
            self.stats = stats
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self._write()
            self._since_flush = 0

    def finish(self, stats: Optional[dict] = None) -> dict[str, Any]:
        if stats is not None:
            self.stats = stats
        self.complete = True
        self._write()
        return self.document()

    def _write(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.document(), indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
