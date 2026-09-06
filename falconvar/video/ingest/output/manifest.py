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
from typing import Any, Optional, Sequence

MANIFEST_VERSION = 1


class FileManifestWriter:
    """Accumulates chunks and keeps a valid manifest on disk throughout.

    Implements ``ManifestSink``. The name says *file* because the atomic
    rewrite below is a file-shaped answer to a file-shaped problem: JSON
    cannot be appended to while someone reads it. A sink writing rows has no
    such constraint and should not inherit this one.
    """

    def __init__(self, path: str | Path) -> None:
        # Construction takes only what is specific to writing a file. Every
        # fact about the run arrives through begin(), the same way it does for
        # every other sink.
        self.path = Path(path)
        self.video_id = ""
        self.source: dict[str, Any] = {}
        self.config: dict[str, Any] = {}
        self.chunks: list[dict[str, Any]] = []
        self.stats: dict[str, Any] = {}
        self.complete = False

    def begin(
        self, video_id: str, source: dict[str, Any], config: dict[str, Any]
    ) -> None:
        """Write the empty manifest, so the file is valid from the first chunk."""
        self.video_id = video_id
        self.source = source
        self.config = config
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
        """Record a finished chunk and rewrite the file.

        Every chunk, because the file is the live view: a reader polling it
        should see chunk 12 while chunk 13 is still being sampled. The cost is
        quadratic in chunks -- chunk 100's write emits all 100 -- which is
        17 MB of writes for a 137-chunk video whose manifest is 253 KB. That
        is affordable, and batching it would trade the freshness this whole
        design is for.
        """
        self.chunks.append(chunk)
        if stats is not None:
            self.stats = stats
        self._write()

    def finish(
        self,
        stats: Optional[dict] = None,
        chunks: Optional[Sequence[dict[str, Any]]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Write the final document.

        ``chunks`` and ``config`` re-state what the pipeline only knows at the
        end -- the last chunk's true ``end_ts`` and the chunker's closing
        counters -- so no caller needs to reach in and assign attributes.
        """
        if stats is not None:
            self.stats = stats
        if chunks is not None:
            self.chunks = list(chunks)
        if config is not None:
            self.config = config
        self.complete = True
        self._write()
        return self.document()

    def _write(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.document(), indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
