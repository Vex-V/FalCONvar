"""The transcript document, and the shape of what audio produces.

Written whole, via a temporary file and `os.replace`, so a reader sees a
complete document or none -- the same discipline the manifest and the
description document use, for the same reason.

Two views of one pass, and the distinction is load-bearing:

  `segments`  the record. Every word with its own timestamp, the speaker who
              said it, and the model's confidence. Nothing here depends on a
              chunk grid, which is why re-cutting to a different one costs
              nothing and loses nothing.

  `chunks`    the current grid's view, keyed by the same `chunk_id` the
              manifest uses. Derived, and regenerable from `segments` plus a
              timeline, so it can be thrown away.

`timeline_fingerprint` is recorded because that correspondence is the whole
claim: if it does not match the manifest's, the two documents were cut on
different grids and `chunk_id` means two different things in them. Making that
a comparison a reader can perform is the point -- the alternative is two
documents that look joinable and are not.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

TRANSCRIPT_VERSION = 1


def build_document(
    video_id: str,
    uri: str,
    transcript: Any,
    diarization: Any,
    chunks: list[dict[str, Any]],
    timeline: Any,
    audio: dict[str, Any],
    stats: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble the document. Pure: it reads its arguments and nothing else."""
    return {
        "transcript_version": TRANSCRIPT_VERSION,
        "video_id": video_id,
        "complete": True,
        "timeline_fingerprint": timeline.fingerprint(),
        "source": {"uri": uri, "video_id": video_id, "audio": audio},
        "model": {
            "transcriber": transcript.model,
            "diarizer": diarization.model,
        },
        "language": transcript.language,
        "language_probability": round(transcript.language_probability, 4),
        "speakers": diarization.speakers,
        "stats": stats or {},
        "timeline": timeline.as_dict(),
        "chunks": chunks,
        "segments": [s.as_dict() for s in transcript.segments],
    }


class TranscriptDocument:
    """Writes the document to one path. A ``TranscriptSink``."""

    name = "file"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, document: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(document, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
        return document
