"""Aggregates on disk: one file each, under the video's own directory.

    out/<video-id>/aggregates/<aggregate_id>.json

**One file per aggregator, not one document for all of them**, because their
costs differ by orders of magnitude. `stats` is arithmetic; `summary` is a paid
hierarchical reduce. A single document means re-running the free ones rewrites
the expensive ones' results, and a failure while writing risks both. Listing
what a video has is reading the directory, which is how `out/` is already read.

Each file is written to a temporary path and `os.replace`d into position, so a
reader sees a whole document or the previous one, never a torn file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

AGGREGATE_VERSION = 1


class AggregateDocuments:
    """Writes one file per aggregate. An ``AggregateSink``."""

    name = "file"

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)

    def write(self, record: dict[str, Any]) -> dict[str, Any]:
        self.dir.mkdir(parents=True, exist_ok=True)
        document = {"aggregate_version": AGGREGATE_VERSION, **record}
        path = self.dir / f"{record['aggregate_id']}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(document, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return document

    def read(self, aggregate_id: str) -> dict[str, Any] | None:
        path = self.dir / f"{aggregate_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def existing(self) -> dict[str, str]:
        """aggregate_id -> the fingerprint of the inputs it was built from.

        What makes a re-run able to say "3 stale, 5 current" instead of either
        redoing everything or trusting everything.
        """
        if not self.dir.is_dir():
            return {}
        out = {}
        for path in sorted(self.dir.glob("*.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            out[path.stem] = document.get("inputs_fingerprint", "")
        return out
