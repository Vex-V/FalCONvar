"""Getting hold of a manifest, from a file or from Postgres.

Three ways in, and they differ only in how much of the manifest exists yet:

  a path            the whole document, already on disk
  a video id        the whole document, reassembled by ``export_manifest``
  a video id, live  the header alone -- source and config, no chunks, because
                    ingest is still producing them

The third is what makes following possible. A run claims its ``video_id``
before it decodes a frame, so a follower can learn everything about the run
except its results and then wait for those.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ver2 import db


def from_file(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


#: Reading a manifest out of Postgres is not describe's business -- ingest
#: writes it and retrieve reads it too -- so it lives in ver2/db.py and these
#: are the names this package uses it by.
from_supabase = db.fetch_manifest
header = db.manifest_header
