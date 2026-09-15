"""Writing a document where it was asked to go.

One fan-out for every component: a document goes to a file, to Postgres, or to
both. Nothing about that differs by component.

The file write is atomic -- temp file plus `os.replace` -- because components
hand off through files, so a partially written document is the next
component's input rather than a corrupt log line.

File is primary and Postgres best-effort: when both are asked for, a Postgres
failure is reported and the run continues, because the local document is what
the next component reads.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .. import paths

BACKENDS = ("file", "supabase")


class UnknownBackend(ValueError):
    """A sink name that is not `file` or `supabase`."""


def parse(sink: str | Sequence[str]) -> list[str]:
    """`"file,supabase"` or `["file"]` -> a validated list, file first.

    Order is normalised rather than honoured: `file` is always primary, so
    "supabase,file" and "file,supabase" mean the same thing and neither can
    accidentally make the network the thing a run depends on.
    """
    names = ([s.strip() for s in sink.split(",")] if isinstance(sink, str)
             else [str(s).strip() for s in sink])
    names = [n for n in names if n]
    if not names:
        raise UnknownBackend(f"name at least one of: {', '.join(BACKENDS)}")
    unknown = [n for n in names if n not in BACKENDS]
    if unknown:
        raise UnknownBackend(
            f"unknown sink(s) {', '.join(unknown)}; known: {', '.join(BACKENDS)}")
    return [n for n in BACKENDS if n in names]


def write_json(path: Path, document: dict[str, Any]) -> Path:
    """Atomically. A reader never sees a torn document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(video_id: str, artifact: str, document: dict[str, Any],
          sink: str | Sequence[str] = "file",
          to_supabase: Optional[Callable[[str, dict[str, Any]], None]] = None,
          on_problem: Optional[Callable[[str], None]] = None) -> dict[str, str]:
    """Put one document where the caller asked. Returns {backend: location}.

    ``to_supabase`` may be passed in, but it is normally looked up by artifact
    name from `rows.WRITERS` -- imported *inside* the branch, so a file-only
    run never loads the database client and never needs a key. A document with
    no row mapping cannot be written to Postgres, and saying so beats writing
    nothing and reporting success.
    """
    written: dict[str, str] = {}
    for backend in parse(sink):
        if backend == "file":
            written["file"] = str(write_json(paths.artifact(video_id, artifact),
                                             document))
        elif backend == "supabase":
            handler = to_supabase
            if handler is None:
                from . import rows
                handler = rows.writer_for(artifact)
            if handler is None:
                raise UnknownBackend(
                    f"{artifact!r} has no Postgres representation to write")
            try:
                handler(video_id, document)
                written["supabase"] = f"{artifact}@supabase"
            except Exception as exc:                        # noqa: BLE001
                # Best-effort by design: the file is what the next component
                # reads, so a database that is down is a report, not a failure.
                message = f"supabase write failed for {artifact}: {exc}"
                if on_problem is not None:
                    on_problem(message)
                else:
                    raise
    return written


__all__ = ["BACKENDS", "UnknownBackend", "parse", "write", "write_json",
           "read_json"]
