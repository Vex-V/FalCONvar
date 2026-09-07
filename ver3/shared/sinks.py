"""Writing a document where it was asked to go.

`falconvar` has four of these -- `video/ingest/output/`, `video/describe/output/`,
`audio/output/`, `aggregate/output/` -- each with its own `base`, `document`,
`multi` and `supabase`. Nineteen files implementing one idea four times, because
each stage grew its own sink package before anyone noticed the shape repeated.

There is one idea: a component produces one document, and that document goes to
a file, to Postgres, or to both. Nothing about that differs by component, so
nothing about it needs a per-component implementation.

**The file write is atomic.** Temp file plus `os.replace`, so a reader polling
a directory never sees a half-written document. That matters more here than in
`falconvar`, because components hand off *through* files: a partially written
`timeline.json` is not a corrupt log line, it is the next component's input.

**File is primary, Postgres is best-effort.** When both are asked for, a
Postgres failure is reported and the run continues, because the local document
is the one the next component reads. The reverse -- treating the database as
authoritative -- would make an unreachable network a reason not to have
produced a manifest that was already computed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from . import paths

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

    ``to_supabase`` is passed in rather than imported, so this module knows
    nothing about the database: a component that has rows to write supplies the
    function that writes them, and one that does not simply never asks for that
    backend. That keeps `sinks` free of `db`, and `db` out of the import path
    of every file-only run.
    """
    written: dict[str, str] = {}
    for backend in parse(sink):
        if backend == "file":
            written["file"] = str(write_json(paths.artifact(video_id, artifact),
                                             document))
        elif backend == "supabase":
            if to_supabase is None:
                # Asked for, but this component has no row mapping. Saying so
                # beats silently writing nothing and reporting success.
                raise UnknownBackend(
                    f"{artifact!r} has no Postgres representation to write")
            try:
                to_supabase(video_id, document)
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
