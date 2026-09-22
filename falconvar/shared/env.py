"""Reading `.env`, once, before a key is needed.

Every driver's `run()` calls `load()` before it needs a key, because each
component is callable on its own -- there is no single process start to hang
this off. It is idempotent and cheap, so calling it four times in one run costs
nothing.

**Where the file is depends on who is running.** From a checkout it is the
checkout's `.env`, which is what a developer means. Installed, there is no
checkout, so it is `.env` in the working directory -- the convention every
other library follows, and the only file a consumer would expect to be read.
`load(path)` names one explicitly when neither is right.

**Never overrides the process.** A shell variable is a deliberate act and a
file is a default, so `override=False` always.

**Not for paths.** `shared/paths.py` resolves its roots on first use, which can
be earlier or later than this runs -- so a data root set in `.env` would take
effect or not depending on which module touched a path first. The path knobs
are process variables and `paths.configure()`; this file handles keys only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from . import paths

_loaded = False


def env_file() -> Optional[Path]:
    """The `.env` this process would read, or None if there is none."""
    candidate = (paths.CHECKOUT / ".env" if paths.CHECKOUT is not None
                 else Path.cwd() / ".env")
    return candidate if candidate.exists() else None


def load(path: Optional[Path | str] = None, force: bool = False) -> bool:
    """Read `.env`. True if anything was read.

    `path` names a file explicitly; without it, see `env_file`.
    """
    global _loaded
    if _loaded and not force and path is None:
        return True

    found = Path(path).expanduser() if path is not None else env_file()
    if found is None or not found.exists():
        _loaded = True
        return False

    load_dotenv(found, override=False)
    _loaded = True
    return True


def first(*names: str) -> Optional[str]:
    """The first of these environment variables that is set."""
    load()
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


__all__ = ["env_file", "first", "load"]
