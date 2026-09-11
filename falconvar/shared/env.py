"""Reading `.env`, once, at the top of an entry point.

Every driver's `run()` calls `load()` before it needs a key, because each
component is callable on its own -- there is no single process start to hang
this off. It is idempotent and cheap, so calling it four times in one run costs
nothing.

**Not for paths.** `shared/paths.py` resolves `FALCONVAR_DATA` at import, which
is earlier than this can possibly run, so a data root set in `.env` would take
effect or not depending on which module was imported first. That is a worse
failure than a variable that simply cannot live here, which is why the path
knobs are documented as process variables and this file handles keys only.
"""

from __future__ import annotations

import os
from typing import Optional

from .paths import REPO_ROOT

_loaded = False


def load(force: bool = False) -> bool:
    """Load `.env` from the checkout root. True if anything was read.

    `python-dotenv` if it is installed, a minimal parser otherwise: a missing
    optional dependency should not be the reason a key is not found.
    """
    global _loaded
    if _loaded and not force:
        return True

    path = REPO_ROOT / ".env"
    if not path.exists():
        _loaded = True
        return False

    try:
        from dotenv import load_dotenv
        load_dotenv(path, override=False)
    except ImportError:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            # Never override what the process was actually launched with: a
            # shell variable is a deliberate act and a file is a default.
            os.environ.setdefault(key, value)
    _loaded = True
    return True


def first(*names: str) -> Optional[str]:
    """The first of these environment variables that is set."""
    load()
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


__all__ = ["load", "first"]
