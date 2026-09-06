"""Where things live on disk.

Imports nothing, like `timeline.py` and for the same reason: every stage needs
these and none of them may own them. It is also the answer to a bug this file
was written to fix -- `components/detectors.py` computed its weights directory
as ``Path(__file__).parents[3] / "weights"``, which resolves to
``falconvar/video/weights``. That directory has never existed, so `weight_path` fell
through to the bare filename on every call and ultralytics downloaded wherever
its own settings pointed. Nothing failed; the "keep them in one place" comment
above it simply was not in effect. A relative-parent count is a fact about a
file's depth in the tree, which is exactly the fact a reorganisation changes.

Two roots:

``REPO_ROOT``   the checkout. Everything below is anchored to it rather than to
                the working directory, so `python -m falconvar.…` behaves the same
                from anywhere -- which it did not before, and which is why the
                drivers carry a `sys.path` insert to be runnable as scripts.

``DATA_ROOT``   everything a run writes: ``data/out`` and ``data/uploads``.
                Overridable with the ``FALCONVAR_DATA`` environment variable --
                a *process* variable, set before launch, not a `.env` key.
                `.env` is read by `db.load_env()` at the top of each entry
                point, which is later than these constants resolve; a path that
                changed depending on how early it was read would be worse than
                one that cannot be put in `.env` at all. ``FALCONVAR_DATA=.``
                restores the old layout, with `out/` and `uploads/` at the root.

Model weights stay outside ``DATA_ROOT``: they are a cache shared by every run
and not this project's output, so deleting the data directory should not cost a
338 MB re-download.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The checkout: this file is `<root>/falconvar/paths.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


#: Everything a run produces. One directory to back up, copy or delete.
DATA_ROOT = _from_env("FALCONVAR_DATA", REPO_ROOT / "data")

#: `out/<video-id>/` -- manifest, store, descriptions, transcript, aggregates.
OUT_ROOT = DATA_ROOT / "out"

#: Where the API parks an upload until the run reads it.
UPLOADS = DATA_ROOT / "uploads"

#: Detector and embedder checkpoints. A cache, not output.
WEIGHTS = _from_env("FALCONVAR_WEIGHTS", REPO_ROOT / "weights")

__all__ = ["REPO_ROOT", "DATA_ROOT", "OUT_ROOT", "UPLOADS", "WEIGHTS"]
