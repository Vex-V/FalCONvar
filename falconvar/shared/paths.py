"""Where everything lives.

Imports only `errors`, which imports nothing -- so this stays a leaf and no
cycle is possible. The only module that knows an artifact's filename: a caller
names a video and an artifact and never concatenates a path.

**A checkout is recognised, not guessed.** Searching upward for
`pyproject.toml` finds *a* marker, not necessarily this project's: installed
into a venv inside someone else's repo, the search walked out of
`site-packages` and returned *their* root, so the library imported fine and
wrote every artifact and 338 MB of checkpoints into their tree with nothing
reporting it. A marker counts only when the package under it is this one --
`root/falconvar/shared/paths.py` has to be this very file. That is also why a
parent count is wrong: it is a fact about a file's depth, which is what a
reorganisation changes, and this module has been moved once already.

**The roots resolve on first use, not at import.** They used to be module
constants, so an installed copy raised `RuntimeError` from `import falconvar`
before a caller could say where its data should go. Now `configure()` can be
called after the import that triggers it, and a library with no checkout
around it has somewhere to write.

Precedence, highest first: `configure()`, then `FALCONVAR_DATA` /
`FALCONVAR_WEIGHTS`, then the checkout when there is one, then
`~/.falconvar`. `configure()` wins because an embedding application must be
able to guarantee where it writes; the variables are for when you do not
control the calling code.

**Not for keys.** `shared/env.py` reads `.env`, which happens later than this
resolves -- so a data root set in `.env` would take effect or not depending on
which module was imported first.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional
from .errors import FalconvarError

#: Where an installed copy writes when nothing says otherwise.
FALLBACK_HOME = Path.home() / ".falconvar"


def _checkout_root() -> Optional[Path]:
    """The checkout this file belongs to, or None if it is installed.

    A `pyproject.toml` above us is only ours if the package beneath it is this
    package -- otherwise it belongs to whoever we were installed into.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if not (parent / "pyproject.toml").exists():
            continue
        candidate = parent / "falconvar" / "shared" / "paths.py"
        if candidate.exists() and candidate.resolve() == here:
            return parent
    return None


#: The checkout, or None when installed. Resolved once: it cannot change.
CHECKOUT = _checkout_root()

#: What `configure()` set, if anything.
_configured: dict[str, Path] = {}


def configure(data_root: Optional[Path | str] = None,
              weights: Optional[Path | str] = None) -> None:
    """Say where this process reads and writes. Takes precedence over both the
    environment and the checkout.

    Call it before anything runs. Paths already handed out are not revisited --
    they are computed per call, so only work started earlier keeps the old root.
    """
    for key, value in (("data", data_root), ("weights", weights)):
        if value is not None:
            _configured[key] = Path(value).expanduser().resolve()


def _resolve(key: str, variable: str, in_checkout: str) -> Path:
    if key in _configured:
        return _configured[key]
    value = os.environ.get(variable)
    if value:
        return Path(value).expanduser().resolve()
    if CHECKOUT is not None:
        return CHECKOUT / in_checkout
    return FALLBACK_HOME / in_checkout


def data_root() -> Path:
    return _resolve("data", "FALCONVAR_DATA", "data")


def weights_root() -> Path:
    return _resolve("weights", "FALCONVAR_WEIGHTS", "weights")


def out_root() -> Path:
    return data_root() / "out"


def checkout_root() -> Path:
    """The checkout, for the tools that write *source* rather than output.

    Raises when installed: `schemas --check` regenerates files under `db/`,
    which only exists in a checkout, and a silent wrong directory is what this
    module exists to prevent.
    """
    if CHECKOUT is None:
        raise RuntimeError(
            "not running from a FalCONvar checkout -- this is a development "
            "tool and needs the repository, not an installed copy")
    return CHECKOUT


#: Lazily-resolved module attributes, each `paths.NAME` a call to its function.
#: Kept as names so every reader stays `paths.OUT_ROOT`, while the value is
#: computed now rather than at import.
_LAZY: dict[str, Any] = {
    "REPO_ROOT": checkout_root,
    "DATA_ROOT": data_root,
    "OUT_ROOT": out_root,
    "WEIGHTS": weights_root,
    "UPLOADS": lambda: data_root() / "uploads",
    #: Custom describe prompts, added through the API. Under the data root
    #: rather than in the package because a request writes it; the built-ins
    #: stay in `falconvar/video_rag/describe/prompts.json`, read-only.
    "PROMPTS": lambda: data_root() / "prompts.json",
    #: Custom aggregate prompts and link profiles. The built-ins live in
    #: `falconvar/aggregates/definitions/definitions.json`.
    "AGGREGATE_DEFINITIONS": lambda: data_root() / "aggregates.json",
    #: Model endpoints added or overridden per deployment. Hand-edited, and
    #: never holds a key -- it names the variables keys are read from.
    "PROVIDERS": lambda: data_root() / "providers.json",
}


def __getattr__(name: str) -> Any:
    """PEP 562: `paths.OUT_ROOT` resolves on access, not at import."""
    if name in _LAZY:
        return _LAZY[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


#: artifact name -> filename under `data/out/<video-id>/`.
#: The whole naming convention, in one place. A component asks for "timeline";
#: nothing anywhere else spells "timeline.json".
ARTIFACTS: dict[str, str] = {
    "media": "media.json",
    "raw_transcript": "transcript.raw.json",
    "cuts": "cuts.json",
    "timeline": "timeline.json",
    "manifest": "manifest.json",
    "transcript": "transcript.json",
    "descriptions": "descriptions.json",
    "embedded": "embedded.json",
}

#: Artifacts that are directories rather than documents.
DIRECTORIES: dict[str, str] = {
    "store": "store",
    "aggregates": "aggregates",
}


class UnknownArtifact(FalconvarError, KeyError):
    """An artifact name nothing in ARTIFACTS or DIRECTORIES answers to."""


def home(video_id: str) -> Path:
    """Everything one video produced, in one directory.

    Grouped by video rather than by artifact type, so a video's whole output is
    one thing to inspect, copy or delete, and a later component adds to it
    without a new top-level directory.
    """
    return out_root() / video_id


def artifact(video_id: str, name: str) -> Path:
    """The path an artifact would occupy. It need not exist yet."""
    if name in ARTIFACTS:
        return home(video_id) / ARTIFACTS[name]
    if name in DIRECTORIES:
        return home(video_id) / DIRECTORIES[name]
    known = ", ".join(sorted({*ARTIFACTS, *DIRECTORIES}))
    raise UnknownArtifact(f"unknown artifact {name!r}; known: {known}")


def exists(video_id: str, name: str) -> bool:
    return artifact(video_id, name).exists()


def present(video_id: str) -> list[str]:
    """Which artifacts this video actually has, in pipeline order.

    Read from disk rather than remembered, so a restarted process still knows
    everything it produced -- and so a listing advertises no artifact that
    would 404, which reads as breakage rather than as a stage never run.
    """
    order = [*ARTIFACTS, *DIRECTORIES]
    return [name for name in order if exists(video_id, name)]


#: A leading underscore marks a directory under OUT_ROOT that is not a video.
#: The embedded Qdrant store lives at `_qdrant`, beside the videos rather than
#: inside one, and without this it was listed as a video with no artifacts --
#: by `paths.videos()`, and so by `GET /videos`.
RESERVED_PREFIX = "_"


def videos() -> list[str]:
    """Every video id with an output directory."""
    root = out_root()
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and not d.name.startswith(RESERVED_PREFIX))


__all__ = ["ARTIFACTS", "CHECKOUT", "DIRECTORIES", "FALLBACK_HOME",
           "RESERVED_PREFIX", "UnknownArtifact",
           "artifact", "checkout_root", "configure", "data_root", "exists",
           "home", "out_root", "present", "videos", "weights_root"]
