"""Where everything lives.

Imports nothing. The only module that knows an artifact's filename: a caller
names a video and an artifact and never concatenates a path.

The checkout is found by searching upward for `pyproject.toml`, not by counting
parents -- a parent count is a fact about a file's depth in the tree, which is
exactly what a reorganisation changes.

`FALCONVAR_DATA` moves the data root. It is a process variable, not a `.env`
key: `.env` is read at the top of an entry point, which is later than these
constants resolve.
"""

from __future__ import annotations

import os
from pathlib import Path

def _repo_root() -> Path:
    """Find the checkout by a marker, not by counting parents.

    `falconvar` computed its weights directory as `parents[3] / "weights"`,
    which resolved to a path that had never existed -- so every lookup fell
    through to a bare filename and nothing reported it. A parent count is a
    fact about a file's depth in the tree, which is exactly the fact a
    reorganisation changes, and this module has now been moved once already:
    `parents[1]` was correct at `falconvar/paths.py` and silently wrong the moment
    it became `falconvar/shared/paths.py`, redirecting every artifact into
    `falconvar/data/`.

    `pyproject.toml` marks the root. Falling back to the old count if no marker
    is found would reintroduce exactly the silent-wrong-answer this replaces,
    so an unmarked tree raises instead.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(
        f"no pyproject.toml above {here} -- cannot locate the checkout root")


REPO_ROOT = _repo_root()


def _from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


DATA_ROOT = _from_env("FALCONVAR_DATA", REPO_ROOT / "data")

OUT_ROOT = DATA_ROOT / "out"
UPLOADS = DATA_ROOT / "uploads"
WEIGHTS = _from_env("FALCONVAR_WEIGHTS", REPO_ROOT / "weights")

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


class UnknownArtifact(KeyError):
    """An artifact name nothing in ARTIFACTS or DIRECTORIES answers to."""


def home(video_id: str) -> Path:
    """Everything one video produced, in one directory.

    Grouped by video rather than by artifact type, so a video's whole output is
    one thing to inspect, copy or delete, and a later component adds to it
    without a new top-level directory.
    """
    return OUT_ROOT / video_id


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
    if not OUT_ROOT.exists():
        return []
    return sorted(d.name for d in OUT_ROOT.iterdir()
                  if d.is_dir() and not d.name.startswith(RESERVED_PREFIX))


__all__ = ["REPO_ROOT", "DATA_ROOT", "OUT_ROOT", "RESERVED_PREFIX",
           "UPLOADS", "WEIGHTS",
           "ARTIFACTS", "DIRECTORIES", "UnknownArtifact",
           "home", "artifact", "exists", "present", "videos"]
