"""Where everything lives.

Imports nothing, like `documents.py`. Every component needs these and none of
them may own them.

This is the *only* module that knows an artifact's filename. A caller names a
video and an artifact; it never concatenates a path. That is what lets a
component read its input from a file or from Postgres without the caller
knowing which, and it is why `falconvar`'s habit of writing `Path("out")` in
eight places -- each relative to the working directory -- is not repeated here.

``FALCONVAR_DATA`` moves the data root. It is a *process* variable, set before
launch, not a `.env` key: `.env` is read at the top of an entry point, which is
later than these constants resolve, and a path that moved depending on how
early it was read would be worse than one that cannot go in `.env` at all.
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
    `parents[1]` was correct at `ver3/paths.py` and silently wrong the moment
    it became `ver3/shared/paths.py`, redirecting every artifact into
    `ver3/data/`.

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

#: `data/ver3/` while both trees exist, not `data/out/`.
#:
#: The two pipelines write documents of the same name -- `timeline.json`,
#: `manifest.json` -- with different schemas, into a directory keyed only by
#: video id. Sharing it means one tree silently overwrites measurements taken
#: from the other, and the corrupted artifact still parses, which is the worst
#: version of that failure. One line, deleted when `falconvar` goes.
OUT_ROOT = DATA_ROOT / "ver3"
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


def videos() -> list[str]:
    """Every video id with an output directory."""
    if not OUT_ROOT.exists():
        return []
    return sorted(d.name for d in OUT_ROOT.iterdir() if d.is_dir())


__all__ = ["REPO_ROOT", "DATA_ROOT", "OUT_ROOT", "UPLOADS", "WEIGHTS",
           "ARTIFACTS", "DIRECTORIES", "UnknownArtifact",
           "home", "artifact", "exists", "present", "videos"]
