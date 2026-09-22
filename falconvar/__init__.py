"""FalCONvar -- video in, searchable moments and higher-level answers out.

Two tiers, each with one driver over the components in its folder:

    video_rag/   extraction and search. The picture and the soundtrack read onto
                 one chunk grid, described, embedded, and queryable -- a
                 complete RAG engine on its own.
    aggregates/  answers over what video_rag extracted: counts, speakers,
                 summaries, chapters, events, entities. Never reads the video.
    shared/      what both tiers need: paths, env, document contracts, storage,
                 model providers.

`workflow.py` runs video_rag's driver, then aggregates'. Components exchange
files rather than objects, and every one is `run(video_id, ...) -> Produced`.
"""

from __future__ import annotations

from .shared.errors import FalconvarError, ModelUnavailable, Unavailable
from .shared.paths import configure

#: Read from the installed metadata rather than restated, so it cannot drift
#: from `pyproject.toml` -- but resolved on first access, not at import.
#: `importlib.metadata.version` scans the environment's distributions, which
#: measured **~120 ms** on this machine: a tenth of a second added to every
#: `import falconvar` to compute a string almost nobody reads.
def __getattr__(name: str) -> str:
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version
        try:
            value = version("falconvar")
        except PackageNotFoundError:          # a checkout that was never installed
            value = "0+unknown"
        globals()["__version__"] = value      # resolve once
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FalconvarError", "ModelUnavailable", "Unavailable",
           "__version__", "configure"]

