"""Which embedder and which index, decided in one place.

`embed` writes vectors and `retrieve` reads them, and the two have to name the
**same embedder** or the search is well-formed and meaningless -- a mismatch
between two models of the same width returns a ranking with nothing wrong on
the face of it. Leaving that agreement to whoever types the second command is
the one configuration mistake here that does not announce itself, so the
defaults live in a module both drivers read rather than in two argparse calls
that happen to match today.

Every value is overridable from the environment, which is what makes the
eventual switch to a local stack a line in `.env` rather than a flag on every
command:

    FALCONVAR_EMBEDDER=local
    FALCONVAR_EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5
    FALCONVAR_INDEX=qdrant

An explicit flag still wins over the environment, and the environment over
these constants. Nothing is read at import time: `db.load_env()` runs first in
both drivers, so a `.env` is in effect before a default is resolved.

Postgres is the default index because it is the only one carrying the lexical
half. Qdrant holds dense vectors alone, so a search against it is missing a
component that measured 0.429 top-1 against 0.714 on the same corpus -- a real
option, and the one to develop next, but not the one to get by saying nothing.
"""

from __future__ import annotations

import os
from typing import Optional

#: What is used when neither a flag nor the environment says otherwise.
INDEX = "pgvector"
EMBEDDER = "openai"
MODEL: Optional[str] = None          # None means the embedder's own default

ENV_INDEX = "FALCONVAR_INDEX"
ENV_EMBEDDER = "FALCONVAR_EMBEDDER"
ENV_MODEL = "FALCONVAR_EMBED_MODEL"


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    return value.strip() or None if value else None


def index() -> str:
    """Comma-separated index backends; the first named is the primary."""
    return _env(ENV_INDEX) or INDEX


def embedder() -> str:
    return _env(ENV_EMBEDDER) or EMBEDDER


def model() -> Optional[str]:
    """The model id, or None to let the embedder choose its own."""
    return _env(ENV_MODEL) or MODEL


def describe() -> str:
    """A one-line summary of where the defaults came from, for `--help`."""
    parts = []
    for label, name, value in (("index", ENV_INDEX, index()),
                               ("embedder", ENV_EMBEDDER, embedder()),
                               ("model", ENV_MODEL, model())):
        if value is not None:
            parts.append(f"{label}={value}" + (f" [{name}]" if _env(name) else ""))
    return ", ".join(parts)
