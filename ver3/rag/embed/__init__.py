"""8 · embed -- documents to vectors, keyed by a hash of the text."""

from __future__ import annotations

from . import indexes
from .driver import (DEFAULT_EMBEDDER, DEFAULT_INDEX, collect, main, run)
from .embedders import Embedder, EmbedderUnavailable, available, build
from .indexes.local import LocalIndex, index_path
from .units import Unit, from_descriptions, from_transcript, render

__all__ = ["DEFAULT_EMBEDDER", "DEFAULT_INDEX", "Embedder", "indexes", "EmbedderUnavailable", "LocalIndex",
           "Unit", "available", "build", "collect", "from_descriptions",
           "from_transcript", "index_path", "main", "render", "run"]
