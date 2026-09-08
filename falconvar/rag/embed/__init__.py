"""8 · embed -- documents to vectors, keyed by a hash of the text.

Both modalities land here: a description from the picture and a transcript
chunk from the soundtrack are both text with a span and some bound structure,
so audio needed no index of its own and no code past `units.from_transcript`.

Two real indexes, `qdrant` and `supabase`, both fusing a dense and a lexical
ranking. Plus `embedded.json`, written alongside them -- the text without the
vectors, for reading rather than searching.
"""

from __future__ import annotations

from . import indexes, readable
from .driver import DEFAULT_EMBEDDER, DEFAULT_INDEX, collect, main, run
from .embedders import Embedder, EmbedderUnavailable, available, build
from .units import Unit, from_descriptions, from_transcript, render

__all__ = ["DEFAULT_EMBEDDER", "DEFAULT_INDEX", "Embedder", "EmbedderUnavailable",
           "Unit", "available", "build", "collect", "from_descriptions",
           "from_transcript", "indexes", "main", "readable", "render", "run"]
