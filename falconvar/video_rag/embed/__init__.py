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
#: `run` is the only public spelling. The function in `driver.py` is named for
#: its component so a traceback frame says which one failed -- eight frames
#: called `run` carry no information -- but exporting both names would give the
#: library two ways to say the same thing, and `load`, `build` and `available`
#: collide across components anyway, so a bare-name style needs aliases the
#: moment a caller wants a second thing from the same module.
from .driver import DEFAULT_INDEX, collect, main, run
#: Embedding is not a video_rag concern -- it takes text and returns vectors,
#: and both tiers do it. It lives in `shared/models/` beside `llm` and
#: `providers`, which it already used. Re-exported because `embed.build` and
#: `embed.available()` are part of this component's surface.
from ...shared.models.embedders import (Embedder, EmbedderUnavailable,
                                        available, build, query_vector)
from .units import Unit, from_descriptions, from_transcript, render

__all__ = ["DEFAULT_INDEX", "Embedder", "EmbedderUnavailable",
           "Unit", "available", "build", "collect", "from_descriptions",
           "from_transcript", "indexes", "main", "query_vector", "readable",
           "render", "run"]
