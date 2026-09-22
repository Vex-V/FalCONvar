"""video_rag -- a video in, a searchable index of moments out, and the search.

The extraction half of FalCONvar, and a complete RAG engine on its own:

    media        1  what streams the file carries
    audio        2  the soundtrack, scanned whole
    boundaries 3+4  THE GRID, from the picture or the soundtrack
    video        5  which frames each sampler keeps
    cut          6  the transcript, onto the grid
    describe     7  one model answer per (chunk, sampler:question)
    embed        8  both modalities to vectors, keyed by a hash of the text
    retrieve        a query to ranked moments over what embed built

`driver.py` runs them in the order the grid policy implies, the way
`workflow.py` runs this driver and `aggregates`'. Nothing in this package reads
anything `aggregates` produced.
"""

from __future__ import annotations

from typing import Any

#: Lazily, so `import falconvar.video_rag` stays two modules and 1.3 ms.
#: Binding `process` here eagerly would import every component -- 53 modules,
#: 352 ms and `av` -- for anyone who only wanted one of them. PEP 562 makes
#: `from falconvar.video_rag import video_rag` resolve through this instead.
_LAZY = {"video_rag": ("driver", "video_rag"),
         "process": ("driver", "process"),
         "Options": ("driver", "Options"),
         "Run": ("driver", "Run"),
         "validate": ("driver", "validate"),
         "search": ("retrieve", "search")}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib
        module, attribute = _LAZY[name]
        return getattr(importlib.import_module(f".{module}", __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


__all__ = ["Options", "Run", "process", "search", "validate", "video_rag"]

