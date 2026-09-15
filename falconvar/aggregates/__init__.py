"""aggregates -- higher-level answers over what video_rag extracted.

The second tier. Reads the documents video_rag wrote, through video_rag's
driver -- never the video, and never a video_rag component. Answers the
questions embeddings cannot: counts, coverage, who dominated, how much of this
is speech, what the whole video is about, who is who across chunks.

Three tiers, cheapest first. `free` is arithmetic, `local` adds GPU models,
`llm` adds paid calls. Both dear tiers are registered lazily, so importing this
pulls in neither torch nor an API client and needs no key.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import TIERS, Context, missing, resolve_order
from .statistics import (CoverageAggregator, SpeakersAggregator,
                         StatsAggregator)

REGISTRY: dict[str, Any] = {
    cls.name: cls for cls in
    (StatsAggregator, SpeakersAggregator, CoverageAggregator)
}

#: name -> ("module:Class", tier, about). Resolved on first use, so a `--tier
#: free` run never imports the LLM client and never needs a key.
_LAZY: dict[str, tuple[str, str, str]] = {
    "ner": ("model.ner:NERAggregator", "local",
            "named entities, and which chunks each appears in"),
    "sentiment": ("model.sentiment:SentimentAggregator", "local",
                  "tone per chunk, and where it turns"),
    "summary": ("llm.summary:SummaryAggregator", "llm",
                "what the whole video is about, in one pass over every chunk"),
    "chapters": ("llm.chapters:ChaptersAggregator", "llm",
                 "a table of contents: contiguous chapters over the video"),
    "events": ("llm.events:EventsAggregator", "llm",
               "discrete things that happened, each pinned to a chunk"),
    "entities": ("llm.entities:EntitiesAggregator", "llm",
                 "the same person or thing across chunks, and what each did"),
}

#: What each lazy entry costs, without importing it. Needed because `--tier
#: free` must be answerable without loading the modules it is excluding.
TIER_OF: dict[str, str] = {**{n: c.tier for n, c in REGISTRY.items()},
                           **{n: t for n, (_, t, _) in _LAZY.items()}}
ABOUT: dict[str, str] = {**{n: c.about for n, c in REGISTRY.items()},
                         **{n: a for n, (_, _, a) in _LAZY.items()}}


def resolve(name: str) -> Any:
    if name in REGISTRY:
        return REGISTRY[name]
    if name not in _LAZY:
        raise KeyError(f"unknown aggregator {name!r}; "
                       f"known: {', '.join(available())}")
    module_name, class_name = _LAZY[name][0].split(":")
    module = importlib.import_module(f".{module_name}", __package__)
    REGISTRY[name] = getattr(module, class_name)
    return REGISTRY[name]


def available() -> list[str]:
    return sorted(set(REGISTRY) | set(_LAZY))


def about(name: str) -> str:
    return ABOUT[name]


from .driver import context_for, load, main, run, validate  # noqa: E402

__all__ = ["REGISTRY", "TIERS", "Context", "about", "available", "context_for",
           "load", "main", "missing", "resolve_order", "run", "validate"]
