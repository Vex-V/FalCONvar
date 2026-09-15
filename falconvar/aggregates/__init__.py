"""aggregates -- higher-level answers over what video_rag extracted.

The second tier. Reads the documents video_rag wrote, through video_rag's
driver -- never the video, and never a video_rag component. Answers the
questions embeddings cannot: counts, coverage, who dominated, how much of this
is speech, what the whole video is about, who is who across chunks.

Three tiers, cheapest first. `free` is arithmetic, `local` adds GPU models,
`llm` adds paid calls. Both dear tiers are registered lazily, so importing this
pulls in neither torch nor an API client and needs no key.

Two sources of aggregators. Code: `stats`, `speakers`, `coverage`, `ner`,
`sentiment`. Data: every prompt and link profile in `definitions` -- `summary`,
`chapters`, `events`, `entities:people` and whatever a user has added -- each run
by its kind's runner.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional, Sequence

from . import definitions
from .base import TIERS, Context, missing
from .statistics import (CoverageAggregator, SpeakersAggregator,
                         StatsAggregator)

REGISTRY: dict[str, Any] = {
    cls.name: cls for cls in
    (StatsAggregator, SpeakersAggregator, CoverageAggregator)
}

#: name -> ("module:Class", tier, about). Resolved on first use, so a `--tier
#: free` run never imports torch.
_LAZY: dict[str, tuple[str, str, str]] = {
    "ner": ("model.ner:NERAggregator", "local",
            "named entities, and which chunks each appears in"),
    "sentiment": ("model.sentiment:SentimentAggregator", "local",
                  "tone per chunk, and where it turns"),
}

#: kind -> the runner every definition of that kind is built with.
RUNNERS: dict[str, str] = {
    "fold": "llm.fold:FoldAggregator",
    "spans": "llm.spans:SpansAggregator",
    "items": "llm.items:ItemsAggregator",
    "link": "llm.entities:EntitiesAggregator",
}


def _import(target: str) -> Any:
    module_name, class_name = target.split(":")
    return getattr(importlib.import_module(f".{module_name}", __package__), class_name)


def available() -> list[str]:
    """Every aggregator id: code first, then definitions. Read now, so a
    definition added through the API is runnable without a restart."""
    return [*REGISTRY, *_LAZY, *definitions.ids()]


def expand(names: Optional[Sequence[str] | str]) -> list[str]:
    """What `only` names, in run order. `entities` means every link profile."""
    if names is None:
        return available()
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    out: list[str] = []
    for name in names:
        found = ([i for i in definitions.ids() if i.startswith(definitions.PROFILE_PREFIX)]
                 if name == "entities" else [name])
        out += [n for n in found if n not in out]
    return out


def kind_of(name: str) -> Optional[str]:
    """A definition's kind; None for an aggregator written as code."""
    if name in REGISTRY or name in _LAZY:
        return None
    section, definition = definitions.locate(name)
    return "link" if section == "profiles" else definitions.get(section, definition)["kind"]


def tier_of(name: str) -> str:
    if name in REGISTRY:
        return REGISTRY[name].tier
    if name in _LAZY:
        return _LAZY[name][1]
    kind_of(name)                     # raises for an unknown definition
    return "llm"


def about(name: str) -> str:
    """What an aggregator -- or an answer id, `summary~severity` -- is about."""
    from .inputs import definition_of
    name = definition_of(name)
    if name in REGISTRY:
        return REGISTRY[name].about
    if name in _LAZY:
        return _LAZY[name][2]
    try:
        return definitions.get(*definitions.locate(name)).get("about", "")
    except definitions.DefinitionError:
        return ""


def takes_inputs(name: str) -> bool:
    return name not in REGISTRY


def build(name: str, llm: Optional[str] = None, embedder: Optional[str] = None) -> Any:
    """One aggregator, constructed. Only the llm tier takes a provider, and
    only a link profile an embedder; local models name their own checkpoints."""
    if name in REGISTRY:
        return REGISTRY[name]()
    if name in _LAZY:
        return _import(_LAZY[name][0])()
    kind = kind_of(name)
    runner = _import(RUNNERS[kind])
    return runner(name, llm, embedder) if kind == "link" else runner(name, llm)


from .driver import context_for, load, main, run, validate  # noqa: E402

__all__ = ["REGISTRY", "RUNNERS", "TIERS", "Context", "about", "available",
           "build", "context_for", "expand", "kind_of", "load", "main", "missing",
           "run", "takes_inputs", "tier_of", "validate"]
