"""Video-level structure over what the chunk stages wrote.

`retrieve` finds the moment that best matches a question. That is the wrong
shape for a whole class of questions -- how many, who most, what overall, which
part is unlike the rest -- and this stage answers those instead. Nothing here
looks at a video; everything reads documents the other stages produced.

Registered by name, like the samplers and the describers, and resolved into a
run order that puts dependencies first and **drops** anything whose sources are
missing. That last part is the important one: a speaker aggregate on silent
CCTV is not an error to report, it is a question that does not apply, and
running it anyway would produce an empty result claiming otherwise.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Type

from .base import TIERS, Aggregator, Context
from .speakers import SpeakerStatsAggregator
from .stats import StatsAggregator

#: name -> "module:ClassName", resolved on first use, so importing this package
#: pulls in neither a model runtime nor an API client.
_LAZY: dict[str, str] = {
    "novelty": "novelty:NoveltyAggregator",
    "summary": "summary:SummaryAggregator",
    "chapters": "chapters:ChaptersAggregator",
    "events": "events:EventsAggregator",
    "ner": "ner:NERAggregator",
    "sentiment": "sentiment:SentimentAggregator",
}
_REGISTRY: dict[str, Type] = {}

#: What each lazy entry costs, without importing it. Needed because `--tier
#: free` must be answerable without loading the modules it is excluding.
TIER_OF: dict[str, str] = {
    "stats": "free", "speakers": "free", "novelty": "free",
    "ner": "local", "sentiment": "local",
    "summary": "llm", "chapters": "llm", "events": "llm",
}


#: One line per aggregator, for a caller listing what it could ask for. Kept
#: beside `TIER_OF` for the same reason: answerable without importing the
#: module, so a browser can render the menu without a GPU being touched.
ABOUT: dict[str, str] = {
    "stats": "counts and coverage: chunks, samplers, words, frames",
    "speakers": "who talked, for how long, over how many turns",
    "novelty": "which chunk is least like the rest, by its own vectors",
    "ner": "the names, places and organisations mentioned",
    "sentiment": "the tone of each chunk, and the arc across the video",
    "summary": "what the whole video is about, in one pass over every chunk",
    "chapters": "where the subject changes, and what each part covers",
    "events": "what happened, as a list with timestamps",
}


def about(name: str) -> str:
    """One line on what an aggregate answers. Unknown names get a placeholder
    rather than an error: this is labelling, not dispatch."""
    return ABOUT.get(name, "a video-level result")


def register(cls: Type) -> Type:
    _REGISTRY[cls.id] = cls
    return cls


register(StatsAggregator)
register(SpeakerStatsAggregator)


def _resolve(name: str) -> Type:
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name not in _LAZY:
        raise KeyError(f"unknown aggregator {name!r}; known: {', '.join(available())}")
    import importlib

    module_name, class_name = _LAZY[name].split(":")
    module = importlib.import_module(f".{module_name}", __package__)
    return register(getattr(module, class_name))


def build(name: str, **kwargs) -> Aggregator:
    return _resolve(name)(**kwargs)


def available() -> list[str]:
    return sorted(set(_REGISTRY) | set(_LAZY))


def by_tier(tier: str) -> list[str]:
    """Every aggregator at or below ``tier``, cheapest first.

    Ordered so that asking for `llm` still runs the free ones, and a run that
    dies partway has produced the cheap results rather than none.
    """
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}; known: {', '.join(TIERS)}")
    ceiling = TIERS.index(tier)
    return [name for name in available()
            if TIERS.index(TIER_OF.get(name, "llm")) <= ceiling]


def resolve_order(names: Sequence[str], sources: Iterable[str]) -> tuple[list[str], dict[str, str]]:
    """Order the requested aggregators, dropping what cannot run.

    A dependency naming a source (`yolo`, `transcript`) is a requirement on the
    video; one naming another aggregator pulls it in and orders it first.
    Returns the runnable order and, for everything dropped, why -- because
    "speakers did not run" is only useful alongside "this video has no speech".
    """
    have = set(sources)
    ordered: list[str] = []
    skipped: dict[str, str] = {}
    visiting: set[str] = set()

    def visit(name: str, chain: tuple[str, ...] = ()) -> bool:
        if name in ordered:
            return True
        if name in skipped:
            return False
        if name in visiting:
            raise ValueError("circular aggregator dependency: "
                             + " -> ".join(chain + (name,)))
        aggregator = _resolve(name)
        visiting.add(name)
        try:
            for dependency in aggregator.depends_on:
                if dependency in _REGISTRY or dependency in _LAZY:
                    if not visit(dependency, chain + (name,)):
                        skipped[name] = f"needs {dependency}, which cannot run"
                        return False
                elif dependency not in have:
                    skipped[name] = f"needs {dependency}, which this video has no output for"
                    return False
        finally:
            visiting.discard(name)
        ordered.append(name)
        return True

    for name in names:
        visit(name)
    return ordered, skipped


__all__ = ["ABOUT", "Aggregator", "Context", "TIERS", "TIER_OF", "about",
           "available", "build", "by_tier", "register", "resolve_order"]
