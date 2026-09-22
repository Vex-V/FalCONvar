"""The whole run: extract with `video_rag`, then aggregate what it extracted.

Two tiers, and this file is the only place both are named:

    video_rag    a video in, a searchable index of moments out -- media, audio,
                 boundaries, video, cut, describe, embed, and retrieve over
                 what they built. A complete RAG engine on its own.
    aggregates   higher-level answers over what video_rag extracted -- counts,
                 speakers, summaries, chapters, entities. It never reads the
                 video, only the documents video_rag wrote.

Each tier has one driver, and each driver calls the components in its own
folder: the shape the pipeline always had, one level up. This file calls the
two drivers and nothing below them.

`Options` stays flat, because it is what a request supplies: the API's upload
form and `/capabilities.defaults` read it, and neither should need to know
which tier a field belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import aggregates
from .shared.contracts.documents import Produced
from .video_rag import driver as video_rag

#: Every component a run may invoke, in the order it can run.
COMPONENTS = (*video_rag.COMPONENTS, "aggregate")

Run = video_rag.Run


@dataclass
class Options:
    """The shape of one run. Tuning lives on the component CLIs."""

    source: Path
    video_id: Optional[str] = None
    policy: str = "uniform"                  # decides who runs first
    use_video: bool = True
    use_audio: bool = True
    sampler: str = "uniform"                 # what to look at
    describer: Optional[str] = None          # frames -> answers
    embedder: Optional[str] = None           # text -> vectors
    llm: Optional[str] = None                # the `llm` aggregate tier
    tier: str = "free"                       # a cost ceiling
    sink: str = "file"                       # where documents go
    index: str = video_rag.DEFAULT_INDEX     # where vectors go


def extraction(options: Options) -> video_rag.Options:
    """The part of a run video_rag decides."""
    return video_rag.Options(
        source=options.source, video_id=options.video_id, policy=options.policy,
        use_video=options.use_video, use_audio=options.use_audio,
        sampler=options.sampler, describer=options.describer,
        embedder=options.embedder, sink=options.sink, index=options.index)


def validate(options: Options) -> list[str]:
    """Everything either tier would refuse, before either runs."""
    return (video_rag.validate(extraction(options))
            + aggregates.validate(options.tier, options.llm))


def process(options: Options,
            on_step: Optional[Callable[[str, Optional[Produced]], None]] = None
            ) -> Run:
    """Extract, then aggregate."""
    problems = validate(options)
    if problems:
        raise ValueError("; ".join(problems))
    say = on_step or (lambda component, produced: None)

    run = video_rag.process(extraction(options), on_step)

    say("aggregate", None)
    produced = aggregates.run(run.video_id, options.tier, sink=options.sink,
                              llm=options.llm, embedder=options.embedder,
                              index=options.index)
    run.steps.append(produced)
    say(produced.component, produced)
    return run


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Run the whole pipeline once: extract, then aggregate. "
                    "Per-stage tuning lives on each component's own CLI, e.g. "
                    "`python -m falconvar.video_rag.video`.")
    ap.add_argument("source", type=Path)
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--policy", default="uniform")
    ap.add_argument("--sampler", default="uniform",
                    help="comma-separated; any may carry a question after a "
                         "colon, e.g. `yolo:overview`")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--describer", default=None,
                    help="a provider or provider/model, e.g. ollama/gemma3:4b; "
                         "default FALCONVAR_DESCRIBER, then openai")
    ap.add_argument("--embedder", default=None,
                    help="a provider or provider/model, e.g. local; "
                         "default FALCONVAR_EMBEDDER, then openai")
    ap.add_argument("--llm", default=None,
                    help="who answers --tier llm: a provider or provider/model; "
                         "default FALCONVAR_LLM, then openai")
    ap.add_argument("--tier", default="free", choices=aggregates.TIERS)
    ap.add_argument("--sink", default="file",
                    help="where documents go: file | supabase | both")
    ap.add_argument("--index", default=video_rag.DEFAULT_INDEX,
                    help="where vectors go: qdrant | supabase | both")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    options = Options(
        source=args.source, video_id=args.video_id, policy=args.policy,
        use_video=not args.no_video, use_audio=not args.no_audio,
        sampler=args.sampler, describer=args.describer,
        embedder=args.embedder, llm=args.llm,
        tier=args.tier, sink=args.sink, index=args.index)

    problems = validate(options)
    if problems:
        for problem in problems:
            print(f"error: {problem}")
        return 2
    return video_rag.report(options, lambda on_step: process(options, on_step),
                            args.json)


if __name__ == "__main__":
    raise SystemExit(main())
