"""The whole run, as a list of component calls.

This file exists to be *read*. Every step is one component's `run()`, in
dependency order, and nothing here does work of its own -- no decoding, no
models, no path arithmetic.

**There is no ordering rule here.** `falconvar` has four cases in
`orchestrate.process` deciding who goes first; this has none, because the
answer falls out of what the chosen policy depends on:

    uniform    nothing.  arithmetic over a duration `media.json` already has
    scene      the picture.  boundaries.evidence decodes and scores it
    vad        a transcript. audio must finish first
    speaker    a transcript. audio must finish first

`boundaries.POLICIES` is that table and it is data. The only branch below is
"does this policy need evidence", which is a dependency, not a decision.

**It passes only what it must decide.** Which file, which grid, which
modalities, what to look at, and what it is allowed to cost. Every other knob
-- strides, thresholds, model names, frame-store scope -- is a component's own
business and is reachable on that component's CLI:

    python -m ver3.boundaries <id> --policy scene --stride 10 --threshold 40
    python -m ver3.video <id> --sampler yolo --per-second 4 --min-interval 3
    python -m ver3.describe <id> --describer openai --limit 5

Forwarding all of them here made this file 32 flags long and gave every default
two homes. A pipeline that runs once needs the shape of the run, not the tuning
of each stage.

**Progress is a callback, not a print.** A terminal wants lines as they happen
and a job runner wants the latest state to answer a poll with, so `process`
emits `(component, Produced)` and neither policy lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import aggregate, audio, boundaries, cut, describe, media, video
from .rag import embed
from .shared import paths
from .shared.documents import Produced

#: Every component, in the order it can run.
COMPONENTS = ("media", "audio", "boundaries.evidence", "boundaries",
              "video", "cut", "describe", "embed", "aggregate")


@dataclass
class Options:
    """The shape of one run. Tuning lives on the component CLIs."""

    source: Path
    video_id: Optional[str] = None
    policy: str = "uniform"                  # decides who runs first
    use_video: bool = True
    use_audio: bool = True
    sampler: str = "uniform"                 # what to look at
    describer: str = describe.driver.DEFAULT_DESCRIBER   # costs money
    embedder: str = embed.DEFAULT_EMBEDDER               # costs money
    tier: str = "free"                                   # a cost ceiling
    sink: str = "file"                       # where documents go
    index: str = embed.DEFAULT_INDEX         # where vectors go


@dataclass
class Run:
    video_id: str
    steps: list[Produced] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"video_id": self.video_id,
                "steps": [s.as_dict() for s in self.steps],
                "skipped": self.skipped,
                "artifacts": paths.present(self.video_id)}


def validate(options: Options) -> list[str]:
    """Contradictions only, as messages. Empty means valid.

    Just the ones no single component can see -- a policy derived from a stream
    this run is not reading. Everything else is checked by the component that
    owns it, where the message can be specific. Returned rather than raised so
    a CLI prints them all at once and an API can answer 422 with the list.
    """
    problems: list[str] = []
    if options.policy not in boundaries.POLICIES:
        problems.append(f"policy must be one of {', '.join(boundaries.POLICIES)}")
    else:
        needs = boundaries.POLICIES[options.policy]
        if needs == "video" and not options.use_video:
            problems.append(f"policy {options.policy!r} is found in the picture, "
                            "which this run is not reading")
        if needs == "audio" and not options.use_audio:
            problems.append(f"policy {options.policy!r} is derived from the "
                            "soundtrack, which this run is not reading")
    if not options.use_video and not options.use_audio:
        problems.append("nothing to do: read the picture, the soundtrack, or both")
    if not Path(options.source).exists():
        problems.append(f"{options.source} does not exist")
    return problems


def process(options: Options,
            on_step: Optional[Callable[[str, Produced], None]] = None) -> Run:
    """One run, top to bottom. Every step is a component's `run()`."""
    problems = validate(options)
    if problems:
        raise ValueError("; ".join(problems))

    say = on_step or (lambda component, produced: None)
    run = Run(video_id="")

    def step(produced: Produced) -> Produced:
        run.steps.append(produced)
        say(produced.component, produced)
        return produced

    # 1 · what the file is
    first = step(media.run(options.source, options.video_id, options.sink))
    run.video_id = video_id = first.video_id
    described = media.load(video_id)

    # Whether a file carries a soundtrack is a property of the file, not of the
    # request, so it is found here rather than in `validate`.
    use_audio = options.use_audio and described.has_audio
    use_video = options.use_video and described.has_video
    for name, wanted, present in (("audio", options.use_audio, described.has_audio),
                                  ("video", options.use_video, described.has_video)):
        if wanted and not present:
            run.skipped[name] = f"the file carries no {name} stream"
            if boundaries.POLICIES[options.policy] == name:
                raise ValueError(f"policy {options.policy!r} needs the {name} "
                                 f"stream, and {described.path} has none")

    # 2 · the soundtrack. Before the grid when the policy needs a transcript to
    #     derive one; the order is the dependency, not a rule about modalities.
    if use_audio:
        step(audio.run(video_id, sink=options.sink))

    # 3 · boundary evidence, if this policy needs any
    evidence = boundaries.evidence(video_id, options.policy, sink=options.sink)
    if evidence is None:
        run.skipped["boundaries.evidence"] = (
            f"policy {options.policy!r} needs none -- it is arithmetic over "
            "the container duration")
    else:
        step(evidence)

    # 4 · THE GRID
    step(boundaries.run(video_id, options.policy, sink=options.sink))

    # 5 · the picture, onto that grid. The grid is an input here and is never
    #     edited, which is why nothing needs a Chunker.
    if use_video:
        step(video.run(video_id, options.sampler, sink=options.sink))

    # 6 · the transcript, onto the same grid. Cheap: Whisper timestamped every
    #     word, so this can be redone against a different grid for nothing.
    if use_audio:
        step(cut.run(video_id, options.sink))

    # 7 · one answer per (chunk, sampler)
    if use_video:
        step(describe.run(video_id, options.describer, sink=options.sink))

    # 8 · vectors, from both modalities
    step(embed.run(video_id, options.embedder, index_name=options.index))

    # 9 · video-level structure
    step(aggregate.run(video_id, options.tier, sink=options.sink))

    return run


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Run the whole pipeline once. Per-stage tuning lives on "
                    "each component's own CLI, e.g. `python -m ver3.video`.")
    ap.add_argument("source", type=Path)
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--policy", default="uniform",
                    choices=sorted(boundaries.POLICIES))
    ap.add_argument("--sampler", default="uniform",
                    help="comma-separated; any may carry a question after a "
                         "colon, e.g. `yolo:overview`")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--describer", default=describe.driver.DEFAULT_DESCRIBER)
    ap.add_argument("--embedder", default=embed.DEFAULT_EMBEDDER)
    ap.add_argument("--tier", default="free", choices=aggregate.TIERS)
    ap.add_argument("--sink", default="file",
                    help="where documents go: file | supabase | both")
    ap.add_argument("--index", default=embed.DEFAULT_INDEX,
                    help=f"where vectors go; known: "
                         f"{', '.join(embed.indexes.available())}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    options = Options(
        source=args.source, video_id=args.video_id, policy=args.policy,
        use_video=not args.no_video, use_audio=not args.no_audio,
        sampler=args.sampler, describer=args.describer,
        embedder=args.embedder, tier=args.tier, sink=args.sink,
        index=args.index)

    problems = validate(options)
    if problems:
        for problem in problems:
            print(f"error: {problem}")
        return 2

    def report(component: str, produced: Produced) -> None:
        if not args.json:
            print(f"  {component:<22} -> {', '.join(produced.artifacts) or '-'}")

    if not args.json:
        print(f"{args.source}   policy={args.policy}")
    try:
        run = process(options, on_step=report)
    except Exception as exc:                             # noqa: BLE001
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(run.as_dict(), indent=2))
        return 0

    for name, why in run.skipped.items():
        print(f"  {name:<22} -- skipped: {why}")
    print(f"\n{run.video_id} -> {paths.home(run.video_id)}")
    print(f"  artifacts: {', '.join(paths.present(run.video_id))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
