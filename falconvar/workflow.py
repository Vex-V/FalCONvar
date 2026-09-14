"""The whole run, as a list of component calls.

Every step is one component's `run()`, in dependency order. Nothing here
decodes, loads a model or builds a path.

Order is not decided here -- it falls out of what the chosen policy depends on,
and `boundaries.POLICIES` is that table:

    uniform    nothing;  arithmetic over a duration `media.json` already has
    scene      the picture;  boundaries.evidence decodes and scores it
    vad        a transcript; audio must finish first
    speaker    a transcript; audio must finish first

`Options` carries only what a run must decide -- which file, which grid, which
modalities, what to look at, what it may cost, where output goes. Per-stage
tuning lives on each component's own CLI.

`process` emits `(component, Produced)` to an optional callback: a terminal
prints it, a job runner keeps the latest as the answer to a poll.
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
    # Who answers. None resolves through `shared.providers` when the stage runs
    # -- the call, then FALCONVAR_* from .env, then openai -- rather than being
    # captured at import, before .env has been read. A name may carry its
    # model: `ollama/gemma3:4b`.
    describer: Optional[str] = None          # frames -> answers
    describe_model: Optional[str] = None
    embedder: Optional[str] = None           # text -> vectors
    embed_model: Optional[str] = None
    llm: Optional[str] = None                # the `llm` aggregate tier
    llm_model: Optional[str] = None
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

    # Both halves of every `name:question` pair, against the two registries.
    #
    # Imported here rather than at module scope: this function is a composition
    # root, and the one thing it wants the vocabularies for is catching a typo.
    # `video` still does not import `describe` -- a sampler records its question
    # as an opaque string.
    #
    # Checked here because the alternative is where it used to be caught: in
    # `describe`, after ingest has decoded the whole video. `yolo:overvew` was
    # a 202 that ran media, audio, boundaries and a full video pass before
    # failing on the typo -- which is the late failure this function exists to
    # prevent. `question_for` falls back to the scene question, so a spelling
    # nobody checks is a run that completes and answers something nobody asked.
    if options.use_video:
        from .describe import prompts
        from .video import samplers as _samplers
        from .video.driver import parse_spec, split_specs
        known_samplers, known_questions = _samplers.available(), prompts.questions()
        for spec in split_specs(options.sampler):
            name, asked = parse_spec(spec)
            if name not in known_samplers:
                problems.append(f"unknown sampler {name!r} in {spec!r}; "
                                f"known: {', '.join(known_samplers)}")
            for question in asked:
                if question not in known_questions:
                    problems.append(f"unknown question {question!r} in {spec!r}; "
                                    f"known: {', '.join(known_questions)}")

    # Every model this run will call: a known provider, able to do the job,
    # with a model and a key. Found here for the same reason as a question
    # typo -- a missing ANTHROPIC_API_KEY discovered by `describe` arrives
    # after the whole video has been decoded. Whether a local server is up is
    # not asked: that would make validation a network call.
    from .shared import providers
    wanted = [("embed", options.embedder, options.embed_model)]
    if options.use_video:
        wanted.append(("describe", options.describer, options.describe_model))
    if options.tier == "llm":
        wanted.append(("llm", options.llm, options.llm_model))
    for role, name, model in wanted:
        problems += providers.problems(role, name, model)
    return problems


def process(options: Options,
            on_step: Optional[Callable[[str, Optional[Produced]], None]] = None
            ) -> Run:
    """One run, top to bottom. Every step is a component's `run()`."""
    problems = validate(options)
    if problems:
        raise ValueError("; ".join(problems))

    say = on_step or (lambda component, produced: None)
    run = Run(video_id="")

    def starting(name: str) -> None:
        """Announce a component before it runs.

        The name is the caller's, because before a component runs it is the
        only word for it there is -- `boundaries.evidence` answers as
        `boundaries.scenes` or `boundaries.audio`, naming which modality
        supplied it, and that is not knowable in advance. Without this, the
        latest thing a poller hears is the last component to *finish*, so the
        longest stage in the run reports as the one before it.
        """
        say(name, None)

    def step(produced: Produced) -> Produced:
        run.steps.append(produced)
        say(produced.component, produced)
        return produced

    # 1 · what the file is
    starting("media")
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
        starting("audio")
        step(audio.run(video_id, sink=options.sink))

    # 3 · boundary evidence, if this policy needs any
    starting("boundaries.evidence")
    evidence = boundaries.evidence(video_id, options.policy, sink=options.sink)
    if evidence is None:
        run.skipped["boundaries.evidence"] = (
            f"policy {options.policy!r} needs none -- it is arithmetic over "
            "the container duration")
    else:
        step(evidence)

    # 4 · THE GRID
    starting("boundaries")
    step(boundaries.run(video_id, options.policy, sink=options.sink))

    # 5 · the picture, onto that grid. The grid is an input here and is never
    #     edited, which is why nothing needs a Chunker.
    if use_video:
        starting("video")
        step(video.run(video_id, options.sampler, sink=options.sink))

    # 6 · the transcript, onto the same grid. Cheap: Whisper timestamped every
    #     word, so this can be redone against a different grid for nothing.
    if use_audio:
        starting("cut")
        step(cut.run(video_id, options.sink))

    # 7 · one answer per (chunk, sampler)
    if use_video:
        starting("describe")
        step(describe.run(video_id, options.describer, options.describe_model,
                          sink=options.sink))

    # 8 · vectors, from both modalities
    starting("embed")
    step(embed.run(video_id, options.embedder, options.embed_model,
                   index_name=options.index))

    # 9 · video-level structure
    starting("aggregate")
    step(aggregate.run(video_id, options.tier, sink=options.sink,
                       llm=options.llm, model=options.llm_model))

    return run


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Run the whole pipeline once. Per-stage tuning lives on "
                    "each component's own CLI, e.g. `python -m falconvar.video`.")
    ap.add_argument("source", type=Path)
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--policy", default="uniform",
                    choices=sorted(boundaries.POLICIES))
    ap.add_argument("--sampler", default="uniform",
                    help="comma-separated; any may carry a question after a "
                         "colon, e.g. `yolo:overview`")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--describer", default=None,
                    help="a provider or provider/model, e.g. ollama/gemma3:4b; "
                         "default FALCONVAR_DESCRIBER, then openai")
    ap.add_argument("--describe-model", default=None)
    ap.add_argument("--embedder", default=None,
                    help="a provider or provider/model, e.g. local; "
                         "default FALCONVAR_EMBEDDER, then openai")
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--llm", default=None,
                    help="who answers --tier llm; default FALCONVAR_LLM, then openai")
    ap.add_argument("--llm-model", default=None)
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
        describe_model=args.describe_model, embedder=args.embedder,
        embed_model=args.embed_model, llm=args.llm, llm_model=args.llm_model,
        tier=args.tier, sink=args.sink, index=args.index)

    problems = validate(options)
    if problems:
        for problem in problems:
            print(f"error: {problem}")
        return 2

    def report(component: str, produced: Optional[Produced]) -> None:
        if args.json:
            return
        if produced is None:                       # about to run
            print(f"  {component:<22} ...", flush=True)
            return
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
