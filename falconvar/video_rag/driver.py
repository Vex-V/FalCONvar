"""The video_rag driver: extraction as a list of component calls, and the search.

Every step is one component's `run()`, in dependency order. Nothing here
decodes, loads a model or builds a path.

Order is not decided here -- it falls out of what the chosen policy depends on,
and `boundaries.POLICIES` is that table:

    uniform    nothing;  arithmetic over a duration `media.json` already has
    scene      the picture;  boundaries.evidence decodes and scores it
    vad        a transcript; audio must finish first
    speaker    a transcript; audio must finish first

`Options` carries only what extraction must decide -- which file, which grid,
which modalities, what to look at, who answers, where output goes. Per-stage
tuning lives on each component's own CLI.

The bottom of the file is the whole of what `aggregates` may ask of this tier:
`vocabulary()`, so an input naming a question or a sampler can be checked
before a job is queued. It was seven functions; the other six were shared
concerns misfiled here and now live in `shared/`, which both tiers import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..shared import paths
from ..shared.contracts.documents import Produced
from . import audio, boundaries, cut, describe, embed, media, video
from .retrieve import search, videos

#: Every component, in the order it can run.
COMPONENTS = ("media", "audio", "boundaries.evidence", "boundaries",
              "video", "cut", "describe", "embed")

DEFAULT_INDEX = embed.DEFAULT_INDEX


@dataclass
class Options:
    """The shape of one extraction. Tuning lives on the component CLIs."""

    source: Path
    video_id: Optional[str] = None
    policy: str = "uniform"                  # decides who runs first
    use_video: bool = True
    use_audio: bool = True
    sampler: str = "uniform"                 # what to look at
    # Who answers: a provider or `provider/model`. None resolves through
    # `shared.models.providers` when the stage runs -- FALCONVAR_* from .env,
    # then openai -- rather than being captured at import, before .env is read.
    describer: Optional[str] = None          # frames -> answers
    embedder: Optional[str] = None           # text -> vectors
    sink: str = "file"                       # where documents go
    index: str = DEFAULT_INDEX               # where vectors go


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
    # Checked here because the alternative is where it used to be caught: in
    # `describe`, after ingest has decoded the whole video. `yolo:overvew` was
    # a 202 that ran media, audio, boundaries and a full video pass before
    # failing on the typo -- which is the late failure this function exists to
    # prevent. `question_for` falls back to the scene question, so a spelling
    # nobody checks is a run that completes and answers something nobody asked.
    # `video` still does not import `describe`: this driver is the composition
    # root, and a sampler records its question as an opaque string.
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

    # Every model extraction will call: a known provider, able to do the job,
    # with a model and a key. A missing key discovered by `describe` arrives
    # after the whole video has been decoded. Whether a local server is up is
    # not asked: that would make validation a network call.
    from ..shared.models import providers
    wanted = [("embed", options.embedder)]
    if options.use_video:
        wanted.append(("describe", options.describer))
    for role, spec in wanted:
        problems += providers.problems(role, spec)
    return problems


def process(options: Options,
            on_step: Optional[Callable[[str, Optional[Produced]], None]] = None
            ) -> Run:
    """One extraction, top to bottom. Every step is a component's `run()`."""
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
        step(describe.run(video_id, options.describer, sink=options.sink))

    # 8 · vectors, from both modalities
    starting("embed")
    step(embed.run(video_id, options.embedder, index_name=options.index))

    return run


# --------------------------------------------- what aggregates may ask of this

def video_rag(source: str | Path,
              video_id: Optional[str] = None,
              policy: str = "uniform",
              use_video: bool = True,
              use_audio: bool = True,
              sampler: str = "uniform",
              describer: Optional[str] = None,
              embedder: Optional[str] = None,
              sink: str = "file",
              index: str = DEFAULT_INDEX,
              on_step: Optional[Callable[[str, Optional[Produced]], None]] = None
              ) -> Run:
    """The whole extraction, as keyword arguments. `process` with an `Options`.

    The library front door. `Options` is the validated record a form or an API
    posts and `process` is what runs it; this is the spelling a caller writes
    by hand, so the arguments are named and checked at the call rather than
    assembled into a dataclass first.

    Per-stage tuning is deliberately absent -- a scene threshold, a sampler's
    `confidence`, an audio `compute_type`. Those live on the component that
    owns them, and a caller who wants them drives the components directly;
    every one is `run(video_id, ...)`, in the order this function uses.
    """
    return process(Options(
        source=Path(source), video_id=video_id, policy=policy,
        use_video=use_video, use_audio=use_audio, sampler=sampler,
        describer=describer, embedder=embedder, sink=sink, index=index,
    ), on_step)


# ------------------------------------------- the one thing aggregates asks
#
# This used to be seven functions. Six of them were shared concerns misfiled
# here because video_rag was written first -- the field builder, unit
# rendering, the embedder registry, the vector writer -- and loading a
# video's documents, which is `shared.paths` plus a dataclass and needs no
# help from this tier. They now live in `shared/`, which both tiers already
# import, and `aggregates` reaches none of this module for them.
#
# This one cannot move. Its body is `describe`'s question vocabulary and the
# sampler registry, so `shared` holding it would mean `shared` importing a
# tier -- a cycle, and the one rule that currently holds without exception.
# `aggregates` holding it would be the same dependency spelled wider: two
# component imports instead of one call, coupled to `shape_of`'s internal
# return shape rather than to flat data.
#
# And it cannot be read from a video's output either. `aggregates.validate`
# takes no `video_id` -- a request is checked before any video is named, which
# is what makes a typo a 422 at submit time rather than a job that runs and
# finds nothing.
def vocabulary() -> dict[str, Any]:
    """What an aggregate's input may name: every sampler, and every question
    with its fields -- `{field: [entry keys]}` for a list of objects, `None`
    for anything else."""
    from .describe import library
    from .video import samplers as _samplers

    def fields(question: str) -> dict[str, Optional[list[str]]]:
        out: dict[str, Optional[list[str]]] = {}
        for name, spec in (library.shape_of(question).get("fields") or {}).items():
            items = spec.get("items") if isinstance(spec, dict) else None
            nested = items.get("properties") if isinstance(items, dict) else None
            out[name] = list(nested) if nested else None
        return out

    return {"samplers": _samplers.available(),
            "questions": {q: fields(q) for q in library.questions()}}


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Extract once: media to embed. Per-stage tuning lives on each "
                    "component's own CLI, e.g. `python -m falconvar.video_rag.video`.")
    ap.add_argument("source", type=Path)
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--policy", default="uniform", choices=sorted(boundaries.POLICIES))
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
    ap.add_argument("--sink", default="file",
                    help="where documents go: file | supabase | both")
    ap.add_argument("--index", default=DEFAULT_INDEX,
                    help=f"where vectors go; known: {', '.join(embed.indexes.available())}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    options = Options(
        source=args.source, video_id=args.video_id, policy=args.policy,
        use_video=not args.no_video, use_audio=not args.no_audio,
        sampler=args.sampler, describer=args.describer, embedder=args.embedder,
        sink=args.sink, index=args.index)
    return report(options, lambda on_step: process(options, on_step), args.json)


def report(options: Any, execute: Callable[[Callable], Run], as_json: bool) -> int:
    """Validate, run and print, for this CLI and `workflow`'s."""
    import json

    problems = validate(options) if isinstance(options, Options) else []
    if problems:
        for problem in problems:
            print(f"error: {problem}")
        return 2

    def on_step(component: str, produced: Optional[Produced]) -> None:
        if as_json:
            return
        if produced is None:                       # about to run
            print(f"  {component:<22} ...", flush=True)
            return
        print(f"  {component:<22} -> {', '.join(produced.artifacts) or '-'}")

    if not as_json:
        print(f"{options.source}   policy={options.policy}")
    try:
        run = execute(on_step)
    except Exception as exc:                             # noqa: BLE001
        print(f"error: {exc}")
        return 1

    if as_json:
        print(json.dumps(run.as_dict(), indent=2))
        return 0
    for name, why in run.skipped.items():
        print(f"  {name:<22} -- skipped: {why}")
    print(f"\n{run.video_id} -> {paths.home(run.video_id)}")
    print(f"  artifacts: {', '.join(paths.present(run.video_id))}")
    return 0


__all__ = ["COMPONENTS", "DEFAULT_INDEX", "Options", "Run", "main", "process",
           "report", "search", "validate", "video_rag", "videos", "vocabulary"]
