"""The whole run, as a list of component calls.

This file exists to be *read*. Every step is a call to one component's `run()`,
in dependency order, and nothing here does any work of its own -- no decoding,
no models, no path arithmetic. If the pipeline is hard to follow, that is a
problem with this file rather than with the components.

**There is no ordering rule here.** `falconvar` has four cases in
`orchestrate.process` deciding who goes first; this has none, because the
answer falls out of what the chosen policy depends on:

    uniform    nothing.  arithmetic over a duration `media.json` already has
    scene      the picture.  boundaries.evidence decodes and scores it
    vad        a transcript. listen must finish first
    speaker    a transcript. listen must finish first

`boundaries.POLICIES` is that table, and it is data. The only branch below is
"has the grid's evidence been produced yet", which is a dependency, not a
policy decision.

**Progress is a callback, not a print.** A terminal wants lines as they happen
and a job runner wants the latest state to answer a poll with; emitting
`(component, Produced)` lets neither policy live in here. It is also what the
API will attach to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from . import aggregate, audio, boundaries, cut, describe, media, video
from .rag import embed
from .shared import paths
from .shared.documents import Produced

#: Which components exist, in the order they can possibly run. A component that
#: is not yet built simply is not here; the workflow stops where the pipeline
#: stops rather than pretending.
COMPONENTS = ("media", "audio", "boundaries.evidence", "boundaries",
              "video", "cut", "describe", "embed", "aggregate")


@dataclass
class Options:
    """Everything a run can be asked for."""

    source: Path
    video_id: Optional[str] = None
    policy: str = "uniform"

    use_video: bool = True
    use_audio: bool = True

    # grid
    chunk_s: float = 20.0
    min_s: float = 5.0
    max_s: Optional[float] = None

    # scene evidence
    stride: int = 5
    scene_threshold: float = 27.0

    # speech evidence
    silence_s: float = 0.65

    # ingest
    sampler: str = "uniform"
    per_second: float = 1.0
    every_n: Optional[int] = None
    min_interval_s: float = 0.0
    max_per_chunk: Optional[int] = None
    # Named apart from `scene_threshold` on purpose: a dataclass lets a
    # repeated field name silently win, so two knobs that mean different
    # things must not share one.
    sampler_threshold: Optional[float] = None
    vocabulary: Optional[str] = None
    frame_store: bool = True
    store_scope: str = "sampled"

    # audio models
    transcriber: str = audio.driver.DEFAULT_TRANSCRIBER
    diarizer: str = audio.driver.DEFAULT_DIARIZER
    audio_model: Optional[str] = None
    language: Optional[str] = None

    # describe / embed / aggregate
    describer: str = describe.driver.DEFAULT_DESCRIBER
    describe_model: Optional[str] = None
    describe_limit: Optional[int] = None
    embedder: str = embed.DEFAULT_EMBEDDER
    embed_model: Optional[str] = None
    tier: str = "free"

    #: Stop after this component. The pipeline is a chain of artifacts, so
    #: stopping partway leaves a coherent prefix rather than a broken run.
    until: Optional[str] = None

    sink: str = "file"


@dataclass
class Run:
    """What a whole run produced, component by component."""

    video_id: str
    steps: list[Produced] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def artifact(self, name: str) -> Optional[str]:
        for step in self.steps:
            if name in step.artifacts:
                return step.artifacts[name]
        return None

    def as_dict(self) -> dict[str, Any]:
        return {"video_id": self.video_id,
                "steps": [s.as_dict() for s in self.steps],
                "skipped": self.skipped,
                "artifacts": paths.present(self.video_id)}


def _wanted(component: str, until: Optional[str]) -> bool:
    """Whether to run this component, given `--until`.

    Stopping is by name rather than by index, so adding a component does not
    renumber anything a caller might have written down.
    """
    if until is None:
        return True
    if until not in COMPONENTS:
        raise KeyError(f"unknown component {until!r}; "
                       f"known: {', '.join(COMPONENTS)}")
    return COMPONENTS.index(component) <= COMPONENTS.index(until)


def validate(options: Options) -> list[str]:
    """Everything wrong with these options, as messages. Empty means valid.

    Returned rather than raised, so a CLI prints them all at once and an API
    answers 422 with the list instead of each learning the rules again. What
    cannot be known without opening the file -- whether a track carries speech,
    whether the container reports a duration -- is not checked here; that is a
    property of the media, not of the request.
    """
    problems: list[str] = []
    if options.policy not in boundaries.POLICIES:
        problems.append(f"policy must be one of "
                        f"{', '.join(boundaries.POLICIES)}")
    if not options.use_video and not options.use_audio:
        problems.append("nothing to do: read the picture, the soundtrack, or both")
    needs = boundaries.POLICIES.get(options.policy)
    if needs == "video" and not options.use_video:
        problems.append(f"policy {options.policy!r} is found in the picture, "
                        "which this run is not reading")
    if needs == "audio" and not options.use_audio:
        problems.append(f"policy {options.policy!r} is derived from the "
                        "soundtrack, which this run is not reading")
    if options.chunk_s <= 0:
        problems.append("--chunk-duration must be positive")
    if options.stride < 1:
        problems.append("--stride must be >= 1")
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
    steps: list[Produced] = []
    skipped: dict[str, str] = {}

    def step(produced: Produced) -> Produced:
        steps.append(produced)
        say(produced.component, produced)
        return produced

    # ---- 1 · what the file is ------------------------------------------
    first = step(media.run(options.source, options.video_id, options.sink))
    video_id = first.video_id
    described = media.load(video_id)

    # Whether a file carries a soundtrack is a property of the file, not of the
    # request, so it is found here rather than in `validate`.
    use_audio = options.use_audio and described.has_audio
    use_video = options.use_video and described.has_video
    if options.use_audio and not described.has_audio:
        skipped["audio"] = "the file carries no audio stream"
    if options.use_video and not described.has_video:
        skipped["scenes"] = "the file carries no video stream"
    if boundaries.POLICIES[options.policy] == "audio" and not use_audio:
        raise ValueError(f"policy {options.policy!r} is derived from the "
                         f"soundtrack, and {described.path} has none")
    if boundaries.POLICIES[options.policy] == "video" and not use_video:
        raise ValueError(f"policy {options.policy!r} is found in the picture, "
                         f"and {described.path} has none")

    # ---- 2 · the soundtrack, if this run reads it ----------------------
    # Before the grid when the policy needs a transcript to derive one; the
    # order is the dependency, not a rule about modalities.
    if use_audio and _wanted("audio", options.until):
        step(audio.run(video_id, options.transcriber, options.diarizer,
                        options.audio_model, options.language, options.sink))

    # ---- 3 · boundary evidence, if this policy needs any ---------------
    produced = boundaries.evidence(video_id, options.policy, options.stride,
                                   options.scene_threshold, sink=options.sink,
                                   silence_s=options.silence_s)
    if produced is None:
        skipped["boundaries.evidence"] = (
            f"policy {options.policy!r} needs none -- it is arithmetic over "
            "the container duration")
    else:
        step(produced)

    # ---- 4 · THE GRID --------------------------------------------------
    step(boundaries.run(video_id, options.policy, options.chunk_s,
                        options.min_s, options.max_s, options.sink))

    # ---- 5 · the picture, onto that grid -------------------------------
    # After the grid, always. The grid is an input to this component and is
    # never edited by it, which is the difference from `falconvar` and the
    # reason nothing here needs a `Chunker`.
    if use_video and _wanted("video", options.until):
        vocab = ([v.strip() for v in options.vocabulary.split(",") if v.strip()]
                 if options.vocabulary else None)
        step(video.run(video_id, options.sampler, options.per_second,
                        options.every_n, options.min_interval_s,
                        options.max_per_chunk, options.sampler_threshold, vocab,
                        options.frame_store, options.store_scope, options.sink))
    else:
        skipped["video"] = "this run is not reading the picture"

    # ---- 6 · the transcript, onto that grid ----------------------------
    # After the grid, always -- and cheap, because Whisper timestamped every
    # word, so this can be redone against a different grid for nothing.
    if use_audio and _wanted("cut", options.until):
        step(cut.run(video_id, options.sink))

    # ---- 7 · one answer per (chunk, sampler) ---------------------------
    if use_video and _wanted("describe", options.until):
        step(describe.run(video_id, options.describer, options.describe_model,
                          None, options.describe_limit, True, options.sink))

    # ---- 8 · vectors, from both modalities -----------------------------
    if _wanted("embed", options.until):
        step(embed.run(video_id, options.embedder, options.embed_model))

    # ---- 9 · video-level structure -------------------------------------
    if _wanted("aggregate", options.until):
        step(aggregate.run(video_id, options.tier))

    return Run(video_id=video_id, steps=steps, skipped=skipped)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Run the ver3 pipeline.")
    ap.add_argument("source", type=Path)
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--policy", default="uniform",
                    choices=sorted(boundaries.POLICIES))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--chunk-duration", type=float, default=20.0, dest="chunk_s")
    ap.add_argument("--min-chunk", type=float, default=5.0, dest="min_s")
    ap.add_argument("--max-chunk", type=float, default=None, dest="max_s")
    ap.add_argument("--sampler", default="uniform",
                    help="comma-separated; any may carry a question after a "
                         "colon, e.g. `yolo:overview`")
    ap.add_argument("--per-second", type=float, default=1.0)
    ap.add_argument("--every-frames", type=int, default=None, dest="every_n")
    ap.add_argument("--min-interval", type=float, default=0.0)
    ap.add_argument("--max-per-chunk", type=int, default=None)
    ap.add_argument("--vocabulary", default=None)
    ap.add_argument("--no-frame-store", action="store_true")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=27.0,
                    dest="scene_threshold",
                    help="scene evidence: content difference opening a scene")
    ap.add_argument("--sampler-threshold", type=float, default=None,
                    help="change samplers: per-sampler default if unset")
    ap.add_argument("--silence", type=float, default=0.65, dest="silence_s")
    ap.add_argument("--transcriber", default=audio.driver.DEFAULT_TRANSCRIBER)
    ap.add_argument("--diarizer", default=audio.driver.DEFAULT_DIARIZER)
    ap.add_argument("--audio-model", default=None)
    ap.add_argument("--language", default=None)
    ap.add_argument("--describer", default=describe.driver.DEFAULT_DESCRIBER)
    ap.add_argument("--describe-model", default=None)
    ap.add_argument("--describe-limit", type=int, default=None,
                    help="stop after N describer calls. Costs money")
    ap.add_argument("--embedder", default=embed.DEFAULT_EMBEDDER)
    ap.add_argument("--embed-model", default=None)
    ap.add_argument("--tier", default="free", choices=("free", "local", "llm"))
    ap.add_argument("--until", default=None, choices=COMPONENTS,
                    help="stop after this component")
    ap.add_argument("--sink", default="file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    options = Options(
        source=args.source, video_id=args.video_id, policy=args.policy,
        use_video=not args.no_video, use_audio=not args.no_audio,
        chunk_s=args.chunk_s, min_s=args.min_s, max_s=args.max_s,
        stride=args.stride, scene_threshold=args.scene_threshold,
        sampler_threshold=args.sampler_threshold,
        silence_s=args.silence_s,
        sampler=args.sampler, per_second=args.per_second, every_n=args.every_n,
        min_interval_s=args.min_interval, max_per_chunk=args.max_per_chunk,
        vocabulary=args.vocabulary, frame_store=not args.no_frame_store,
        describer=args.describer, describe_model=args.describe_model,
        describe_limit=args.describe_limit, embedder=args.embedder,
        embed_model=args.embed_model, tier=args.tier, until=args.until,
        transcriber=args.transcriber, diarizer=args.diarizer,
        audio_model=args.audio_model, language=args.language, sink=args.sink)

    problems = validate(options)
    if problems:
        for problem in problems:
            print(f"error: {problem}")
        return 2

    def report(component: str, produced: Produced) -> None:
        if not args.json:
            wrote = ", ".join(produced.artifacts) or "-"
            print(f"  {component:<22} -> {wrote}")

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
    print()
    print(f"{run.video_id} -> {paths.home(run.video_id)}")
    print(f"  artifacts: {', '.join(paths.present(run.video_id))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
