"""The describe component: `manifest.json` + `store/` -> `descriptions.json`."""

from __future__ import annotations

from typing import Optional, Sequence

from ..boundaries import load as load_timeline
from ..video import load as load_manifest
from ..shared import env, paths, sinks
from ..shared.documents import Descriptions, Produced
from . import base, prompts
from .backends import stub  # noqa: F401  -- self-registers
from .frames import FrameSource, StoreUnavailable

DEFAULT_DESCRIBER = "openai"


def run(video_id: str, describer: str = DEFAULT_DESCRIBER,
        model: Optional[str] = None,
        samplers: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        resume: bool = True,
        sink: str | Sequence[str] = "file") -> Produced:
    """One call per (chunk, sampler). The expensive stage."""
    env.load()
    manifest = load_manifest(video_id)
    timeline = load_timeline(video_id)

    if manifest.timeline_fingerprint != timeline.fingerprint():
        raise ValueError(
            f"{video_id}: the manifest was built on a different grid "
            f"({manifest.timeline_fingerprint} vs {timeline.fingerprint()}). "
            "Re-run ingest against the current timeline.")

    known = prompts.questions()
    unknown = sorted({q for s in manifest.config.get("samplers", [])
                      for q in prompts.questions_of(s, s.get("id", ""))
                      if q not in known})
    if unknown:
        raise ValueError(
            f"manifest names unknown question(s) {', '.join(unknown)}; "
            f"known: {', '.join(prompts.questions())}")

    existing = None
    if resume and paths.exists(video_id, "descriptions"):
        existing = load(video_id)

    built = base.build(describer, **({"model": model} if model else {}))
    from .reader import describe

    with FrameSource(video_id, manifest) as source:
        document = describe(manifest, timeline, built, source,
                            samplers, existing, limit)

    written = sinks.write(video_id, "descriptions", document.as_dict(), sink)
    return Produced(
        video_id=video_id, component="describe", backend=",".join(written),
        artifacts={"descriptions": written.get("file", "")},
        stats={**document.stats, "describer": describer,
               "model": document.model.get("model", describer)},
    )


def load(video_id: str) -> Descriptions:
    return Descriptions.from_dict(
        sinks.read_json(paths.artifact(video_id, "descriptions")))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Describe every (chunk, sampler).")
    ap.add_argument("video_id")
    ap.add_argument("--describer", default=DEFAULT_DESCRIBER,
                    choices=sorted(set(base.available()) | {"stub"}))
    ap.add_argument("--model", default=None)
    ap.add_argument("--sampler", default=None,
                    help="comma-separated subset to describe")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N calls. Costs money, so this exists")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--sink", default="file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    samplers = ([s.strip() for s in args.sampler.split(",") if s.strip()]
                if args.sampler else None)
    try:
        produced = run(args.video_id, args.describer, args.model, samplers,
                       args.limit, not args.no_resume, args.sink)
    except (KeyError, ValueError, FileNotFoundError, StoreUnavailable,
            base.DescriberUnavailable, sinks.UnknownBackend) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    s = produced.stats
    print(f"{produced.video_id}   {s['describer']} ({s['model']})")
    print(f"  described    {s['described']}")
    print(f"  skipped      {s['skipped']}   (already current)")
    print(f"  chunks       {s['chunks']}")
    print(f"  elapsed      {s['elapsed_s']:.2f}s")
    print(f"\ndescriptions -> {produced.artifacts['descriptions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
