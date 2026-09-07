"""The aggregate component: finished documents -> `aggregates/<name>.json`.

Each aggregator writes its own file, so a run that dies partway leaves the
results it did produce rather than none -- the same reason tiers run cheapest
first.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..shared import paths, sinks
from ..shared.documents import Aggregate, Produced
from . import TIER_OF, available, resolve
from .base import TIERS, Context, missing, resolve_order


def context_for(video_id: str) -> Context:
    """Every document this video has. Missing ones are None, not an error."""
    from ..boundaries import load as load_timeline

    timeline = load_timeline(video_id)
    manifest = descriptions = transcript = None
    if paths.exists(video_id, "manifest"):
        from ..video import load as load_manifest
        manifest = load_manifest(video_id)
    if paths.exists(video_id, "descriptions"):
        from ..describe import load as load_descriptions
        descriptions = load_descriptions(video_id)
    if paths.exists(video_id, "transcript"):
        from ..cut import load as load_transcript
        transcript = load_transcript(video_id)
    return Context(video_id, timeline, manifest, descriptions, transcript)


def run(video_id: str, tier: str = "free",
        only: Optional[Sequence[str]] = None,
        force: bool = False,
        sink: str | Sequence[str] = "file") -> Produced:
    """Run every aggregator up to ``tier``, cheapest first."""
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}; known: {', '.join(TIERS)}")
    ceiling = TIERS.index(tier)
    names = [n for n in (only or available())
             if TIERS.index(TIER_OF[n]) <= ceiling]
    context = context_for(video_id)
    fingerprint = context.inputs_fingerprint()

    directory = paths.artifact(video_id, "aggregates")
    produced: dict[str, str] = {}
    skipped: dict[str, str] = {}
    ran: list[str] = []
    current = 0

    for name in resolve_order(names, TIER_OF):
        aggregator = resolve(name)()
        why = missing(aggregator, context, ran)
        if why is not None:
            skipped[name] = why
            continue

        path = directory / f"{name}.json"
        if not force and path.exists():
            stored = Aggregate.from_dict(sinks.read_json(path))
            if stored.inputs_fingerprint == fingerprint:
                current += 1
                ran.append(name)
                produced[name] = str(path)
                continue

        payload = aggregator.run(context)
        document = Aggregate(video_id=video_id, aggregate_id=name,
                             tier=aggregator.tier, payload=payload,
                             inputs_fingerprint=fingerprint,
                             stats={"about": aggregator.about})
        produced[name] = str(sinks.write_json(path, document.as_dict()))
        ran.append(name)

    return Produced(
        video_id=video_id, component="aggregate", backend="file",
        artifacts=produced,
        stats={"tier": tier, "ran": len(ran), "current": current,
               "written": len(ran) - current, "aggregates": ran,
               "skipped": skipped},
        skipped=sorted(skipped),
    )


def load(video_id: str, name: str) -> Aggregate:
    path = paths.artifact(video_id, "aggregates") / f"{name}.json"
    return Aggregate.from_dict(sinks.read_json(path))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Video-level structure.")
    ap.add_argument("video_id")
    ap.add_argument("--tier", default="free", choices=TIERS,
                    help="a cost ceiling; cheaper tiers still run")
    ap.add_argument("--only", default=None, help="comma-separated aggregator names")
    ap.add_argument("--force", action="store_true", help="rebuild what is current")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    only = ([n.strip() for n in args.only.split(",") if n.strip()]
            if args.only else None)
    try:
        produced = run(args.video_id, args.tier, only, args.force)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    s = produced.stats
    print(f"{produced.video_id}   tier={s['tier']}")
    print(f"  ran          {s['ran']}   ({s['written']} written, "
          f"{s['current']} already current)")
    for name in s["aggregates"]:
        print(f"    {name}")
    for name, why in s["skipped"].items():
        print(f"    {name:<12} -- skipped: {why}")
    print(f"\naggregates -> {paths.artifact(produced.video_id, 'aggregates')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
