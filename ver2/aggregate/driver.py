"""Command line for the aggregate stage: a video's documents in, structure out.

Argument parsing and terminal output only; `reader.aggregate` does the work and
the API calls it directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):                       # allow running as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ver2 import db
from ver2 import aggregate as aggregate_mod
from ver2.aggregate.output import (AggregateDocuments, MultiAggregateSink)
from ver2.aggregate.reader import aggregate, context_for
from ver2.embed import defaults as embed_defaults


def report(result, ctx, names, out_dir: Path, sinks: list[str]) -> None:
    print()
    print(f"  sources      {', '.join(ctx.sources) or 'none'}")
    print(f"  chunks       {len(ctx.chunks)}   {ctx.duration:.1f}s")
    print(f"  fingerprint  {ctx.fingerprint()}")
    print()
    for name in names:
        if name in result.current:
            print(f"    {name:<12} current           inputs unchanged since it was built")
        elif name in result.produced:
            record = result.produced[name]
            print(f"    {name:<12} ok       {record['elapsed_s']:>6.2f}s  "
                  f"[{record['tier']}]")
        elif name in result.empty:
            print(f"    {name:<12} empty             does not apply to this video")
        elif name in result.skipped:
            print(f"    {name:<12} skipped           {result.skipped[name]}")
        elif name in result.failed:
            print(f"    {name:<12} FAILED            {result.failed[name]}")
    where = [str(out_dir) if s == "file" else "supabase: video_aggregates"
             for s in sinks]
    print(f"\n  elapsed      {result.elapsed_s:.2f}s")
    print("aggregates -> " + "\n               ".join(where))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build video-level structure from a video's descriptions "
                    "and transcript.")
    ap.add_argument("video_id", help="a video already processed into out/")
    ap.add_argument("--aggregator", default=None,
                    help="comma-separated; default is everything at --tier")
    ap.add_argument("--tier", default="free", choices=aggregate_mod.TIERS,
                    help="the most expensive tier to run (default free: no "
                         "model, no network, no cost)")
    ap.add_argument("--out-root", type=Path, default=Path("out"))
    ap.add_argument("--sink", default="file",
                    help="comma-separated: file, supabase (default file)")
    ap.add_argument("--embedder", default=None,
                    help="only novelty needs one, to read the index it wrote")
    ap.add_argument("--index", default=None,
                    help="comma-separated: qdrant, pgvector")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even where the inputs have not changed")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    sinks = [s.strip() for s in args.sink.split(",") if s.strip()]
    if any(s not in ("file", "supabase") for s in sinks) or not sinks:
        ap.error("--sink: expected file and/or supabase")

    db.load_env()
    ctx = context_for(args.video_id, args.out_root)
    if not ctx.chunks:
        print(f"{args.video_id}: nothing to aggregate -- no descriptions or "
              "transcript under " + str(args.out_root / args.video_id),
              file=sys.stderr)
        return 1

    names = ([n.strip() for n in args.aggregator.split(",") if n.strip()]
             if args.aggregator else aggregate_mod.by_tier(args.tier))

    # Only novelty reads the index, so building one is deferred to a run that
    # actually asks for it -- a free pass should not need a key.
    if "novelty" in names:
        try:
            from ver2.embed import embedders as embedders_mod
            from ver2.embed import index as index_mod

            ctx.embedder = embedders_mod.build(
                args.embedder or embed_defaults.embedder())
            ctx.index = index_mod.build(
                [n.strip() for n in (args.index or embed_defaults.index()).split(",")])
        except Exception as exc:                        # noqa: BLE001
            print(f"  note: novelty needs an index and could not reach one "
                  f"({exc}); it will report nothing", file=sys.stderr)

    out_dir = Path(args.out_root) / args.video_id / "aggregates"
    built = [AggregateDocuments(out_dir) if s == "file" else _supabase()
             for s in sinks]
    sink = built[0] if len(built) == 1 else MultiAggregateSink(*built)

    print(f"{args.video_id}   tier={args.tier}", flush=True)
    documents = AggregateDocuments(out_dir)
    result = aggregate(ctx, names, sink=sink, force=args.force,
                       existing=documents.existing())

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        report(result, ctx, names, out_dir, sinks)
    return 0 if not result.failed else 1


def _supabase():
    from ver2.aggregate.output import SupabaseAggregates

    return SupabaseAggregates()


if __name__ == "__main__":
    raise SystemExit(main())
