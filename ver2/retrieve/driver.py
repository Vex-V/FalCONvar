"""Command line for retrieval: ask the index a question, get moments back.

Flat rather than subcommanded, like every other driver here. Building the
index this searches is `ver2.embed.driver`.

The embedder flags are not a convenience: a query embedded by a different
model than the descriptions produces a ranking that is well-formed and
meaningless, so they have to name the same embedder the index was built with.
That is what `embedder_key` is in the collection name for -- a mismatch
searches a collection that does not exist rather than returning nonsense.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):                       # allow running as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ver2 import db
from ver2.embed import defaults
from ver2.embed import embedders as embedders_mod
from ver2.embed import index as index_mod
from ver2.retrieve.search import Moment, search


def report(moments: list[Moment], args) -> None:
    scope = args.index[0] + (f", sampler={args.sampler}" if args.sampler else "")
    print(f'"{args.query}"   via {scope}')
    # Said once, because a dense-only ranking and a hybrid one look
    # identical: every hit simply lacks a `t` marker, which reads as "the
    # query had no lexical match" rather than "this index cannot have one".
    if args.index[0] != "pgvector":
        print(f"  note: {args.index[0]} is dense-only -- no lexical half in "
              "this ranking. --index pgvector for the hybrid.", file=sys.stderr)
    print()
    for position, moment in enumerate(moments, start=1):
        agree = "+".join(moment.samplers)
        print(f"{position}. {moment.video_id} chunk {moment.chunk_id}  "
              f"{moment.span}   score {moment.score:.4f}   [{agree}]")
        print(f"   frames {moment.frame_indexes[:10]}"
              + (" ..." if len(moment.frame_indexes) > 10 else ""))
        for hit in moment.hits:
            ranks = []
            if hit.vector_rank:
                ranks.append(f"v{hit.vector_rank}")
            if hit.text_rank:
                ranks.append(f"t{hit.text_rank}")
            marker = f" ({','.join(ranks)})" if ranks else ""
            print(f"   {hit.sampler}{marker}: {hit.content[:150].strip()}...")
        print()


def main() -> int:
    # Before the parser: the defaults are read from the environment, and they
    # are shared with `embed` precisely so the two cannot disagree about which
    # embedder built the index being searched.
    db.load_env()
    ap = argparse.ArgumentParser(
        description="Search the descriptions and get back moments to play. "
                    "Defaults: " + defaults.describe())
    ap.add_argument("query", help="what to look for")
    ap.add_argument("--embedder", default=defaults.embedder(),
                    choices=embedders_mod.available(),
                    help=f"which embedder (default {defaults.embedder()}; "
                         f"${defaults.ENV_EMBEDDER}) -- must be the one the "
                         "index was built with")
    ap.add_argument("--model", default=defaults.model(),
                    help=f"model id for that embedder (${defaults.ENV_MODEL})")
    ap.add_argument("--index", default=defaults.index(),
                    help=f"comma-separated: {', '.join(index_mod.BACKENDS)} "
                         f"(default {defaults.index()}; ${defaults.ENV_INDEX}). "
                         "The first named is the one that answers")
    ap.add_argument("--qdrant-path", type=Path,
                    default=index_mod.DEFAULT_QDRANT_PATH,
                    help="local Qdrant directory (default out/qdrant)")
    ap.add_argument("--video-id", default=None,
                    help="restrict to one video")
    ap.add_argument("--limit", type=int, default=20,
                    help="descriptions to rank before folding into moments")
    ap.add_argument("--moments", type=int, default=5,
                    help="how many moments to return (default 5)")
    ap.add_argument("--sampler", default=None,
                    help="search only one sampler's descriptions (e.g. yolo for "
                         "people, text for what is written on screen). Note this "
                         "gives up cross-sampler agreement in the ranking")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    args.index = [n.strip() for n in args.index.split(",") if n.strip()]
    unknown = [n for n in args.index if n not in index_mod.BACKENDS]
    if unknown or not args.index:
        ap.error(f"--index: unknown backend {unknown or [args.index]}, expected "
                 + " and/or ".join(index_mod.BACKENDS))

    embedder = embedders_mod.build(args.embedder,
                                   **({"model": args.model} if args.model else {}))
    moments = search(args.query, embedder,
                     index_mod.build(args.index, args.qdrant_path),
                     video_id=args.video_id, limit=args.limit,
                     moments=args.moments, sampler=args.sampler)
    if not moments:
        print("no matches. Is the index built for this embedder?", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([m.as_dict() for m in moments], indent=2))
        return 0
    report(moments, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
