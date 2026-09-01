"""Command line for the embed stage: descriptions in, vectors out.

Flat rather than subcommanded, like `ingest.driver` and `describe.driver`: one
module, one job, one positional argument naming what it reads. Searching what
this writes is `ver2.retrieve.driver`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):                       # allow running as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ver2 import db
from ver2.embed import defaults
from ver2.embed import embedders as embedders_mod
from ver2.embed import index as index_mod
from ver2.embed import units as units_mod
from ver2.embed.indexer import index_units


def _units_for(args) -> list:
    """The descriptions to embed: a local document, or Postgres.

    A description row does not know where its chunk sits in media time, so the
    Postgres path fetches the manifest too -- the same split that makes
    `recovery.supabase_description` fetch two things.
    """
    if args.document:
        document = json.loads(Path(args.document).read_text(encoding="utf-8"))
        return units_mod.from_document(document)
    client = db.client_from_env()
    manifest = db.fetch_manifest(client, args.video_id)
    rows = db.fetch_descriptions(client, args.video_id)
    bounds = {c["chunk_id"]: (c["start_ts"], c["end_ts"]) for c in manifest["chunks"]}
    return units_mod.from_rows(args.video_id, rows, bounds)


def report(stats: dict[str, Any], video_id: str, args) -> None:
    print(f"{video_id}  ({stats['embedder']})")
    print(f"  units        {stats['units']}")
    print(f"  embedded     {stats['embedded']}"
          + (f"   ({stats['restated']} re-embedded: description changed)"
             if stats["restated"] else ""))
    if stats["skipped"]:
        print(f"  skipped      {stats['skipped']}   (already indexed, text unchanged)")
    print(f"  elapsed      {stats['elapsed_s']:.2f}s")
    print("\nindexed -> " + ", ".join(
        str(args.qdrant_path) if n == "qdrant" else "supabase: description_embeddings"
        for n in args.index))


def main() -> int:
    # Before the parser, not after it: the defaults below are resolved from
    # the environment, so a `.env` has to be in effect while they are read.
    db.load_env()
    ap = argparse.ArgumentParser(
        description="Embed descriptions and store the vectors. Defaults: "
                    + defaults.describe())
    ap.add_argument("document", nargs="?", default=None,
                    help="descriptions json; omit to read Postgres via --video-id")
    ap.add_argument("--video-id", default=None,
                    help="read the descriptions from Supabase instead of a file")
    ap.add_argument("--embedder", default=defaults.embedder(),
                    choices=embedders_mod.available(),
                    help=f"which embedder (default {defaults.embedder()}; "
                         f"${defaults.ENV_EMBEDDER})")
    ap.add_argument("--model", default=defaults.model(),
                    help=f"model id for that embedder (${defaults.ENV_MODEL})")
    ap.add_argument("--index", default=defaults.index(),
                    help=f"comma-separated: {', '.join(index_mod.BACKENDS)} "
                         f"(default {defaults.index()}; ${defaults.ENV_INDEX}). "
                         "`pgvector,qdrant` writes both, pgvector authoritative")
    ap.add_argument("--qdrant-path", type=Path,
                    default=index_mod.DEFAULT_QDRANT_PATH,
                    help="local Qdrant directory (default out/qdrant)")
    ap.add_argument("--force", action="store_true",
                    help="re-embed everything, ignoring what is already indexed")
    args = ap.parse_args()

    args.index = [n.strip() for n in args.index.split(",") if n.strip()]
    unknown = [n for n in args.index if n not in index_mod.BACKENDS]
    if unknown or not args.index:
        ap.error(f"--index: unknown backend {unknown or [args.index]}, expected "
                 + " and/or ".join(index_mod.BACKENDS))
    if not args.document and not args.video_id:
        ap.error("give a descriptions document or --video-id")

    embedder = embedders_mod.build(args.embedder,
                                   **({"model": args.model} if args.model else {}))
    units = _units_for(args)
    if not units:
        print("nothing to index: no descriptions found", file=sys.stderr)
        return 1

    video_id = units[0].video_id
    result = index_units(units, embedder, index_mod.build(args.index, args.qdrant_path),
                         video_id=video_id, force=args.force)
    report(result.as_dict(), video_id, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
