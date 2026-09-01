"""Command line for retrieval: build an index, then ask it questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):                       # allow running as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ver2 import db
from ver2.retrieve import embedders as embedders_mod
from ver2.retrieve import units as units_mod
from ver2.retrieve.index import MultiIndex
from ver2.retrieve.indexer import index_units
from ver2.retrieve.search import search

BACKENDS = ("qdrant", "pgvector")


_load_env = db.load_env


def _build_index(names: list[str], qdrant_path: Path):
    from ver2.retrieve.index import PgVectorIndex, QdrantIndex

    built = [QdrantIndex(qdrant_path) if n == "qdrant" else PgVectorIndex()
             for n in names]
    return built[0] if len(built) == 1 else MultiIndex(*built)


def _document_for(args) -> dict[str, Any]:
    """The descriptions to index: a local document, or Postgres."""
    if args.document:
        return json.loads(Path(args.document).read_text(encoding="utf-8"))
    client = db.client_from_env()
    manifest = db.fetch_manifest(client, args.video_id)
    rows = db.fetch_descriptions(client, args.video_id)
    bounds = {c["chunk_id"]: (c["start_ts"], c["end_ts"]) for c in manifest["chunks"]}
    return {"video_id": args.video_id, "rows": rows, "bounds": bounds}


def cmd_index(args) -> int:
    _load_env()
    embedder = embedders_mod.build(args.embedder, **(
        {"model": args.model} if args.model else {}))
    document = _document_for(args)
    if "rows" in document:
        units = units_mod.from_rows(document["video_id"], document["rows"],
                                    document["bounds"])
    else:
        units = units_mod.from_document(document)
    if not units:
        print("nothing to index: no descriptions found", file=sys.stderr)
        return 1

    video_id = units[0].video_id
    index = _build_index(args.index, args.qdrant_path)
    result = index_units(units, embedder, index, video_id=video_id,
                         force=args.force)

    stats = result.as_dict()
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
    return 0


def cmd_search(args) -> int:
    _load_env()
    embedder = embedders_mod.build(args.embedder, **(
        {"model": args.model} if args.model else {}))
    index = _build_index(args.index, args.qdrant_path)
    moments = search(args.query, embedder, index, video_id=args.video_id,
                     limit=args.limit, moments=args.moments,
                     sampler=args.sampler)
    if not moments:
        print("no matches. Is the index built for this embedder?", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([m.as_dict() for m in moments], indent=2))
        return 0

    scope = args.index[0] + (f", sampler={args.sampler}" if args.sampler else "")
    print(f'"{args.query}"   via {scope}\n')
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
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Index descriptions and search them.")
    sub = ap.add_subparsers(dest="command", required=True)

    for name in ("index", "search"):
        p = sub.add_parser(name)
        p.add_argument("--embedder", default="openai",
                       choices=embedders_mod.available(),
                       help="which embedder (default openai)")
        p.add_argument("--model", default=None,
                       help="model id for that embedder")
        p.add_argument("--index", default="qdrant",
                       help="comma-separated: qdrant, pgvector "
                            "(default qdrant; `qdrant,pgvector` writes both)")
        p.add_argument("--qdrant-path", type=Path, default=Path("out/qdrant"),
                       help="local Qdrant directory (default out/qdrant)")
        p.add_argument("--video-id", default=None,
                       help="restrict to one video")

    idx = sub.choices["index"]
    idx.add_argument("document", nargs="?", default=None,
                     help="descriptions json; omit to read Postgres via --video-id")
    idx.add_argument("--force", action="store_true",
                     help="re-embed everything, ignoring what is already indexed")

    qry = sub.choices["search"]
    qry.add_argument("query", help="what to look for")
    qry.add_argument("--limit", type=int, default=20,
                     help="descriptions to rank before folding into moments")
    qry.add_argument("--moments", type=int, default=5,
                     help="how many moments to return (default 5)")
    qry.add_argument("--sampler", default=None,
                     help="search only one sampler's descriptions (e.g. yolo for "
                          "people, text for what is written on screen). Note this "
                          "gives up cross-sampler agreement in the ranking")
    qry.add_argument("--json", action="store_true", help="machine-readable output")

    args = ap.parse_args()
    args.index = [n.strip() for n in args.index.split(",") if n.strip()]
    unknown = [n for n in args.index if n not in BACKENDS]
    if unknown:
        ap.error(f"--index: unknown backend {unknown}, expected {' and/or '.join(BACKENDS)}")
    if args.command == "index" and not args.document and not args.video_id:
        ap.error("give a descriptions document or --video-id")

    return cmd_index(args) if args.command == "index" else cmd_search(args)


if __name__ == "__main__":
    raise SystemExit(main())
