"""Delete every video, locally and in Supabase.

    python db/wipe.py                  # shows what it would delete, then asks
    python db/wipe.py --local          # only data/out and data/uploads
    python db/wipe.py --supabase       # only the rows
    python db/wipe.py --yes            # no question

Irreversible: descriptions, embeddings and llm aggregates cost paid calls to
rebuild. Stop the API first -- the embedded Qdrant store is locked while it runs.

Keeps the schema (no re-install), custom prompts and definitions
(`data/*.json`), `data/eval/` and `weights/`. The `prompts` and
`aggregate_definitions` tables only record what was asked, and the next run that
uses a definition writes it back, so they are emptied too.

Honours `FALCONVAR_DATA`, so it wipes whichever data root the pipeline would use.
"""
from __future__ import annotations

import argparse
import shutil
import sys

from falconvar.shared import env, paths
from falconvar.shared.storage import db

#: Children before parents, though no foreign key would stop another order.
TABLES = ("entity_mentions", "entities", "video_embeddings", "aggregates", "embeddings",
          "descriptions", "chunk_samplers", "manifests", "transcript_chunks", "transcripts",
          "cuts", "chunks", "timelines", "videos", "prompts", "aggregate_definitions")


def column(table: str) -> str:
    return "name" if table in ("prompts", "aggregate_definitions") else "video_id"


def main() -> int:
    ap = argparse.ArgumentParser(description="Delete every video, locally and in Supabase.")
    ap.add_argument("--local", action="store_true", help="only the local files")
    ap.add_argument("--supabase", action="store_true", help="only the Supabase rows")
    ap.add_argument("--yes", action="store_true", help="do not ask")
    args = ap.parse_args()
    local = args.local or not args.supabase
    remote = args.supabase or not args.local
    env.load()

    folders = [p for p in (paths.OUT_ROOT, paths.UPLOADS) if p.exists()]
    api = counts = None
    if remote:
        api = db.client()                       # the secret key: deletes need it
        counts = {}
        for table in TABLES:
            c = column(table)
            counts[table] = (api.table(table).select(c, count="exact")
                             .not_.is_(c, "null").limit(1).execute().count)

    print("Will delete:")
    if local:
        videos = sorted(p.name for p in paths.OUT_ROOT.iterdir()) if paths.OUT_ROOT.exists() else []
        print(f"  local     {', '.join(map(str, folders)) or 'nothing, already empty'}")
        if videos:
            print(f"            {', '.join(videos)}")
    if remote:
        print(f"  supabase  {sum(counts.values())} rows: "
              + ", ".join(f"{t} {n}" for t, n in counts.items() if n))
    if not args.yes and input("\nType 'delete' to continue: ").strip() != "delete":
        print("Nothing deleted.")
        return 1

    failed = False
    if local:
        for folder in folders:
            try:
                shutil.rmtree(folder)
                print(f"removed {folder}")
            except OSError as exc:
                failed = True
                print(f"could not remove {folder}: {exc}\n"
                      "  (is the API or another run still holding the Qdrant store?)",
                      file=sys.stderr)
    if remote:
        for table in TABLES:
            if counts[table]:
                # PostgREST refuses a delete with no filter; every row has this column.
                api.table(table).delete().not_.is_(column(table), "null").execute()
                print(f"emptied {table} ({counts[table]} rows)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
