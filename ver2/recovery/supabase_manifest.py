"""Turn a Supabase video_id back into a manifest file. Standalone.

Like ``recreate.py``, this imports nothing from the rest of the project, and
nothing outside the standard library either -- ``urllib`` is enough to call a
PostgREST endpoint. Adding ``supabase-py`` here would buy nothing and cost the
one property that matters: two small files plus the video are the whole
recovery kit, and its dependencies stay ``av``, ``opencv-python``, ``numpy``.

The split is deliberate. This tool answers "where is the manifest?", and
``recreate.py`` answers "what does the manifest describe?". Keeping them apart
means the second never grows a network dependency, and someone who already has
a manifest -- emailed, copied, produced by a local ingest -- runs recreate
directly and never touches this file.

    python -m ver2.recovery.supabase_manifest test2
    python -m ver2.recovery.supabase_manifest test2 --out manifests/test2.json
    python -m ver2.recovery.supabase_manifest --list
    python -m ver2.recovery.recreate test2.json --out rebuilt/

Reads ``SUPABASE_URL`` and ``SUPABASE_PUBLISHABLE_KEY`` from the environment,
or takes ``--url`` / ``--key``. The publishable key is the right one: it is
read-only under RLS and is meant to be handed out. The secret key would work
and must not be used -- it bypasses RLS on every table in the project.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

TIMEOUT_S = 30


def _env() -> None:
    """Load a .env if python-dotenv happens to be installed. Never required."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _get(url: str, key: str, path: str, **params: str) -> Any:
    """One GET against PostgREST, returning parsed JSON."""
    target = f"{url.rstrip('/')}/rest/v1/{path}"
    if params:
        target += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(target, headers={"apikey": key})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:400]
        raise SystemExit(
            f"{exc.code} {exc.reason} from {target}\n{body}\n\n"
            "A 401 usually means the key is wrong; an empty result with a 200 "
            "usually means row-level security has no read policy for it."
        ) from None
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach {target}: {exc.reason}") from None


def fetch(video_id: str, url: str, key: str) -> dict[str, Any]:
    """The manifest for one video, exactly as the file writer would have written it.

    ``export_manifest`` reassembles the document server-side: the ``videos``
    row supplies the header and the ``chunks`` rows are aggregated back into
    the ``chunks`` array, in ``chunk_id`` order. Nothing is reconstructed here,
    so there is no second implementation of the format to drift.
    """
    document = _get(url, key, "rpc/export_video_manifest", p_video_id=video_id)
    if not document:
        raise SystemExit(
            f"no manifest for video_id {video_id!r}.\n"
            "Run with --list to see what this project holds."
        )
    return document


def listing(url: str, key: str) -> list[dict[str, Any]]:
    return _get(url, key, "video_manifests",
                select="video_id,complete,ingested_at", order="ingested_at.desc")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fetch a manifest from Supabase and write it as JSON, "
                    "ready for recreate.py.")
    ap.add_argument("video_id", nargs="?", default=None,
                    help="the video to fetch; omit with --list")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write (default <video-id>.json)")
    ap.add_argument("--list", action="store_true",
                    help="list the video ids this project holds, and exit")
    ap.add_argument("--url", default=None, help="default $SUPABASE_URL")
    ap.add_argument("--key", default=None,
                    help="default $SUPABASE_PUBLISHABLE_KEY")
    args = ap.parse_args(argv)

    _env()
    url = args.url or os.environ.get("SUPABASE_URL")
    key = args.key or os.environ.get("SUPABASE_PUBLISHABLE_KEY")
    if not url or not key:
        print("set SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY, or pass --url/--key",
              file=sys.stderr)
        return 2

    if args.list:
        rows = listing(url, key)
        if not rows:
            print("no manifests visible with this key.", file=sys.stderr)
            return 1
        print(f"{'video_id':<24} {'complete':<9} ingested_at")
        for row in rows:
            print(f"{row['video_id']:<24} {str(row['complete']):<9} {row['ingested_at']}")
        return 0

    if not args.video_id:
        ap.error("a video_id is required unless --list is given")

    document = fetch(args.video_id, url, key)
    out = args.out or Path(f"{args.video_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # indent=2 to match the file writer's formatting. Key *order* will not
    # match: jsonb does not preserve it, so the two documents are equal as
    # data and noisy under `diff`. Compare parsed, never textually.
    out.write_text(json.dumps(document, indent=2), encoding="utf-8")

    chunks = document.get("chunks", [])
    frames = sum(len(s.get("frames", []))
                 for chunk in chunks
                 for s in chunk.get("samplers", {}).values())
    print(f"{out}  --  {len(chunks)} chunks, {frames} frame records")
    if not document.get("complete"):
        # Not an error: a run still in flight is a legitimate thing to read.
        print("  warning: complete = false. This manifest is either mid-ingest "
              "or was left partial by a failed run; anything rebuilt from it "
              "will be missing the chunks that never arrived.", file=sys.stderr)
    else:
        # Handed over as two loose files, the module path does not exist.
        how = ("python -m ver2.recovery.recreate" if __package__
               else "python recreate.py")
        print(f"  next: {how} {out} --out rebuilt/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
