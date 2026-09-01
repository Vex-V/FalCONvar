"""Turn a Supabase video_id back into a description document. Standalone.

The third file in the recovery kit and the same rules as the other two: it
imports nothing from the rest of the project, and nothing outside the standard
library -- ``urllib`` is enough to call PostgREST.

    python -m ver2.recovery.supabase_description --list
    python -m ver2.recovery.supabase_description test1
    python -m ver2.recovery.supabase_description test1 --out descriptions.json

**Two queries, because a description row does not know where its chunk sits.**
``descriptions`` holds one row per ``(chunk_id, sampler)`` -- the text, the
frames it covers, the model that produced it. What it does not hold is the
chunk's ``start_ts`` and ``end_ts``, which belong to the manifest and are
deliberately not copied into the description rows, since duplicating them is
how two copies start disagreeing. So the manifest is fetched too, and the
document is assembled from both.

**What cannot be recovered is the run's statistics.** ``frames_requested``,
``cache_hits``, how long loading took -- those describe the run that produced
the descriptions rather than the descriptions themselves, and nothing stores
them. The rebuilt document carries an empty ``stats`` and says so. Every field
that says something about the *content* comes back exactly.

The document shape is spelled out here rather than imported, which is the same
trade ``recreate.py`` makes with encoding: a little duplication is the price of
being able to hand someone these files and nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

TIMEOUT_S = 30
DESCRIPTION_VERSION = 1


def _env() -> None:
    """Load a .env if python-dotenv happens to be installed. Never required."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _get(url: str, key: str, path: str, **params: str) -> Any:
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


def fingerprint(manifest: dict[str, Any]) -> str:
    """The hash the describe stage stamps on every row it writes.

    Recomputed here rather than trusted, so this tool can answer a question
    neither table can on its own: do the published descriptions still describe
    the published manifest? Over the settings, excluding ``source.uri`` and
    ``config.frame_store`` -- where the video and its cache happen to live is
    not part of what was done.
    """
    source = {k: v for k, v in manifest["source"].items() if k != "uri"}
    config = {k: v for k, v in (manifest.get("config") or {}).items()
              if k != "frame_store"}
    payload = json.dumps({"source": source, "config": config}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def assemble(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The description document, from the manifest's shape and the rows' content.

    Chunks come from the manifest so the document lists what *should* be
    described as well as what has been: an undescribed pair appears with a null
    description, exactly as the writer leaves it, rather than vanishing.
    """
    by_pair = {(r["chunk_id"], r["sampler"]): r for r in rows}
    chunks = []
    for chunk in sorted(manifest["chunks"], key=lambda c: c["chunk_id"]):
        samplers: dict[str, Any] = {}
        for sampler, block in chunk["samplers"].items():
            row = by_pair.get((chunk["chunk_id"], sampler))
            samplers[sampler] = {
                "frame_count": row["frame_count"] if row else block["frame_count"],
                "frame_indexes": (row["frame_indexes"] if row
                                  else [f["index"] for f in block["frames"]]),
                "description": row["description"] if row else None,
                "structured": (row.get("structured") or {}) if row else {},
                "elapsed_s": row.get("elapsed_s") if row else None,
            }
        chunks.append({
            "chunk_id": chunk["chunk_id"],
            "start_ts": chunk["start_ts"],
            "end_ts": chunk["end_ts"],
            "processed": all(s["description"] is not None for s in samplers.values()),
            "samplers": samplers,
        })

    models = [r["model"] for r in rows if r.get("model")]
    return {
        "description_version": DESCRIPTION_VERSION,
        "video_id": manifest["video_id"],
        "complete": bool(chunks) and all(c["processed"] for c in chunks),
        "manifest_fingerprint": fingerprint(manifest),
        "source": {
            "uri": manifest["source"]["uri"],
            "video_id": manifest["video_id"],
            "manifest_version": manifest.get("manifest_version"),
        },
        "model": models[0] if models else {},
        # Not recoverable: these describe the run, and nothing stores them.
        "stats": {},
        "chunks": chunks,
    }


def fetch(video_id: str, url: str, key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _get(url, key, "rpc/export_video_manifest", p_video_id=video_id)
    if not manifest:
        raise SystemExit(
            f"no manifest for video_id {video_id!r}. Descriptions are assembled "
            "against a manifest, so one has to be published first."
        )
    rows = _get(url, key, "rpc/export_video_descriptions", p_video_id=video_id) or []
    return manifest, rows


def listing(url: str, key: str) -> list[dict[str, Any]]:
    return _get(url, key, "video_descriptions", select="video_id,chunk_id,sampler")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild a description document from the Supabase tables.")
    ap.add_argument("video_id", nargs="?", default=None,
                    help="the video to rebuild; omit with --list")
    ap.add_argument("--out", type=Path, default=None,
                    help="where to write (default <video-id>.descriptions.json)")
    ap.add_argument("--list", action="store_true",
                    help="list the videos that have descriptions, and exit")
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
            print("no descriptions visible with this key.", file=sys.stderr)
            return 1
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["video_id"]] = counts.get(row["video_id"], 0) + 1
        print(f"{'video_id':<24} described pairs")
        for video_id, n in sorted(counts.items()):
            print(f"{video_id:<24} {n}")
        return 0

    if not args.video_id:
        ap.error("a video_id is required unless --list is given")

    manifest, rows = fetch(args.video_id, url, key)
    if not rows:
        print(f"no descriptions for video_id {args.video_id!r}. "
              "Run with --list to see what has been described.", file=sys.stderr)
        return 1

    document = assemble(manifest, rows)
    out = args.out or Path(f"{args.video_id}.descriptions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2), encoding="utf-8")

    described = sum(1 for c in document["chunks"] for s in c["samplers"].values()
                    if s["description"] is not None)
    total = sum(len(c["samplers"]) for c in document["chunks"])
    print(f"{out}  --  {len(document['chunks'])} chunks, "
          f"{described}/{total} pairs described")

    # The check neither table can make alone: were these descriptions produced
    # from the manifest that is published now, or from an earlier one?
    others = {r.get("manifest_fingerprint") for r in rows}
    others.discard(document["manifest_fingerprint"])
    if others:
        print(f"  warning: {len(others)} description(s) carry a different manifest "
              f"fingerprint than the published manifest "
              f"({document['manifest_fingerprint']}). They describe frames the "
              "current manifest no longer claims -- re-describe before trusting "
              "them.", file=sys.stderr)
    if not document["complete"]:
        print("  warning: complete = false -- some (chunk, sampler) pairs have no "
              "description yet.", file=sys.stderr)
    print("  note: stats are empty by construction -- they describe the describe "
          "run, and nothing stores them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
