"""Rebuild a frame store from its manifest, and prove the two agree.

The store is a cache; the manifest is the record. If that is true then a lost,
pruned or corrupted store must be reconstructible from the manifest plus the
source video, and this is the test of it.

Two paths, because the manifest describes the store's contents two different
ways depending on its scope:

  ``sampled``    every frame is listed explicitly, with its PTS. Recovery is
                 random access: seek to each one. Cheap when few frames are
                 wanted, and the only option if the decimator config is gone.

  ``decimated``  the store holds every frame the samplers were offered, which
                 the manifest does *not* list -- it only records the samplers'
                 picks. But it does record ``decimator.per_second``, and
                 decimation is a pure function of media time, so the set is
                 re-derivable by replaying it. That is a sequential pass, and
                 for a whole store it is much cheaper than seeking anyway.

Anything the manifest cannot express is a gap in the manifest, so a mismatch
here is a design bug rather than a test failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ver2.ingest.source import Decimator, FrameFetcher, UnusableSource, probe, read_frames
from ver2.ingest.store import FrameStore


def targets_from(manifest: dict) -> list[tuple[int, Optional[int]]]:
    """Every (index, pts) the manifest names, deduplicated across samplers.

    Samplers overlap heavily, so the same frame is typically listed several
    times; the store is keyed by index and holds one copy.
    """
    seen: dict[int, Optional[int]] = {}
    for chunk in manifest["chunks"]:
        for block in chunk["samplers"].values():
            for f in block["frames"]:
                seen.setdefault(f["index"], f.get("pts"))
    return sorted(seen.items())


def rebuild_sampled(manifest: dict, store: FrameStore, limit: Optional[int] = None) -> dict:
    """Random access: seek to each frame the manifest lists."""
    info = probe(manifest["source"]["uri"])
    targets = targets_from(manifest)
    if limit:
        targets = targets[:limit]

    written = missing = 0
    started = time.perf_counter()
    with FrameFetcher(info) as fetcher:
        for index, pts in targets:
            image = fetcher.fetch(pts=pts, index=index)
            if image is None:
                missing += 1
                continue
            store.write(index, image, overwrite=True)
            written += 1
        seek_stats = fetcher.stats()
    return {
        "path": "sampled/seek",
        "targets": len(targets),
        "written": written,
        "missing": missing,
        "elapsed_s": round(time.perf_counter() - started, 2),
        **seek_stats,
    }


def rebuild_decimated(manifest: dict, store: FrameStore, limit: Optional[int] = None) -> dict:
    """Sequential replay: re-derive the decimated set from the recorded config."""
    info = probe(manifest["source"]["uri"])
    per_second = manifest["config"]["decimator"]["per_second"]
    decimator = Decimator(per_second=per_second)

    written = 0
    started = time.perf_counter()
    for frame in read_frames(info):
        if decimator.accepts(frame):
            store.write(frame.index, frame.image, overwrite=True)
            written += 1
            if limit and written >= limit:
                break
        frame.release()
    return {
        "path": "decimated/replay",
        "targets": written,
        "written": written,
        "missing": 0,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }


def compare(original: Path, rebuilt: Path) -> dict:
    """Byte-compare the two directories. Same pixels + same encoder = same bytes."""
    a = {p.name: p for p in original.glob("*")} if original.exists() else {}
    b = {p.name: p for p in rebuilt.glob("*")} if rebuilt.exists() else {}
    shared = sorted(set(a) & set(b))
    identical = [n for n in shared if a[n].read_bytes() == b[n].read_bytes()]
    differing = [n for n in shared if n not in set(identical)]
    return {
        "in_original": len(a),
        "in_rebuilt": len(b),
        "compared": len(shared),
        "identical": len(identical),
        "differing": differing[:5],
        "only_in_original": sorted(set(a) - set(b))[:5],
        "only_in_rebuilt": sorted(set(b) - set(a))[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild a frame store from its manifest and verify it matches."
    )
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out/store_recreated"),
                    help="where to rebuild (default out/store_recreated)")
    ap.add_argument("--scope", default=None, choices=["sampled", "decimated"],
                    help="override the scope recorded in the manifest")
    ap.add_argument("--limit", type=int, default=None, help="stop after N frames")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    video_id = manifest["video_id"]
    declared = manifest["config"].get("frame_store")
    scope = args.scope or (declared or {}).get("scope") or "sampled"

    print(f"  manifest    {args.manifest}")
    print(f"  video       {manifest['source']['uri']}")
    print(f"  timeline    {manifest['source']['timeline']}"
          f"   time_base {manifest['source']['time_base']}")
    print(f"  scope       {scope}"
          + ("" if declared else "   (no store declared; defaulting)"))

    # Match the original's encoding, or nothing will byte-compare.
    store = FrameStore(
        args.out, video_id,
        max_width=(declared or {}).get("max_width", 1920),
        quality=(declared or {}).get("quality", 85),
    )
    print(f"  rebuilding  -> {store.dir}")
    print()

    try:
        result = (
            rebuild_decimated(manifest, store, args.limit)
            if scope == "decimated"
            else rebuild_sampled(manifest, store, args.limit)
        )
    except UnusableSource as exc:
        print(f"  source unusable: {exc}", file=sys.stderr)
        return 1

    for k, v in result.items():
        print(f"    {k:<12} {v}")
    print(f"    {'stored_mb':<12} {round(store.bytes_written / 1024 / 1024, 2)}")

    if args.no_verify or not declared:
        return 0

    print()
    original = Path(declared["dir"])
    if not original.exists():
        print(f"  original store {original} is gone -- nothing to compare against")
        print("  (which is the situation this exists for; the rebuild above stands alone)")
        return 0

    verdict = compare(original, store.dir)
    print(f"  verifying against {original}")
    for k, v in verdict.items():
        print(f"    {k:<18} {v}")
    ok = verdict["compared"] > 0 and verdict["identical"] == verdict["compared"]
    print()
    print(f"  {'PASS' if ok else 'FAIL'}: "
          f"{verdict['identical']}/{verdict['compared']} frames byte-identical")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
