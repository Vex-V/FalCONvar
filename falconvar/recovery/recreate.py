"""Rebuild a frame store from a manifest and the source video. Standalone.

This file imports nothing from the rest of the project. Hand someone the
manifest, the video and this script and they can reproduce the frame store
byte for byte, with only ``av``, ``opencv-python`` and ``numpy`` installed.

That independence is the point, not a convenience. The manifest claims to be
the authoritative record -- that every stored pixel is reconstructible from it
plus the source. A recovery tool that imported the pipeline could quietly rely
on a default living in the pipeline's code rather than in the manifest, and
the claim would go untested. Duplicating a little seeking and encoding here is
the price of proving the manifest carries everything.

    python recreate.py manifest.json --out rebuilt/
    python recreate.py manifest.json --out rebuilt/ --verify original/

Two rebuild paths, because the manifest describes the store's contents two
different ways depending on its scope:

  ``sampled``    every frame is listed explicitly with its PTS. Recovery is
                 random access: seek to each one.
  ``decimated``  the store holds every frame the samplers were offered, which
                 the manifest does not list -- it records only the samplers'
                 picks. But it does record ``decimator.per_second``, and
                 decimation is a pure function of media time, so the set is
                 re-derivable by replaying it over a sequential read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import av
import cv2
import numpy as np

# OpenCV applies container rotation on decode; PyAV does not, so it is applied
# here. The manifest records the angle the pipeline used.
ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
MAX_SEEK_ATTEMPTS = 5


# --------------------------------------------------------------------------- #
# reading frames back out of the video
# --------------------------------------------------------------------------- #

class Fetcher:
    """Pulls frames out of a video by PTS, verifying each seek landed.

    Seeking is approximate by design: FFmpeg gets you to a keyframe at or
    before a timestamp, and reaching an exact frame is built on top. Whether
    "at or before" is honoured depends on the container's index -- MP4 has an
    exact sample table, MPEG-TS has none and estimates from bitrate. So this
    checks that the first decoded frame is not already past the target, backs
    off and retries if it is, and falls back to a scan from the start. Without
    the check, MPEG-TS returns the wrong frame every time.
    """

    def __init__(self, uri: str, rotation: float = 0.0) -> None:
        self.container = av.open(uri)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"       # without this, decode is 2x slower
        self.time_base = self.stream.time_base
        self.start = self.stream.start_time or 0
        self.second = int(1 / self.time_base) if self.time_base else 1
        self.rotate = ROTATIONS.get(int(rotation))
        self.seeks = self.retries = self.scans = 0

    def close(self) -> None:
        self.container.close()

    def __enter__(self) -> "Fetcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _image(self, frame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        return cv2.rotate(img, self.rotate) if self.rotate is not None else img

    def by_pts(self, target: int) -> Optional[np.ndarray]:
        offset = 0
        for _ in range(MAX_SEEK_ATTEMPTS):
            self.container.seek(max(self.start, target - offset), stream=self.stream,
                                backward=True, any_frame=False)
            self.seeks += 1
            overshot, first = False, True
            for frame in self.container.decode(video=0):
                if frame.pts is None:
                    continue
                if first:
                    first = False
                    if frame.pts > target:
                        overshot = True        # decoding forward can never reach it
                        break
                if frame.pts == target:
                    return self._image(frame)
                if frame.pts > target:
                    break                      # target absent from the stream
            if not overshot:
                break
            offset = self.second if offset == 0 else offset * 4
            self.retries += 1
        return self.scan_for(pts=target)

    def scan_for(self, pts: Optional[int] = None,
                 index: Optional[int] = None) -> Optional[np.ndarray]:
        """Last resort: decode from the beginning. Always correct, never fast."""
        self.scans += 1
        self.container.seek(self.start, stream=self.stream, backward=True)
        i = 0
        for frame in self.container.decode(video=0):
            if pts is not None and frame.pts == pts:
                return self._image(frame)
            if index is not None and i == index:
                return self._image(frame)
            i += 1
        return None

    def sequential(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Every frame in order, as (index, media_ts, image)."""
        self.container.seek(self.start, stream=self.stream, backward=True)
        for i, frame in enumerate(self.container.decode(video=0)):
            ts = float(frame.pts * self.time_base) if frame.pts is not None else 0.0
            yield i, ts, self._image(frame)


# --------------------------------------------------------------------------- #
# writing them back into a store
# --------------------------------------------------------------------------- #

class StoreWriter:
    """Encodes frames exactly as the pipeline did, from the manifest's settings.

    Byte-identical output depends on matching resize interpolation, JPEG
    quality and filename format. All three come from the manifest rather than
    from a default here, so a store written with different settings still
    rebuilds correctly.
    """

    def __init__(self, root: str | Path, video_id: str, max_width: Optional[int],
                 quality: int, suffix: str = ".jpg") -> None:
        self.dir = Path(root) / video_id
        self.max_width = max_width
        self.quality = quality
        self.suffix = suffix
        self.written = 0
        self.bytes_written = 0

    def write(self, index: int, image: np.ndarray) -> None:
        if self.max_width and image.shape[1] > self.max_width:
            h = int(round(image.shape[0] * self.max_width / image.shape[1]))
            image = cv2.resize(image, (self.max_width, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(self.suffix, image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise RuntimeError(f"failed to encode frame {index}")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / f"{index:07d}{self.suffix}").write_bytes(buf)
        self.written += 1
        self.bytes_written += len(buf)


# --------------------------------------------------------------------------- #
# rebuilding
# --------------------------------------------------------------------------- #

def verify_source(manifest: dict, uri: str) -> list[str]:
    """Check the video actually matches the one the manifest was built from.

    Nothing else catches this. A manifest addresses frames by PTS and index; a
    different video answers those addresses perfectly well and produces a store
    full of the wrong pictures with no error. Since the recipient of a manifest
    supplies their own copy of the source, the mismatch has to be detected here.

    Compared against what the manifest already records -- no extra fields, so
    this works on manifests written before the check existed.
    """
    src = manifest["source"]
    problems: list[str] = []
    with av.open(uri) as container:
        stream = container.streams.video[0]
        rate = stream.guessed_rate or stream.average_rate
        actual = {
            "fps": float(rate) if rate else 0.0,
            "time_base": str(stream.time_base) if stream.time_base else None,
            "frame_count": stream.frames or None,
        }
        for frame in container.decode(video=0):
            h, w = frame.height, frame.width
            if int(src.get("rotation") or 0) in (90, 270):
                w, h = h, w
            actual["width"], actual["height"] = w, h
            break

    for key in ("fps", "time_base", "width", "height"):
        want, got = src.get(key), actual.get(key)
        if want in (None, 0) or got in (None, 0):
            continue
        if key == "fps":
            if abs(float(want) - float(got)) > 0.01:
                problems.append(f"fps {got:g} != {want:g}")
        elif want != got:
            problems.append(f"{key} {got} != {want}")
    # frame_count is absent for containers with no index, so only compare it
    # when both sides know.
    if src.get("frame_count") and actual.get("frame_count"):
        if src["frame_count"] != actual["frame_count"]:
            problems.append(f"frame_count {actual['frame_count']} != {src['frame_count']}")
    return problems


def targets_from(manifest: dict) -> list[tuple[int, Optional[int]]]:
    """Every (index, pts) the manifest names, deduplicated across samplers.

    Samplers overlap heavily -- a frame chosen by four of them is listed four
    times -- and the store holds one copy per index.
    """
    seen: dict[int, Optional[int]] = {}
    for chunk in manifest["chunks"]:
        for block in chunk["samplers"].values():
            for f in block["frames"]:
                seen.setdefault(f["index"], f.get("pts"))
    return sorted(seen.items())


def rebuild_sampled(manifest: dict, store: StoreWriter,
                    limit: Optional[int] = None) -> dict:
    """Random access: seek to each frame the manifest lists."""
    src = manifest["source"]
    targets = targets_from(manifest)[:limit] if limit else targets_from(manifest)
    missing = 0
    started = time.perf_counter()
    with Fetcher(src["uri"], src.get("rotation", 0.0)) as fetcher:
        for index, pts in targets:
            image = (fetcher.by_pts(pts) if pts is not None and src["timeline"] == "pts"
                     else fetcher.scan_for(index=index))
            if image is None:
                missing += 1
                continue
            store.write(index, image)
        stats = {"seeks": fetcher.seeks, "retries": fetcher.retries, "scans": fetcher.scans}
    return {"path": "sampled/seek", "targets": len(targets), "written": store.written,
            "missing": missing, "elapsed_s": round(time.perf_counter() - started, 2), **stats}


def rebuild_decimated(manifest: dict, store: StoreWriter,
                      limit: Optional[int] = None) -> dict:
    """Sequential replay: re-derive the decimated set from the recorded config.

    Decimation keeps the first frame in each 1/per_second slice of media time,
    so replaying it over a sequential read reproduces exactly the frames the
    samplers were offered -- including the ones the manifest never listed.
    """
    src = manifest["source"]
    per_second = manifest["config"]["decimator"]["per_second"]
    last_bucket: Optional[int] = None
    started = time.perf_counter()
    with Fetcher(src["uri"], src.get("rotation", 0.0)) as fetcher:
        for index, media_ts, image in fetcher.sequential():
            bucket = int(media_ts * per_second)
            if last_bucket is None or bucket > last_bucket:
                last_bucket = bucket
                store.write(index, image)
                if limit and store.written >= limit:
                    break
    return {"path": "decimated/replay", "targets": store.written, "written": store.written,
            "missing": 0, "elapsed_s": round(time.perf_counter() - started, 2)}


def compare(original: Path, rebuilt: Path) -> dict:
    """Byte-compare two stores. Same pixels through the same encoder = same bytes."""
    a = {p.name: p for p in original.glob("*")} if original.exists() else {}
    b = {p.name: p for p in rebuilt.glob("*")} if rebuilt.exists() else {}
    shared = sorted(set(a) & set(b))
    identical = [n for n in shared if a[n].read_bytes() == b[n].read_bytes()]
    return {"in_original": len(a), "in_rebuilt": len(b), "compared": len(shared),
            "identical": len(identical),
            "differing": [n for n in shared if n not in set(identical)][:5],
            "only_in_original": sorted(set(a) - set(b))[:5],
            "only_in_rebuilt": sorted(set(b) - set(a))[:5]}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild a frame store from its manifest and the source video.")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, default=Path("rebuilt"))
    ap.add_argument("--video", default=None,
                    help="override the source path recorded in the manifest")
    ap.add_argument("--scope", default=None, choices=["sampled", "decimated"],
                    help="override the scope recorded in the manifest")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify", type=Path, default=None,
                    help="an existing store to byte-compare against")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the video does not match the manifest")
    args = ap.parse_args()

    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    src = manifest["source"]
    if args.video:
        src["uri"] = args.video
    declared = manifest["config"].get("frame_store") or {}
    scope = args.scope or declared.get("scope") or "sampled"

    if not Path(src["uri"]).exists():
        print(f"  source video not found: {src['uri']}\n"
              f"  pass --video to point at it", file=sys.stderr)
        return 1

    print(f"  manifest   {args.manifest}")
    print(f"  video      {src['uri']}")
    print(f"  timeline   {src['timeline']}   time_base {src['time_base']}"
          + (f"   rotation {src['rotation']:g}" if src.get("rotation") else ""))
    print(f"  scope      {scope}" + ("" if declared else "   (none declared; defaulting)"))

    problems = verify_source(manifest, src["uri"])
    if problems:
        print()
        print("  SOURCE MISMATCH -- this video is not the one the manifest describes:",
              file=sys.stderr)
        for p_ in problems:
            print(f"    {p_}", file=sys.stderr)
        if not args.force:
            print("  refusing to rebuild; pass --force to override", file=sys.stderr)
            return 1
        print("  --force given, continuing anyway", file=sys.stderr)
    else:
        print("  source     matches the manifest")

    store = StoreWriter(args.out, manifest["video_id"],
                        max_width=declared.get("max_width", 1920),
                        quality=declared.get("quality", 85),
                        suffix="." + declared.get("format", "jpg"))
    print(f"  rebuilding -> {store.dir}")
    print()

    result = (rebuild_decimated(manifest, store, args.limit) if scope == "decimated"
              else rebuild_sampled(manifest, store, args.limit))
    for k, v in result.items():
        print(f"    {k:<12} {v}")
    print(f"    {'stored_mb':<12} {round(store.bytes_written / 1024 / 1024, 2)}")

    original = args.verify or (Path(declared["dir"]) if declared.get("dir") else None)
    if original is None or not original.exists():
        print()
        print("  nothing to verify against -- the rebuild above stands alone")
        return 0

    verdict = compare(original, store.dir)
    print()
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
