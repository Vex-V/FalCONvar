"""STANDALONE: rebuild a frame store from a manifest and the video.

Hand someone this file, a `manifest.json` and the source video, and they get the
store back byte for byte with only `av`, `opencv-python` and `numpy`. Nothing
here imports the pipeline: if it did, it could lean on a default living in code
rather than in the manifest, and the manifest's claim to be authoritative would
go untested.

That is also what makes this the end-to-end oracle. Any change to encoding,
addressing or the manifest is verified by rebuilding a store and byte-comparing
it. If a change makes recreate non-identical, the change is wrong.

    python -m recovery.recreate data/out/<id>/manifest.json --out rebuilt/
    python -m recovery.recreate <manifest> --verify data/out/<id>/store

The manifest names frames by `index`, the reader's count over every frame, and
carries `pts` for each -- the exact address. Seconds are a lossy rendering of
`pts` and are not used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

import av
import cv2

ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class Mismatch(RuntimeError):
    """The video is not the one this manifest was built from."""


def wanted_frames(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """index -> record, for every frame any sampler kept.

    A frame two samplers chose appears once: the store is keyed by index, so
    it holds one copy however many questions were asked about it.
    """
    out: dict[int, dict[str, Any]] = {}
    for chunk in manifest.get("chunks", []):
        for block in chunk.get("samplers", {}).values():
            for record in block.get("frames", []):
                out.setdefault(int(record["index"]), record)
    return out


def check_source(manifest: dict[str, Any], path: Path) -> list[str]:
    """Everything about this file that disagrees with the manifest.

    Compared rather than assumed: recreating from the wrong video produces a
    complete store of wrong frames, which is indistinguishable from a correct
    one until someone looks at a picture.
    """
    source = manifest.get("source", {})
    problems: list[str] = []
    with av.open(str(path)) as container:
        stream = next(iter(container.streams.video), None)
        if stream is None:
            return [f"{path} has no video stream"]
        checks = [
            ("width", stream.codec_context.width, source.get("width")),
            ("height", stream.codec_context.height, source.get("height")),
            ("time_base", str(stream.time_base), source.get("time_base")),
            ("frames", stream.frames or None, source.get("frames")),
        ]
        rate = float(stream.guessed_rate or stream.average_rate or 0) or None
        checks.append(("rate", rate, source.get("rate")))
    for name, found, expected in checks:
        if expected is not None and found is not None and found != expected:
            problems.append(f"{name}: video has {found!r}, manifest says {expected!r}")
    return problems


def recreate(manifest_path: Path, out_dir: Path,
             video: Optional[Path] = None, force: bool = False,
             quality: int = 95) -> dict[str, Any]:
    """Decode the video and write every frame the manifest names."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source", {})
    path = Path(video) if video else Path(source.get("path", ""))
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Pass --video to point at a moved copy.")

    problems = check_source(manifest, path)
    if problems and not force:
        raise Mismatch(
            f"{path} does not match this manifest:\n  "
            + "\n  ".join(problems) + "\n--force overrides.")

    records = wanted_frames(manifest)
    if not records:
        raise ValueError(f"{manifest_path} names no frames")

    rotation = int(source.get("rotation", 0) or 0)
    rotate = ROTATIONS.get(rotation)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    index = 0
    highest = max(records)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(video=0):
            if index in records:
                image = frame.to_ndarray(format="bgr24")
                if rotate is not None:
                    image = cv2.rotate(image, rotate)
                target = out_dir / f"{index:07d}.jpg"
                cv2.imwrite(str(target), image,
                            [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                written += 1
            index += 1
            if index > highest:
                break

    return {"named": len(records), "written": written,
            "missing": sorted(set(records) - {int(p.stem)
                                              for p in out_dir.glob("*.jpg")}),
            "out": str(out_dir), "video": str(path), "problems": problems}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(rebuilt: Path, original: Path,
           names: Optional[set[str]] = None) -> dict[str, Any]:
    """Byte-compare a rebuilt store against the original. The oracle.

    ``names`` is what this manifest actually names. Without it the comparison
    is over whatever the output directory happens to contain -- and an output
    directory accumulates across runs for exactly the reason a frame store
    does, which the orphan rule below already tolerates in the other direction.
    Measured: `rebuilt/` held 206 frames from an earlier manifest, a later run
    wrote 76 into it, and the oracle reported FAIL with 130 "absent from the
    store" when every named frame was present and identical. A false FAIL is
    the worst answer this can give, because the whole point of it is that a
    real one means the change is wrong.
    """
    mine = {p.name: p for p in rebuilt.glob("*.jpg")}
    if names is not None:
        mine = {n: path for n, path in mine.items() if n in names}
    theirs = {p.name: p for p in original.glob("*.jpg")}
    shared = sorted(set(mine) & set(theirs))
    identical = [n for n in shared if digest(mine[n]) == digest(theirs[n])]
    return {
        "compared": len(shared),
        "identical": identical,
        # The only real failure: a frame both have, whose bytes differ.
        "differing": sorted(set(shared) - set(identical)),
        # A frame the manifest names that the original store lacks. Also a
        # failure, in the other direction -- the store is incomplete.
        "missing_from_original": sorted(set(mine) - set(theirs)),
        # Frames the store holds that this manifest does not name. NOT a
        # failure: a store accumulates across runs and is never pruned, so an
        # earlier ingest with a different sampler set leaves files behind.
        # Counting those as a mismatch would make the oracle cry wolf on every
        # video that has been ingested twice.
        "orphaned_in_original": sorted(set(theirs) - set(mine)),
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild a frame store from a manifest and the video.")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, default=Path("rebuilt"))
    ap.add_argument("--video", type=Path, default=None,
                    help="the source, if it has moved since the manifest")
    ap.add_argument("--verify", type=Path, default=None,
                    help="byte-compare against an existing store")
    ap.add_argument("--force", action="store_true",
                    help="recreate even if the video does not match")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args(argv)

    try:
        result = recreate(args.manifest, args.out, args.video, args.force,
                          args.quality)
    except (Mismatch, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    print(f"{result['video']}")
    print(f"  named        {result['named']}")
    print(f"  written      {result['written']}")
    if result["missing"]:
        print(f"  MISSING      {len(result['missing'])}: {result['missing'][:8]}")
    if result["problems"]:
        print(f"  forced past  {len(result['problems'])} mismatch(es)")
    print(f"  out          {result['out']}")

    if args.verify is not None:
        named = {f"{index:07d}.jpg"
                 for index in wanted_frames(json.loads(
                     args.manifest.read_text(encoding="utf-8")))}
        check = verify(args.out, args.verify, named)
        print()
        if check["differing"] or check["missing_from_original"]:
            print(f"  FAIL: {len(check['identical'])}/{check['compared']} "
                  f"identical, {len(check['differing'])} differ, "
                  f"{len(check['missing_from_original'])} absent from the store")
            return 1
        print(f"  PASS: {len(check['identical'])}/{check['compared']} "
              f"frames byte-identical")
        if check["orphaned_in_original"]:
            print(f"        ({len(check['orphaned_in_original'])} other frames "
                  f"in the store this manifest does not name -- an earlier "
                  f"run's, not a mismatch)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
