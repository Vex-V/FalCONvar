"""Command line for the ingest pipeline.

Argument parsing, sampler construction from flags, and the run summary. The
pipeline itself is in pipeline.py and does not import this.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):                       # allow running as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ver2.ingest import chunker as chunker_mod
from ver2.ingest import samplers as samplers_mod
from ver2.ingest.output import FileManifestWriter, FrameStore, MultiSink
from ver2.ingest.pipeline import Result, ingest
from ver2.ingest.samplers import Sampler
from ver2.ingest.source import UnusableSource


def report(result: Result, show: int = 6) -> None:
    m = result.manifest
    src, stats, cfg = m["source"], m["stats"], m["config"]
    print(f"{src['uri']}")
    print(f"  {src['width']}x{src['height']}"
          + ("   [" + "; ".join(src["notes"]) + "]" if src["notes"] else ""))
    print(f"  fps          {src['fps']:g}"
          f"{'' if src['fps_trusted'] else ' (untrusted)'}   timeline={src['timeline']}")
    print(f"  read         {stats['frames_read']}")
    print(f"  decimated    {stats['frames_decimated']}"
          f"   ({stats['frames_read'] / max(stats['frames_decimated'], 1):.1f}:1)")

    chunker_cfg = cfg["chunker"]
    if chunker_cfg["name"] == "scene":
        print(f"  chunks       {stats['chunks']}   scene"
              f" (thr {chunker_cfg['threshold']:g},"
              f" {chunker_cfg['cuts_seen']} cuts,"
              f" {chunker_cfg['cuts_merged']} merged,"
              f" {chunker_cfg['splits_forced']} forced splits)")
        spans = [c.end_ts - c.start_ts for c in result.chunks]
        if spans:
            ordered = sorted(spans)
            print(f"  chunk span   min {ordered[0]:.1f}s  "
                  f"median {ordered[len(ordered) // 2]:.1f}s  max {ordered[-1]:.1f}s")
    else:
        print(f"  chunks       {stats['chunks']}   uniform"
              f" @ {chunker_cfg['duration_s']:g}s")

    print(f"  sampled      {stats['frames_sampled']}")
    print(f"  elapsed      {stats['elapsed_s']:.2f}s")

    per_sampler: dict[str, list[int]] = {}
    for chunk in result.chunks:
        for spec in cfg["samplers"]:
            per_sampler.setdefault(spec["id"], []).append(len(chunk.picks.get(spec["id"], [])))
    print()
    for sampler_id, counts in per_sampler.items():
        total = sum(counts)
        pct = total / max(stats["frames_decimated"], 1) * 100
        print(f"  {sampler_id:<10} {total:>4} frames ({pct:.1f}% of decimated)   per chunk {counts[:show]}"
              + (" ..." if len(counts) > show else ""))


def _build_samplers(args) -> list[Sampler]:
    rate = {"min_interval_s": args.min_interval, "max_per_chunk": args.max_per_chunk}
    tuned = {} if args.threshold is None else {"threshold": args.threshold}
    built = []
    for name in [n.strip() for n in args.sampler.split(",") if n.strip()]:
        if name == "uniform":
            built.append(samplers_mod.build(name, every_n=args.every_n, **rate))
        elif name == "clip":
            built.append(samplers_mod.build(name, mode=args.mode, **tuned, **rate))
        elif name == "objects":
            vocab = (
                [v.strip() for v in args.vocabulary.split(",") if v.strip()]
                if args.vocabulary else None
            )
            built.append(samplers_mod.build(name, vocabulary=vocab, **tuned, **rate))
        else:
            # Thresholds are left unset unless given: the useful value differs
            # by an order of magnitude between samplers (0.96 frame-level,
            # 0.83 person-level, 0.30 object-level) because they compare
            # different things, so each class keeps its calibrated default.
            built.append(samplers_mod.build(name, **tuned, **rate))
    return built


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest a video into a chunk/sampler manifest.")
    ap.add_argument("video", nargs="?", default="media/test.mp4")
    ap.add_argument("--per-second", type=float, default=1.0,
                    help="frames kept per second of media time (default 1)")
    ap.add_argument("--chunking", default="uniform",
                    choices=chunker_mod.available(), help="chunking strategy")
    ap.add_argument("--chunk-duration", type=float, default=20.0,
                    help="uniform: chunk length; scene: max chunk length (default 20)")
    ap.add_argument("--scene-threshold", type=float, default=27.0,
                    help="scene chunking: lower finds more cuts (default 27)")
    ap.add_argument("--scene-min-duration", type=float, default=5.0,
                    help="scene chunking: ignore cuts arriving sooner (default 5)")
    ap.add_argument("--sampler", default="uniform",
                    help=f"comma-separated; known: {', '.join(samplers_mod.available())}")
    ap.add_argument("--every-n", type=int, default=3,
                    help="uniform sampler stride, in decimated frames (default 3)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="change samplers: per-sampler default if unset")
    ap.add_argument("--target-rate", type=float, default=None,
                    help="measure the threshold instead of using one: solve for the "
                         "value that keeps roughly this fraction of decimated frames "
                         "(e.g. 0.15). Overrides --threshold")
    ap.add_argument("--calibration-windows", type=int, default=8,
                    help="spans sampled across the video when calibrating (default 8)")
    ap.add_argument("--vocabulary", default=None,
                    help="objects: comma-separated class names. There is no useful "
                         "default -- YOLO-World takes the class list as text, so the "
                         "vocabulary IS the configuration and must match the footage")
    ap.add_argument("--mode", default="reference",
                    help="clip: reference or consecutive")
    ap.add_argument("--min-interval", type=float, default=0.0,
                    help="minimum seconds between sampled frames")
    ap.add_argument("--max-per-chunk", type=int, default=None,
                    help="hard cap on frames kept per chunk")
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--frame-store", nargs="?", type=Path, const=True,
                    default=None,
                    help="write sampled pixels here, keyed by frame index. Bare "
                         "--frame-store uses out/<video-id>/store/; a path is "
                         "used exactly as given")
    ap.add_argument("--store-scope", default="sampled", choices=["sampled", "decimated"],
                    help="sampled stores only what a sampler kept -- all the VLM needs "
                         "(default). decimated also stores rejected frames, for "
                         "inspecting what a sampler passed over; it is not a retuning "
                         "cache, since JPEG artifacts (1.6/255) exceed the perturbation "
                         "that already shifts detection samplers by 12-15%%")
    ap.add_argument("--store-width", type=int, default=1920,
                    help="max width written to the store (default 1920)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="manifest path (default out/manifests/<video-id>.json)")
    ap.add_argument("--sink", default="file",
                    help="comma-separated manifest destinations: file, supabase "
                         "(default file; `file,supabase` writes both). The first "
                         "is primary and its failures stop the run; the rest are "
                         "best-effort. supabase needs SUPABASE_URL and "
                         "SUPABASE_SECRET_KEY")
    args = ap.parse_args()

    # Everything a video produces lives under one directory named after it:
    # out/<video-id>/{manifest.json, store/, descriptions.json}. Grouping by
    # video rather than by artifact type means one video's whole output is one
    # thing to inspect, copy or delete, and it keeps growing cleanly as later
    # stages add artifacts of their own.
    video_id = args.video_id or Path(args.video).stem
    home = Path("out") / video_id
    if args.out is None:
        args.out = home / "manifest.json"
    if args.frame_store is True:
        args.frame_store = home / "store"

    names = [n.strip() for n in args.sink.split(",") if n.strip()]
    unknown = [n for n in names if n not in ("file", "supabase")]
    if unknown or not names:
        ap.error(f"--sink: unknown destination {unknown or [args.sink]}, "
                 "expected file and/or supabase")
    # Each sink is constructed from its own destination settings and nothing
    # else; the pipeline tells all of them the same thing about the run. Built
    # before the decode starts, so an unreachable database fails in a second
    # rather than after minutes of inference.
    built = []
    for n in names:
        if n == "file":
            built.append(FileManifestWriter(args.out))
        else:
            # Imported here so a file-only run never pays for supabase-py, and
            # a missing install fails only the run that asked for it.
            from ver2.ingest.output import SupabaseManifestWriter

            # A CLI convenience only: the sink itself reads the environment
            # and says so, which is what a library should do.
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
            built.append(SupabaseManifestWriter())
    sink = built[0] if len(built) == 1 else MultiSink(*built)

    options: dict[str, Any] = {}
    if args.chunking == "scene":
        options["threshold"] = args.scene_threshold
        options["min_duration_s"] = args.scene_min_duration

    built = _build_samplers(args)
    calibrations = []
    if args.target_rate is not None:
        from ver2.ingest.calibrate import calibrate as run_calibration

        for sampler in built:
            if not hasattr(sampler, "threshold"):
                continue                      # positional samplers have no threshold
            c = run_calibration(
                args.video, sampler,
                target_rate=args.target_rate,
                windows=args.calibration_windows,
                per_second=args.per_second,
                min_interval_s=args.min_interval,
                chunk_duration_s=args.chunk_duration,
            )
            sampler.threshold = c.threshold
            calibrations.append(c.as_dict())
            print(f"  calibrated {sampler.sampler_id}: threshold {c.threshold:.3f} "
                  f"-> ~{c.achieved_rate:.1%} (target {args.target_rate:.1%}, "
                  f"{c.frames_examined} frames over {c.windows} windows)")
        print()

    try:
        result = ingest(
            args.video,
            per_second=args.per_second,
            chunking=args.chunking,
            chunking_options=options,
            chunk_duration_s=args.chunk_duration,
            sampler_specs=built,
            video_id=args.video_id,
            sink=sink,
            store=(
                FrameStore(args.frame_store, max_width=args.store_width)
                if args.frame_store else None
            ),
            store_scope=args.store_scope,
        )
    except UnusableSource as exc:
        print(exc, file=sys.stderr)
        return 1

    report(result)
    where = [str(args.out) if n == "file" else
             f"supabase: videos/chunks where video_id = {video_id}"
             for n in names]
    print("\nmanifest -> " + ("\n             ").join(where))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
