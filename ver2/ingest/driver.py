"""Runs the ingest pipeline and writes a manifest.

    probe -> read -> chunker.observe -> decimate -> chunk -> sampler(s)

The manifest is the handoff to the VLM stage: for each chunk, for each
sampler, which frames that sampler chose. It carries positions rather than
pixels, so the describer decides for itself when to read them back, and it is
rewritten as each chunk closes rather than at the end.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

if __package__ in (None, ""):                       # allow running as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ver2.ingest import chunker as chunker_mod
from ver2.ingest import samplers as samplers_mod
from ver2.ingest.chunker import Chunker
from ver2.ingest.manifest import ManifestWriter
from ver2.ingest.samplers import Sampler
from ver2.ingest.source import Decimator, Frame, UnusableSource, probe, read_frames
from ver2.ingest.store import FrameStore

FrameHook = Callable[[Frame, int, str], None]       # frame, chunk_id, sampler id
ChunkHook = Callable[["Chunk"], None]


@dataclass
class Chunk:
    """One window of media time, and what each sampler kept from it."""

    chunk_id: int
    start_ts: float
    end_ts: float
    decimated: int = 0
    # media_ts of the last frame that landed here. The only end a still-open
    # chunk has, since the chunker cannot know where it stops until it does.
    last_ts: float = 0.0
    # sampler id -> list of {index, media_ts, chunk_local_index, score?}
    picks: dict[str, list[dict]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "start_ts": round(self.start_ts, 3),
            "end_ts": round(self.end_ts, 3),
            "decimated_frames": self.decimated,
            "samplers": {
                sampler_id: {"frame_count": len(frames), "frames": frames}
                for sampler_id, frames in self.picks.items()
            },
        }


@dataclass
class Result:
    manifest: dict
    chunks: list[Chunk]
    frames_read: int
    frames_decimated: int
    frames_sampled: int
    elapsed_s: float


def ingest(
    uri: str,
    per_second: float = 1.0,
    chunking: str = "uniform",
    chunking_options: Optional[dict[str, Any]] = None,
    chunk_duration_s: float = 20.0,
    sampler_specs: Sequence[Sampler] = (),
    video_id: Optional[str] = None,
    out: Optional[str | Path] = None,
    store: Optional[FrameStore] = None,
    store_scope: str = "sampled",
    on_sampled: Optional[FrameHook] = None,
    on_chunk: Optional[ChunkHook] = None,
) -> Result:
    """Run one decode pass, feeding every sampler from it.

    One pass rather than one per sampler: a live source can only be consumed
    once, and re-decoding a file per sampler multiplies the most expensive
    stage for no gain.
    """
    info = probe(uri)
    video_id = video_id or Path(uri).stem
    decimator = Decimator(per_second=per_second)

    options = dict(chunking_options or {})
    if chunking == "scene":
        # Scene detection needs the source rate to turn frame numbers into
        # media time, and the duration cap doubles as the chunk length.
        options.setdefault("max_duration_s", chunk_duration_s)
        options.setdefault("fps", info.fps if info.fps_trusted else 30.0)
    else:
        options.setdefault("duration_s", chunk_duration_s)
    chunker: Chunker = chunker_mod.build(chunking, **options)

    samplers = list(sampler_specs) or [samplers_mod.build("uniform")]
    # The manifest keys frames by sampler id, so a collision would silently
    # drop one sampler's results into another's.
    ids = [s.sampler_id for s in samplers]
    if len(set(ids)) != len(ids):
        raise ValueError(f"sampler ids must be unique, got {ids}")

    writer: Optional[ManifestWriter] = None
    if out is not None:
        writer = ManifestWriter(
            out,
            video_id=video_id,
            source=info.as_dict(),
            config={
                "decimator": decimator.config(),
                "chunker": chunker.config(),
                "samplers": [s.config() for s in samplers],
                "frame_store": (
                    {**store.config(), "scope": store_scope} if store else None
                ),
            },
        )

    chunks: list[Chunk] = []
    current: Optional[Chunk] = None
    chunk_local_index = 0
    read = decimated = sampled = 0
    started = time.perf_counter()

    def stats() -> dict:
        return {
            "frames_read": read,
            "frames_decimated": decimated,
            "frames_sampled": sampled,
            "chunks": len(chunks),
            "elapsed_s": round(time.perf_counter() - started, 3),
            **({"stored_frames": store.written,
                "stored_mb": round(store.bytes_written / 1024 / 1024, 2)} if store else {}),
        }

    def close(chunk: Chunk) -> None:
        # The chunker only knows a chunk's end once the chunk is over, which
        # for scene cuts is not the same as when it was opened. The last
        # chunk has no next boundary at all, so its own last frame is the
        # only end it has -- and on a live source that stays true.
        _, end = chunker.bounds_of(chunk.chunk_id)
        chunk.end_ts = end if end is not None else chunk.last_ts
        chunks.append(chunk)
        if on_chunk is not None:
            on_chunk(chunk)
        if writer is not None:
            writer.chunk_closed(chunk.as_dict(), stats())

    for frame in read_frames(info):
        read += 1

        # Native rate, before decimation and before any release: a scene cut
        # is indistinguishable from ordinary motion once decimated to 1 fps.
        chunker.observe(frame)

        if decimator.accepts(frame):
            decimated += 1
            chunk_id = chunker.chunk_id_of(frame.media_ts)

            if current is None or chunk_id != current.chunk_id:
                if current is not None:
                    close(current)
                start, end = chunker.bounds_of(chunk_id)
                current = Chunk(chunk_id, start, end if end is not None else start)
                # Samplers forget everything at a boundary, so a chunk's
                # sampling never depends on the chunk before it.
                for sampler in samplers:
                    sampler.reset(chunk_id)
                chunk_local_index = 0

            current.decimated += 1
            current.last_ts = frame.media_ts
            # "decimated" keeps every frame the samplers were offered, so a
            # threshold can be retuned later without decoding the video again.
            if store is not None and store_scope == "decimated":
                store.write(frame.index, frame.image)
            for sampler in samplers:
                if not sampler.accepts(frame, chunk_local_index):
                    continue
                sampled += 1
                record = {
                    "index": frame.index,
                    "media_ts": round(frame.media_ts, 3),
                    "chunk_local_index": chunk_local_index,
                }
                # The address a fetcher can use. Seconds are a lossy rendering
                # of this; at a 1/1200000 timebase a rounded float lands on
                # the wrong frame.
                if frame.pts is not None:
                    record["pts"] = frame.pts
                if store is not None and store_scope == "sampled":
                    store.write(frame.index, frame.image)
                score = sampler.last_score()
                if score is not None:
                    record["score"] = round(score, 4)
                current.picks.setdefault(sampler.sampler_id, []).append(record)
                if on_sampled is not None:
                    on_sampled(frame, current.chunk_id, sampler.sampler_id)

            chunk_local_index += 1

        # A describer that wants the whole window at once would hold sampled
        # frames until the chunk closes. Nothing does yet, so pixels go back
        # immediately and peak memory stays at one frame.
        frame.release()

    if current is not None:
        close(current)

    # The last chunk ends where the video does, not where the boundary grid
    # would put it and not at its last decimated frame. Only a file knows this;
    # a live source keeps the last-frame answer set in close().
    if chunks and info.frame_count and info.fps_trusted:
        duration = info.frame_count / info.fps
        if duration > chunks[-1].start_ts:
            chunks[-1].end_ts = duration

    elapsed = time.perf_counter() - started
    final = stats()
    if writer is not None:
        # Re-emit chunks so a late end_ts correction and the final chunker
        # counters (cuts seen, splits forced) land in the written document.
        writer.chunks = [c.as_dict() for c in chunks]
        writer.config["chunker"] = chunker.config()
        document = writer.finish(final)
    else:
        document = {
            "video_id": video_id,
            "complete": True,
            "source": info.as_dict(),
            "config": {
                "decimator": decimator.config(),
                "chunker": chunker.config(),
                "samplers": [s.config() for s in samplers],
                "frame_store": (
                    {**store.config(), "scope": store_scope} if store else None
                ),
            },
            "stats": final,
            "chunks": [c.as_dict() for c in chunks],
        }
    return Result(document, chunks, read, decimated, sampled, elapsed)


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
    ap.add_argument("--frame-store", type=Path, default=None,
                    help="write sampled pixels here, keyed by frame index")
    ap.add_argument("--store-scope", default="sampled", choices=["sampled", "decimated"],
                    help="sampled stores only what a sampler kept -- all the VLM needs "
                         "(default). decimated also stores rejected frames, for "
                         "inspecting what a sampler passed over; it is not a retuning "
                         "cache, since JPEG artifacts (1.6/255) exceed the perturbation "
                         "that already shifts detection samplers by 12-15%%")
    ap.add_argument("--store-width", type=int, default=1920,
                    help="max width written to the store (default 1920)")
    ap.add_argument("-o", "--out", type=Path, default=Path("out/ingest.json"))
    args = ap.parse_args()

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
            out=args.out,
            store=(
                FrameStore(args.frame_store, args.video_id or Path(args.video).stem,
                           max_width=args.store_width)
                if args.frame_store else None
            ),
            store_scope=args.store_scope,
        )
    except UnusableSource as exc:
        print(exc, file=sys.stderr)
        return 1

    report(result)
    print(f"\nmanifest -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
