"""The ingest component: `media.json` + `timeline.json` -> `manifest.json`, `store/`.

Two artifacts, which is why `Produced` lists what was written rather than
returning one path: a `--no-frame-store` run produces a manifest and no store,
and a caller should learn that from the result rather than by looking.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..boundaries import load as load_timeline
from ..media import load as load_media
from ..shared import paths, sinks
from ..shared.documents import Manifest, Produced
from . import samplers as samplers_mod
from .pipeline import ingest
from .reader import UnreadableSource
from .store import FrameStore


def build_samplers(specs: Sequence[str], every_n: Optional[int] = None,
                   min_interval_s: float = 0.0,
                   max_per_chunk: Optional[int] = None,
                   threshold: Optional[float] = None,
                   vocabulary: Optional[Sequence[str]] = None,
                   questions: Optional[Sequence[str]] = None
                   ) -> list[samplers_mod.Sampler]:
    """`["yolo", "uniform:text"]` -> sampler objects.

    Any name may carry a question after a colon. The two halves are
    independent: `uniform:text` reads the screen on a stride without paying OCR
    to decide *when*; `yolo:overview` keeps frames where the people changed and
    asks for prose. Unpaired, the question is the sampler's own name.

    ``questions`` is the vocabulary to validate against, passed in rather than
    imported: ingest does not depend on describe, and a sampler records the
    prompt as an opaque string. The caller that knows the question registry
    supplies it; without one, any name is accepted and validated later.
    """
    rate = {"min_interval_s": min_interval_s, "max_per_chunk": max_per_chunk}
    tuned = {} if threshold is None else {"threshold": threshold}
    stride = {} if every_n is None else {"every_n": every_n}

    built: list[samplers_mod.Sampler] = []
    for spec in [s.strip() for s in specs if s.strip()]:
        name, _, prompt = spec.partition(":")
        if prompt and questions is not None and prompt not in questions:
            raise ValueError(
                f"{spec!r}: unknown question {prompt!r}; "
                f"known: {', '.join(questions)}")
        ask = {"prompt": prompt} if prompt else {}
        if name == "uniform":
            built.append(samplers_mod.build(name, **ask, **stride, **rate))
        elif name == "objects":
            built.append(samplers_mod.build(name, vocabulary=list(vocabulary)
                                            if vocabulary else None,
                                            **ask, **tuned, **rate))
        else:
            # Thresholds are left unset unless given: the useful value differs
            # by an order of magnitude between samplers because they compare
            # different things, so each class keeps its calibrated default.
            built.append(samplers_mod.build(name, **ask, **tuned, **rate))
    return built


def run(video_id: str, sampler: str | Sequence[str] = "uniform",
        per_second: float = 1.0,
        every_n: Optional[int] = None,
        min_interval_s: float = 0.0,
        max_per_chunk: Optional[int] = None,
        threshold: Optional[float] = None,
        vocabulary: Optional[Sequence[str]] = None,
        frame_store: bool = True,
        store_scope: str = "sampled",
        prune_store: bool = False,
        sink: str | Sequence[str] = "file") -> Produced:
    """One decode pass over the picture, onto a grid decided elsewhere."""
    media = load_media(video_id)
    timeline = load_timeline(video_id)

    specs = ([s.strip() for s in sampler.split(",")]
             if isinstance(sampler, str) else list(sampler))
    built = build_samplers(specs, every_n, min_interval_s, max_per_chunk,
                           threshold, vocabulary)

    store = (FrameStore(paths.artifact(video_id, "store"))
             if frame_store else None)
    manifest = ingest(media, timeline, built, per_second, store, store_scope)

    pruned: list[int] = []
    if store is not None and prune_store:
        # After the pass, so a failure mid-run leaves the old store whole.
        named = {f["index"] for c in manifest.chunks
                 for b in c["samplers"].values() for f in b["frames"]}
        pruned = store.prune(named)

    written = sinks.write(video_id, "manifest", manifest.as_dict(), sink)
    artifacts = {"manifest": written.get("file", "")}
    if store is not None and store.written:
        artifacts["store"] = str(store.root)

    return Produced(
        video_id=video_id, component="video", backend=",".join(written),
        artifacts=artifacts,
        stats={**manifest.stats,
               "timeline_fingerprint": manifest.timeline_fingerprint,
               "manifest_fingerprint": manifest.fingerprint(),
               "samplers": manifest.sampler_ids(),
               "pruned_frames": len(pruned)},
        skipped=[] if store is not None else ["store"],
    )


def load(video_id: str) -> Manifest:
    return Manifest.from_dict(sinks.read_json(paths.artifact(video_id, "manifest")))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Decide which frames are worth describing.")
    ap.add_argument("video_id")
    ap.add_argument("--sampler", default="uniform",
                    help=f"comma-separated; known: "
                         f"{', '.join(samplers_mod.available())}. Any may carry "
                         f"a question after a colon, e.g. `yolo:overview`")
    ap.add_argument("--per-second", type=float, default=1.0,
                    help="frames kept per second of media time (default 1)")
    ap.add_argument("--every-frames", type=int, default=None, dest="every_n",
                    help="uniform: stride over the decimated stream (default 1)")
    ap.add_argument("--min-interval", type=float, default=0.0)
    ap.add_argument("--max-per-chunk", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="change samplers: per-sampler default if unset")
    ap.add_argument("--vocabulary", default=None,
                    help="objects: comma-separated class names")
    ap.add_argument("--no-frame-store", action="store_true")
    ap.add_argument("--prune-store", action="store_true",
                    help="delete stored frames this manifest does not name. "
                         "A store accumulates across runs; this is the only "
                         "irreversible thing ingest can do, so it is opt-in")
    ap.add_argument("--store-scope", default="sampled",
                    choices=("sampled", "decimated"))
    ap.add_argument("--sink", default="file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    vocab = ([v.strip() for v in args.vocabulary.split(",") if v.strip()]
             if args.vocabulary else None)
    try:
        produced = run(args.video_id, args.sampler, args.per_second, args.every_n,
                       args.min_interval, args.max_per_chunk, args.threshold,
                       vocab, not args.no_frame_store, args.store_scope,
                       args.prune_store, args.sink)
    except (KeyError, ValueError, FileNotFoundError, UnreadableSource,
            sinks.UnknownBackend) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    s = produced.stats
    manifest = load(produced.video_id)
    print(f"{produced.video_id}")
    print(f"  decimated    {s['frames_decimated']}   "
          f"over {s['chunks']} chunks ({s['chunks_with_frames']} with frames)")
    print(f"  sampled      {s['frames_sampled']}")
    print(f"  elapsed      {s['elapsed_s']:.2f}s")
    if "stored_frames" in s:
        print(f"  stored       {s['stored_frames']} frames, {s['stored_mb']:g} MB")
    if s.get("pruned_frames"):
        print(f"  pruned       {s['pruned_frames']} frames no longer named")
    print()
    for sampler_id in s["samplers"]:
        counts = [c["samplers"].get(sampler_id, {}).get("frame_count", 0)
                  for c in manifest.chunks]
        total = sum(counts)
        pct = total / max(s["frames_decimated"], 1) * 100
        print(f"  {sampler_id:<18} {total:>4} frames ({pct:.1f}% of decimated)"
              f"   per chunk {counts[:6]}"
              + (" ..." if len(counts) > 6 else ""))
    print()
    print(f"  timeline fp  {s['timeline_fingerprint']}")
    print(f"  manifest fp  {s['manifest_fingerprint']}")
    print(f"\nmanifest -> {produced.artifacts['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
