"""The boundaries component: evidence -> `cuts.json`, then -> `timeline.json`.

Two callables, because the component has two jobs and the orchestrator needs to
see both:

    evidence(video_id, policy)   run the precursor this policy needs, if any
    run(video_id, policy)        decide the grid and write it

They stay separate rather than `run` calling `evidence` itself, so the
dependency chain is visible in `workflow.py` rather than hidden one level down.
The whole argument for the grid being its own component is that ordering falls
out of what a policy depends on -- burying the expensive pass inside the cheap
one would put that back out of sight.

`run()` is the callable and `main()` a shim over it, never the reverse:
`falconvar` had to undo the opposite arrangement, where a server would have had
to import an argparse module to reach the work behind it.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..media import load as load_media
from ..shared import paths
from ..shared.storage import sinks
from ..shared.contracts.documents import Cuts, Produced, RawTranscript, Timeline
from . import scenes, speech
from .grid import POLICIES, build


def load_cuts(video_id: str) -> Cuts:
    return Cuts.from_dict(sinks.read_json(paths.artifact(video_id, "cuts")))


def load(video_id: str) -> Timeline:
    """Read back the grid. Every consumer starts here."""
    return Timeline.from_dict(sinks.read_json(paths.artifact(video_id, "timeline")))


def evidence(video_id: str, policy: str,
             stride: int = scenes.DEFAULT_STRIDE,
             threshold: float = scenes.DEFAULT_THRESHOLD,
             detect_width: int = scenes.DETECT_WIDTH,
             silence_s: float = speech.DEFAULT_SILENCE_S,
             sink: str | Sequence[str] = "file") -> Optional[Produced]:
    """Run whichever precursor this policy needs. None for `uniform`.

    `uniform` needs nothing -- it is arithmetic over a duration `media.json`
    already recorded -- so this returns None rather than doing work to produce
    an empty cut list.
    """
    if policy not in POLICIES:
        raise KeyError(f"unknown policy {policy!r}; known: {', '.join(POLICIES)}")
    needs = POLICIES[policy]
    if needs is None:
        return None

    if needs == "video":
        cuts = scenes.detect(load_media(video_id), stride, threshold, detect_width)
    else:
        raw = RawTranscript.from_dict(
            sinks.read_json(paths.artifact(video_id, "raw_transcript")))
        cuts = speech.detect(raw, policy, silence_s)

    written = sinks.write(video_id, "cuts", cuts.as_dict(), sink)
    return Produced(
        video_id=video_id, component=f"boundaries.{needs}",
        backend=",".join(written),
        artifacts={"cuts": written.get("file", "")},
        stats={**cuts.stats, **cuts.params},
    )


def retune(video_id: str, threshold: float,
           sink: str | Sequence[str] = "file") -> Produced:
    """A different threshold over the cached scores. Runs no model.

    This is what caching the score series buys, and it is exact rather than an
    estimate: `detect` and `rethreshold` both threshold the same array, so a
    retune and a re-run at the same value cannot disagree.
    """
    cuts = load_cuts(video_id)
    module = scenes if cuts.source == "video" else speech
    retuned = module.rethreshold(cuts, threshold)
    written = sinks.write(video_id, "cuts", retuned.as_dict(), sink)
    return Produced(
        video_id=video_id, component="boundaries.retune",
        backend=",".join(written),
        artifacts={"cuts": written.get("file", "")},
        stats={**retuned.stats, "threshold": threshold,
               "was": cuts.params.get("threshold", cuts.params.get("silence_s"))},
    )


def _cuts_for(video_id: str, policy: str) -> tuple[Optional[list[float]], str,
                                                   dict[str, Any]]:
    """The cuts this policy needs, read as a file.

    Reading rather than importing is what keeps `boundaries` free of an edge to
    `listen`, and lets the evidence have been produced by an earlier run, on
    another machine, or by hand.
    """
    needs = POLICIES[policy]
    if needs is None:
        return None, "grid", {}
    path = paths.artifact(video_id, "cuts")
    if not path.exists():
        producer = "scenes" if needs == "video" else "speech"
        raise FileNotFoundError(
            f"policy {policy!r} needs cuts from boundaries.{producer}; "
            f"{path} does not exist -- run `evidence` first")
    cuts = load_cuts(video_id)
    expected = "content" if policy == "scene" else policy
    if cuts.detector != expected:
        raise ValueError(
            f"{path} holds {cuts.detector!r} cuts, but the policy is "
            f"{policy!r}. Re-run the evidence pass for this policy.")
    return list(cuts.cuts), needs, cuts.params


def run(video_id: str, policy: str = "uniform",
        chunk_s: float = 20.0, min_s: float = 5.0,
        max_s: Optional[float] = None,
        sink: str | Sequence[str] = "file") -> Produced:
    """Decide the grid and write it. The one place boundaries are chosen."""
    if policy not in POLICIES:
        raise KeyError(f"unknown policy {policy!r}; known: {', '.join(POLICIES)}")

    # `enforce` merges up to `min_s` and *then* splits at `max_s`, so the split
    # runs last and wins. Asking for a floor above the ceiling therefore
    # produced chunks below the floor and reported success -- measured, `--min-
    # chunk 30` against a `max_s` defaulting to `--chunk-duration` 20 gave a
    # grid whose shortest span was 18.07s. Refused rather than resolved,
    # because there is no reading of "at least 30, at most 20" to honour.
    ceiling = chunk_s if max_s is None else max_s
    if ceiling and min_s > ceiling:
        raise ValueError(
            f"--min-chunk {min_s:g} is larger than the ceiling {ceiling:g} "
            f"({'--max-chunk' if max_s is not None else '--chunk-duration, '
               'which --max-chunk defaults to'}). Raise the ceiling or lower "
            "the floor.")

    media = load_media(video_id)
    if media.duration_s is None:
        raise ValueError(f"{video_id}: the container reports no duration, so no "
                         "grid can be derived from it")
    if POLICIES[policy] == "video" and not media.has_video:
        raise ValueError(f"policy {policy!r} is found in the picture, and "
                         f"{video_id} has no video stream")
    if POLICIES[policy] == "audio" and not media.has_audio:
        raise ValueError(f"policy {policy!r} is derived from the soundtrack, and "
                         f"{video_id} has no audio stream")

    cuts, derived_from, params = _cuts_for(video_id, policy)
    timeline = build(video_id, policy, media.duration_s, cuts=cuts,
                     derived_from=derived_from, chunk_s=chunk_s,
                     min_s=min_s, max_s=max_s, params=params)

    written = sinks.write(video_id, "timeline", timeline.as_dict(), sink)
    spans = [e - s for s, e in timeline.spans]
    return Produced(
        video_id=video_id, component="boundaries", backend=",".join(written),
        artifacts={"timeline": written.get("file", "")},
        stats={"policy": policy, "derived_from": derived_from,
               "chunks": len(timeline), "fingerprint": timeline.fingerprint(),
               "duration_s": round(timeline.duration_s, 3),
               "shortest_s": round(min(spans), 3) if spans else 0.0,
               "longest_s": round(max(spans), 3) if spans else 0.0,
               "cuts_offered": 0 if cuts is None else len(cuts)},
    )


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Find boundary evidence, and decide the chunk grid.")
    ap.add_argument("video_id")
    ap.add_argument("--policy", default="uniform", choices=sorted(POLICIES))
    ap.add_argument("--evidence", action="store_true",
                    help="run only the precursor this policy needs")
    ap.add_argument("--chunk-duration", type=float, default=20.0, dest="chunk_s",
                    help="uniform: the length; otherwise the maximum (default 20)")
    ap.add_argument("--min-chunk", type=float, default=5.0, dest="min_s")
    ap.add_argument("--max-chunk", type=float, default=None, dest="max_s",
                    help="defaults to --chunk-duration")
    ap.add_argument("--stride", type=int, default=scenes.DEFAULT_STRIDE,
                    help="scene: score every Nth frame (default 5). Higher is "
                         "cheaper and needs a HIGHER --threshold")
    ap.add_argument("--threshold", type=float, default=scenes.DEFAULT_THRESHOLD,
                    help="scene: content difference that opens a scene")
    ap.add_argument("--silence", type=float, default=speech.DEFAULT_SILENCE_S,
                    help="vad: a gap this long or longer is a boundary")
    ap.add_argument("--retune", type=float, default=None, metavar="VALUE",
                    help="re-threshold the cached scores. Runs no model")
    ap.add_argument("--calibrate", action="store_true",
                    help="what every threshold would cost here. Reads the "
                         "cached scores; runs nothing")
    ap.add_argument("--sink", default="file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        if args.calibrate:
            cuts = load_cuts(args.video_id)
            rows = scenes.sweep(cuts, [5, 10, 15, 20, 27, 35, 45, 60, 80]
                                if cuts.source == "video"
                                else [0.25, 0.5, 0.65, 1.0, 1.5, 2.0, 3.0])
            current = cuts.params.get("threshold", cuts.params.get("silence_s"))
            print(f"{args.video_id}   {cuts.detector}   "
                  f"{len(cuts.scores['values'])} scored   currently {current}")
            print(f"  {'value':>10} {'cuts':>6} {'rate':>8} {'median gap':>12}")
            for r in rows:
                gap = "--" if r["median_gap_s"] is None else f"{r['median_gap_s']:g}s"
                mark = "  <- current" if r["threshold"] == current else ""
                print(f"  {r['threshold']:>10g} {r['cuts']:>6} {r['rate']:>8.1%} "
                      f"{gap:>12}{mark}")
            print("\n  A threshold is a property of the footage, not a default. "
                  "This reports; it does not choose.")
            return 0

        if args.retune is not None:
            produced = retune(args.video_id, args.retune, args.sink)
        elif args.evidence:
            produced = evidence(args.video_id, args.policy, args.stride,
                                args.threshold, scenes.DETECT_WIDTH,
                                args.silence, args.sink)
            if produced is None:
                print(f"{args.video_id}: policy 'uniform' needs no evidence "
                      "-- it is arithmetic over the container duration")
                return 0
        else:
            produced = run(args.video_id, args.policy, args.chunk_s, args.min_s,
                           args.max_s, args.sink)
    except (KeyError, ValueError, FileNotFoundError, scenes.NoPicture,
            sinks.UnknownBackend) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    s = produced.stats
    if produced.component == "boundaries":
        print(f"{produced.video_id}   policy={s['policy']}  "
              f"derived_from={s['derived_from']}")
        print(f"  chunks       {s['chunks']}   over {s['duration_s']:g}s")
        print(f"  span         min {s['shortest_s']:g}s   max {s['longest_s']:g}s")
        if s["cuts_offered"]:
            print(f"  cuts offered {s['cuts_offered']}")
        print(f"  fingerprint  {s['fingerprint']}")
        print(f"\ntimeline -> {produced.artifacts['timeline']}")
    else:
        print(f"{produced.video_id}   {produced.component}")
        for key in ("frames_read", "frames_scored", "speech_spans", "elapsed_s",
                    "ms_per_frame", "threshold", "silence_s", "stride",
                    "cuts_found", "was"):
            if key in s and s[key] is not None:
                print(f"  {key:<14} {s[key]}")
        print(f"\ncuts -> {produced.artifacts['cuts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
