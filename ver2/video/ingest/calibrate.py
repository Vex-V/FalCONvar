"""Measure what a threshold would do on this video, and where it must not go.

A change sampler is supposed to decide *for itself* how many frames a video
needs -- a still car park should yield one frame a chunk, a busy shop many.
Solving for a fixed keep rate destroys exactly that: force 15% and you have
built a worse UniformSampler, which picks the same number of frames without
running a model at all.

So this does not choose the threshold. It reports two things a human choosing
one cannot get by eye, and enforces the one bound that is not a judgement call.

**The cost of a threshold, without running the video.** Descriptors are cached
once per sampled frame, so the sampler's real decision logic can be replayed at
any threshold for microseconds. That turns "what would 0.94 cost me?" from a
35-second pipeline run into a lookup.

**The noise floor, which is a hard ceiling.** Two frames 1/fps apart in a
motionless scene are as close to "nothing happened" as this camera and codec
can produce. They are still not identical: a single PNG looped for 60 s and
H.264-encoded decodes to 60 *different* frames, differing by 0.005/255, which
CLIP scores at 0.99999 rather than 1.0. A threshold above that level samples
encoder quantization -- measured, a threshold of 1.000 kept 100% of a frozen
video. The floor is video-specific (codec, bitrate, sensor) so it has to be
measured, and it is the one place a threshold is simply wrong rather than
merely expensive.
"""

from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from .samplers import Sampler
from .source import Decimator, FrameFetcher, SourceInfo, probe

# How far below the measured noise floor a threshold must stay. Small, because
# the floor is already the most generous "nothing happened" reading.
NOISE_MARGIN = 0.002
# If ordinary frames are this close to the noise floor, the video is not
# changing enough for any threshold to separate content from quantization.
STATIC_GAP = 0.005


@dataclass
class Window:
    """One contiguous run of decimated frames, with the model's output cached."""

    start_ts: float
    stamps: list[float] = field(default_factory=list)
    descriptors: list[Any] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.descriptors)


@dataclass
class Report:
    sampler_id: str
    threshold: float                 # what the sampler is currently set to
    noise_floor: float               # similarity when nothing happened
    safe_ceiling: float              # noise_floor - margin
    is_static: bool
    frames_examined: int
    windows: int
    scores: list[float]              # decimated-frame similarities
    adjacent: list[float]            # adjacent-frame similarities (the floor)
    curve: list[tuple[float, float]] # (threshold, keep rate)

    @property
    def safe(self) -> bool:
        return self.threshold <= self.safe_ceiling

    @property
    def recommended(self) -> float:
        """The current threshold, lowered only if it reaches into noise."""
        return min(self.threshold, self.safe_ceiling)

    def rate_at(self, threshold: float) -> Optional[float]:
        best = None
        for t, r in self.curve:
            if best is None or abs(t - threshold) < abs(best[0] - threshold):
                best = (t, r)
        return best[1] if best else None

    def as_dict(self) -> dict:
        s = sorted(self.scores)
        pct = lambda p: round(s[min(len(s) - 1, int(len(s) * p / 100))], 4) if s else None
        return {
            "sampler": self.sampler_id,
            "threshold": round(self.threshold, 4),
            "recommended": round(self.recommended, 4),
            "safe": self.safe,
            "noise_floor": round(self.noise_floor, 6),
            "safe_ceiling": round(self.safe_ceiling, 6),
            "is_static": self.is_static,
            "predicted_rate": self.rate_at(self.recommended),
            "frames_examined": self.frames_examined,
            "windows": self.windows,
            "score_median": round(st.median(s), 4) if s else None,
            "score_percentiles": {f"p{p}": pct(p) for p in (5, 25, 50, 75, 95)} if s else {},
        }


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #

def window_starts(info: SourceInfo, windows: int, span: float,
                  chunk_duration_s: Optional[float]) -> list[float]:
    """Where to sample, spread across the whole timeline.

    Spread rather than front-loaded because a video's opening is often a title
    card or an empty room. On homogeneous CCTV it makes no difference (measured
    0.002); on an animated explainer it moved the answer by 0.062, and you
    cannot tell which you have in advance.
    """
    duration = (info.frame_count / info.fps) if (info.frame_count and info.fps) else None
    if duration is None or duration <= span:
        return [0.0]
    # Windows must not overlap or the middle of the video is counted twice.
    windows = max(1, min(windows, int(duration // span)))
    usable = duration - span
    starts = [round(usable * i / max(windows - 1, 1), 3) for i in range(windows)]
    if chunk_duration_s:
        # A window stands in for a chunk during replay, so it should start
        # where a chunk starts -- otherwise the guaranteed first-of-chunk frame
        # lands in places production never puts it.
        starts = sorted({round(int(t / chunk_duration_s) * chunk_duration_s, 3)
                         for t in starts})
    return starts


def collect(
    info: SourceInfo,
    sampler: Sampler,
    windows: int = 8,
    per_second: float = 1.0,
    chunk_duration_s: float = 20.0,
    noise_pairs: int = 8,
    progress: Optional[Callable[[str], None]] = None,
) -> tuple[list[Window], list[float]]:
    """Cache descriptors for sampled windows, and measure the noise floor.

    Both come from the same seeks: at each window start, the first two *source*
    frames are compared to each other for the noise floor, then the window is
    decimated as usual for the score distribution.
    """
    window_frames = max(2, int(round(chunk_duration_s * per_second)))
    span = window_frames / per_second
    starts = window_starts(info, windows, span, chunk_duration_s)

    out: list[Window] = []
    adjacent: list[float] = []

    with FrameFetcher(info) as fetcher:
        for n, start in enumerate(starts):
            if progress:
                progress(f"window {n + 1}/{len(starts)} @ {start:.0f}s")

            # Noise floor: two frames 1/fps apart. In a scene with motion these
            # differ by motion too, which is why the floor is taken from the
            # *most similar* pairs rather than the average -- those are the
            # moments when nothing happened.
            if len(adjacent) < noise_pairs:
                pair = list(fetcher.stream_from(start, 2))
                if len(pair) == 2:
                    a = sampler.compare(sampler.describe(pair[1]), sampler.describe(pair[0]))
                    if a is not None:
                        adjacent.append(a)
                for f in pair:
                    f.release()

            window = Window(start_ts=start)
            decimator = Decimator(per_second=per_second)
            budget = int(window_frames * (info.fps / per_second) * 1.2) + 10
            for frame in fetcher.stream_from(start, budget):
                if decimator.accepts(frame):
                    window.descriptors.append(sampler.describe(frame))
                    window.stamps.append(frame.media_ts)
                    if len(window.descriptors) >= window_frames:
                        frame.release()
                        break
                frame.release()
            if len(window) >= 2:
                out.append(window)
    return out, adjacent


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #

def replay(
    windows: Sequence[Window],
    sampler: Sampler,
    threshold: float,
    min_interval_s: float = 0.0,
) -> tuple[float, list[float]]:
    """Re-run the sampler's decision over cached descriptors at one threshold.

    Each window stands in for a chunk, so the reference resets between them.
    Replaying rather than cutting a fixed distribution at a percentile is not
    optional: the threshold decides which frames become references, and the
    reference decides every later score, so each threshold produces a
    *different* distribution.
    """
    kept = total = 0
    scores: list[float] = []
    for window in windows:
        reference = None
        last_kept_ts: Optional[float] = None
        for ts, descriptor in zip(window.stamps, window.descriptors):
            total += 1
            if reference is None:
                kept += 1
                reference = descriptor
                last_kept_ts = ts
                continue
            if last_kept_ts is not None and ts - last_kept_ts < min_interval_s:
                continue
            score = sampler.compare(descriptor, reference)
            if score is None:
                continue
            scores.append(score)
            if score < threshold:
                kept += 1
                reference = descriptor
                last_kept_ts = ts
    return (kept / total if total else 0.0), scores


def build_curve(windows, sampler, min_interval_s: float,
                points: Sequence[float]) -> list[tuple[float, float]]:
    return [(round(t, 5), round(replay(windows, sampler, t, min_interval_s)[0], 4))
            for t in points]


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #

def analyse(
    uri: str,
    sampler: Sampler,
    windows: int = 8,
    per_second: float = 1.0,
    chunk_duration_s: float = 20.0,
    min_interval_s: Optional[float] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Report:
    """Measure this video's noise floor and the cost of a range of thresholds."""
    info = probe(uri)
    collected, adjacent = collect(
        info, sampler, windows, per_second, chunk_duration_s, progress=progress
    )
    if not collected:
        raise RuntimeError(f"{uri}: collected no usable windows")

    mi = sampler.min_interval_s if min_interval_s is None else min_interval_s
    threshold = getattr(sampler, "threshold", 0.0)

    # The floor is the *most similar* an adjacent pair got: the moment nothing
    # happened. Averaging would fold in real motion and set the bar too low.
    noise_floor = max(adjacent) if adjacent else 1.0
    safe_ceiling = noise_floor - NOISE_MARGIN

    _, scores = replay(collected, sampler, 1.01, mi)
    # Ordinary one-second change, compared against the nothing-happened case.
    typical = st.median(scores) if scores else 1.0
    is_static = (noise_floor - typical) < STATIC_GAP

    grid = [0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
            0.995, 0.999, 0.9999]
    grid = sorted({*grid, threshold, round(safe_ceiling, 5)})
    curve = build_curve(collected, sampler, mi, [t for t in grid if 0.0 <= t <= 1.0])

    return Report(
        sampler_id=sampler.sampler_id,
        threshold=threshold,
        noise_floor=noise_floor,
        safe_ceiling=safe_ceiling,
        is_static=is_static,
        frames_examined=sum(len(w) for w in collected),
        windows=len(collected),
        scores=scores,
        adjacent=adjacent,
        curve=curve,
    )


def render(report: Report) -> str:
    d = report.as_dict()
    lines = [
        f"  sampler        {report.sampler_id}   threshold {report.threshold:g}",
        f"  examined       {report.frames_examined} frames over {report.windows} windows",
        "",
        f"  noise floor    {report.noise_floor:.6f}   "
        f"(most similar adjacent pair -- the 'nothing happened' reading)",
        f"  safe ceiling   {report.safe_ceiling:.6f}   "
        f"(a threshold above this samples codec noise)",
        f"  typical change {d['score_median']}   (median one-second similarity)",
        "",
        "  what each threshold would keep:",
    ]
    for t, r in report.curve:
        mark = ""
        if abs(t - report.threshold) < 1e-9:
            mark = "  <- current"
        elif abs(t - report.safe_ceiling) < 1e-5:
            mark = "  <- safe ceiling"
        over = " (ABOVE THE FLOOR: samples noise)" if t > report.safe_ceiling + 1e-9 else ""
        lines.append(f"    {t:<9.5g} -> {r:>6.1%}{mark}{over}")
    lines.append("")
    if report.is_static:
        lines.append("  VERDICT: this video barely changes -- ordinary frames are as similar")
        lines.append("           as motionless ones. No threshold can separate content from")
        lines.append("           noise here; the per-chunk guarantee is the right output.")
    elif not report.safe:
        lines.append(f"  VERDICT: threshold {report.threshold:g} is ABOVE the noise floor. It will")
        lines.append(f"           sample encoder noise. Lower it to at most {report.recommended:.4f}.")
    else:
        rate = report.rate_at(report.threshold)
        lines.append(f"  VERDICT: threshold {report.threshold:g} is safe "
                     f"(floor is {report.noise_floor:.6f}).")
        lines.append(f"           It keeps about {rate:.1%} of decimated frames. If that costs")
        lines.append(f"           too much, use min_interval_s / max_per_chunk rather than")
        lines.append(f"           moving the threshold -- they keep the most-changed frames.")
    return "\n".join(lines)


def main() -> int:
    import argparse
    import sys
    from pathlib import Path

    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ver2.ingest import samplers as samplers_mod

    ap = argparse.ArgumentParser(description="Report what a sampler's threshold does here.")
    ap.add_argument("video")
    ap.add_argument("--sampler", default="clip")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--chunk-duration", type=float, default=20.0)
    ap.add_argument("--min-interval", type=float, default=0.0)
    ap.add_argument("--per-second", type=float, default=1.0)
    args = ap.parse_args()

    kwargs = {"min_interval_s": args.min_interval}
    if args.threshold is not None:
        kwargs["threshold"] = args.threshold
    sampler = samplers_mod.build(args.sampler, **kwargs)

    rep = analyse(args.video, sampler, windows=args.windows,
                  per_second=args.per_second, chunk_duration_s=args.chunk_duration)
    print(f"\n  {args.video}")
    print(render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
