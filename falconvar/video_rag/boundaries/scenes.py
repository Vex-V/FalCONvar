"""Scene evidence: content scores, thresholded into cuts.

Its own pass over the picture, so the grid is an input to ingest rather than
something ingest derives while consuming it.

Converts lazily -- straight to the detection size through the decoder's own
scaler, never building a full-resolution frame.

The detector scores the difference between *consecutive frames it was given*,
so a stride widens the gap between them and a threshold calibrated for adjacent
frames fires far more often. Every real cut survives a stride; false positives
rise. The stride is therefore a parameter and the threshold moves with it.

`cuts.json` records the per-frame score series, not just the cuts, so a
different threshold is arithmetic over a cached array instead of another pass.
`cuts_from_scores` is that cheap half and both `detect` and `rethreshold` call
it, so a retune and a re-run cannot disagree.
"""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Any, Optional, Sequence

import av

from ..shared.contracts.documents import Cuts, Media

#: Cuts are a global property of the frame; full resolution buys nothing but
#: time. Detection runs on a downscaled copy.
DETECT_WIDTH = 320

#: PySceneDetect's default, calibrated for *adjacent* frames. At any stride
#: above 1 it is too low -- see the module docstring.
DEFAULT_THRESHOLD = 27.0

#: Frames of the decimated-for-detection stream between scored frames.
DEFAULT_STRIDE = 5


class NoPicture(RuntimeError):
    """Asked to find scene cuts in a file with no video stream."""


def _timestamps(stream) -> tuple[Optional[Fraction], float]:
    time_base = stream.time_base
    rate = stream.guessed_rate or stream.average_rate
    return time_base, float(rate) if rate else 0.0


def score_frames(media: Media, stride: int = DEFAULT_STRIDE,
                 detect_width: int = DETECT_WIDTH) -> tuple[list[tuple[float, float]],
                                                            dict[str, Any]]:
    """Decode the picture and score every ``stride``-th frame.

    Returns ``[(media_ts, score), ...]`` and the pass's own stats. This is the
    expensive half and the only part that touches the video.
    """
    if not media.has_video:
        raise NoPicture(f"{media.path} has no video stream")
    if stride < 1:
        raise ValueError("stride must be >= 1; it is a frame stride")

    from scenedetect import ContentDetector, FrameTimecode
    from scenedetect.stats_manager import StatsManager

    stats_manager = StatsManager()
    # `min_scene_len` is left at its default but its effect is irrelevant here:
    # cuts come from thresholding the score series, not from `process_frame`'s
    # return value, and the minimum chunk length is `grid.enforce`'s job. One
    # guard, in one place, meaning the same thing under every policy.
    detector = ContentDetector(threshold=DEFAULT_THRESHOLD)
    detector.stats_manager = stats_manager

    container = av.open(media.path)
    try:
        stream = container.streams.video[0]
        # Mandatory, not an optimisation: 7.15 ms/frame without it against
        # 3.97 with, which is slower than OpenCV.
        stream.thread_type = "AUTO"
        time_base, fps = _timestamps(stream)
        height = max(1, int(media.video.height * detect_width / media.video.width))

        scored: list[tuple[float, float]] = []
        read = converted = 0
        started = time.perf_counter()

        for frame in container.decode(video=0):
            if read % stride == 0:
                # Straight to the detection size. No full-resolution array is
                # ever built -- that is the 6.2 ms this pass does not pay.
                small = frame.reformat(width=detect_width, height=height,
                                       format="bgr24").to_ndarray()
                detector.process_frame(FrameTimecode(read, fps or 30.0), small)
                value = stats_manager.get_metrics(
                    read, [ContentDetector.FRAME_SCORE_KEY])[0]
                if value is not None:
                    # The first scored frame has nothing to compare against and
                    # returns None. It is not a zero-difference frame; it is a
                    # frame with no predecessor, and recording 0.0 for it would
                    # put a real value in the series that nothing measured.
                    media_ts = (float(frame.pts * time_base)
                                if frame.pts is not None and time_base
                                else read / (fps or 30.0))
                    scored.append((media_ts, float(value)))
                converted += 1
            read += 1
    finally:
        container.close()

    elapsed = time.perf_counter() - started
    return scored, {
        "frames_read": read,
        "frames_scored": converted,
        "elapsed_s": round(elapsed, 3),
        "ms_per_frame": round(elapsed / max(read, 1) * 1000, 3),
    }


def cuts_from_scores(scored: Sequence[tuple[float, float]],
                     threshold: float) -> list[float]:
    """The cheap half: which scored frames are cuts, at this threshold.

    A frame whose content differs from its predecessor by more than
    ``threshold`` opens a new scene, so the cut lands at that frame's own
    timestamp. No minimum-separation filtering here -- `grid.enforce` applies
    `min_s` under every policy, and doing it twice would mean two guards with
    the same name and different values.
    """
    return [ts for ts, value in scored if value >= threshold]


def detect(media: Media, stride: int = DEFAULT_STRIDE,
           threshold: float = DEFAULT_THRESHOLD,
           detect_width: int = DETECT_WIDTH) -> Cuts:
    """One pass over the picture, as a `Cuts` document."""
    scored, stats = score_frames(media, stride, detect_width)
    cuts = cuts_from_scores(scored, threshold)
    return Cuts(
        video_id=media.video_id,
        source="video",
        detector="content",
        params={"threshold": threshold, "stride": stride,
                "detect_width": detect_width},
        cuts=cuts,
        scores={
            "metric": "content_val",
            "stride": stride,
            # Timestamps travel with the values. A bare array would need the
            # reader to reconstruct positions from a stride and a frame rate,
            # which is arithmetic this module has already done correctly once.
            "at": [round(ts, 3) for ts, _ in scored],
            "values": [round(v, 4) for _, v in scored],
        },
        stats={**stats, "cuts_found": len(cuts)},
    )


def rethreshold(cuts: Cuts, threshold: float) -> Cuts:
    """A different threshold over the cached scores. No decoding.

    This is what the cached series is for. It agrees exactly with a re-run of
    `detect` at the same threshold and stride, because both call
    `cuts_from_scores`.
    """
    if not cuts.scores:
        raise ValueError(
            f"{cuts.video_id}: this cuts document carries no score series, so a "
            "threshold cannot be changed without re-running the pass")
    scored = list(zip(cuts.scores["at"], cuts.scores["values"]))
    return Cuts(
        video_id=cuts.video_id, source=cuts.source, detector=cuts.detector,
        params={**cuts.params, "threshold": threshold},
        cuts=cuts_from_scores(scored, threshold),
        scores=cuts.scores,
        stats={**cuts.stats, "cuts_found": len(cuts_from_scores(scored, threshold)),
               "rethresholded": True},
    )


def sweep(cuts: Cuts, thresholds: Sequence[float]) -> list[dict[str, Any]]:
    """What every threshold would cost here. Reports; does not choose.

    The same stance `falconvar`'s calibrator takes, for the same reason: the
    useful threshold is a property of the footage, and a number solved for on
    one video is wrong on the next. Measured there: one sampler at one
    threshold kept 13.4%, 18.0% and 59.2% of frames across three videos of the
    same domain.
    """
    if not cuts.scores:
        raise ValueError("no score series to sweep")
    scored = list(zip(cuts.scores["at"], cuts.scores["values"]))
    values = [v for _, v in scored]
    rows = []
    for threshold in sorted(thresholds):
        found = [ts for ts, v in scored if v >= threshold]
        gaps = [b - a for a, b in zip(found, found[1:])]
        rows.append({
            "threshold": threshold,
            "cuts": len(found),
            "rate": round(len(found) / max(len(values), 1), 4),
            "median_gap_s": round(sorted(gaps)[len(gaps) // 2], 2) if gaps else None,
        })
    return rows


__all__ = ["DETECT_WIDTH", "DEFAULT_THRESHOLD", "DEFAULT_STRIDE", "NoPicture",
           "score_frames", "cuts_from_scores", "detect", "rethreshold", "sweep"]
