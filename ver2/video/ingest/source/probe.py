"""Establish how a source's timeline can be read, before ingesting it.

Deciding here rather than per frame matters: a file whose timestamps are
scrambled should use the fallback from frame 0, not switch over midway and
leave the opening frames stamped by a method already known to be wrong.

Reading through PyAV removes most of the guesswork that OpenCV forced. Where
OpenCV reports a float ``POS_MSEC`` and a single ``FPS`` that containers
routinely lie about, PyAV exposes the integer PTS, the exact ``time_base``,
and ``guessed_rate`` -- which is right on files where the reported average is
not. Measured: an H.264-in-AVI reporting 600 fps has ``guessed_rate`` 15, and
a raw .h264 that OpenCV called 25 fps has ``guessed_rate`` 15, correctly.

What PyAV does *not* give is the display matrix: version 18.1 exposes it
through none of ``side_data``, ``rotation``, ``display_matrix`` or
``metadata``. So OpenCV is opened once, here, purely to read the rotation.
"""

from __future__ import annotations

from fractions import Fraction

import av
import cv2

from .types import SourceInfo

# Above this is metadata corruption, not a capture rate.
MAX_PLAUSIBLE_FPS = 240.0
PROBE_FRAMES = 12
MAX_PLAUSIBLE_FRAME_COUNT = 100_000_000


class UnusableSource(RuntimeError):
    """The source cannot be read, or offers no trustworthy timeline."""


def _rotation_of(uri: str) -> float:
    """Container rotation in degrees, via OpenCV -- PyAV 18.1 does not expose it."""
    cap = cv2.VideoCapture(uri)
    try:
        return float(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0.0) if cap.isOpened() else 0.0
    finally:
        cap.release()


def probe(uri: str, n: int = PROBE_FRAMES) -> SourceInfo:
    """Open, sample the first ``n`` frames, and decide how to read time.

    Raises UnusableSource rather than guessing: a wrong timeline silently
    produces wrong chunk boundaries, which is worse than an error.
    """
    try:
        container = av.open(uri)
    except Exception as exc:
        raise UnusableSource(
            f"cannot open {uri} -- unsupported, missing, or corrupt ({type(exc).__name__})"
        ) from None

    notes: list[str] = []
    try:
        try:
            stream = container.streams.video[0]
        except IndexError:
            raise UnusableSource(f"{uri}: no video stream") from None
        stream.thread_type = "AUTO"

        time_base: Fraction | None = stream.time_base
        rate = stream.guessed_rate or stream.average_rate
        fps = float(rate) if rate else 0.0
        count = stream.frames or 0

        stamps: list[int] = []
        shape = None
        for av_frame in container.decode(video=0):
            stamps.append(av_frame.pts)
            if shape is None:
                shape = (av_frame.height, av_frame.width)
            if len(stamps) >= n:
                break
    finally:
        container.close()

    if shape is None:
        raise UnusableSource(f"{uri}: opened but decoded no frames")

    fps_trusted = 0 < fps <= MAX_PLAUSIBLE_FPS
    if not fps_trusted:
        notes.append(f"implausible frame rate {fps:g}")

    # A raw elementary stream carries no timing at all: every pts is None.
    # This is a fact from the decoder, not a heuristic over sampled values.
    have_pts = time_base is not None and all(p is not None for p in stamps)
    # Rebuilding a container that never stored reorder information (AVI, raw
    # streams) yields correct timestamps emitted in decode order.
    monotonic = have_pts and all(b >= a for a, b in zip(stamps, stamps[1:]))

    if not have_pts:
        notes.append("container carries no presentation timestamps")
    elif not monotonic:
        notes.append("timestamps not in presentation order")

    timeline = "pts" if (have_pts and monotonic) else "derived"
    if timeline == "derived" and not fps_trusted:
        raise UnusableSource(
            f"{uri}: no usable timestamps and frame rate implausible ({fps:g}) -- "
            "no trustworthy timeline available"
        )

    frame_count = count if 0 < count < MAX_PLAUSIBLE_FRAME_COUNT else None

    rotation = _rotation_of(uri)
    height, width = shape
    if int(rotation) in (90, 270):
        # The reader rotates on the way out, so the dimensions downstream
        # sees are the transposed ones.
        width, height = height, width
    if rotation:
        notes.append(f"rotation {rotation:g} deg applied")

    return SourceInfo(
        uri=uri,
        fps=fps,
        fps_trusted=fps_trusted,
        time_base=time_base,
        width=width,
        height=height,
        frame_count=frame_count,
        timeline=timeline,
        rotation=rotation,
        notes=notes,
    )
