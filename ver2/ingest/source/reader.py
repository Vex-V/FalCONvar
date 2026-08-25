"""Sequential decode: one Frame per picture.

Sequential is not a simplification. Frames reference each other, so producing
the frame at second 47 means decoding forward from its keyframe regardless --
seeking to sample would cost *more* decode work, not less.

This is a generator, so decoding happens between iterations of the caller's
loop rather than before it. Exactly one frame is in flight; the alternative
would be 4485 frames x ~6 MB for a five-minute 1080p file.
"""

from __future__ import annotations

from typing import Iterator

import av
import cv2

from .probe import UnusableSource
from .types import Frame, SourceInfo

# OpenCV auto-applies container rotation; PyAV does not, so the reader does.
ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def read_frames(info: SourceInfo) -> Iterator[Frame]:
    """Yield stamped frames from an already-probed source.

    The container is reopened rather than reused: probe() consumed frames from
    its own decoder, and a decoder is a cursor with no rewind. Reopening a file
    is cheap -- and is one of the things a live source will not be able to do.
    """
    try:
        container = av.open(info.uri)
    except Exception as exc:
        raise UnusableSource(f"could not reopen {info.uri}: {exc}") from None

    rotate = ROTATIONS.get(int(info.rotation))
    index = 0
    try:
        stream = container.streams.video[0]
        # Mandatory, not an optimisation: without it decoding runs at
        # 7.15 ms/frame against 3.97 with it, which is slower than OpenCV.
        stream.thread_type = "AUTO"
        time_base = stream.time_base

        for av_frame in container.decode(video=0):
            image = av_frame.to_ndarray(format="bgr24")
            if rotate is not None:
                image = cv2.rotate(image, rotate)

            # A lookup, not a decision -- probe() already established whether
            # this source has usable timestamps. pts * time_base is exact
            # rational arithmetic; the float is only for downstream comparison.
            if info.timeline == "pts" and av_frame.pts is not None:
                pts = av_frame.pts
                media_ts = float(pts * time_base)
            else:
                pts = None
                media_ts = index / info.fps

            yield Frame(
                index=index,
                media_ts=media_ts,
                pts=pts,
                image=image,
                is_keyframe=bool(av_frame.key_frame),
            )
            index += 1
    finally:
        container.close()
