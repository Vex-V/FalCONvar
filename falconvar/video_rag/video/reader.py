"""Sequential decode, converting only the frames something asked for.

Decoding a frame costs ~0.4 ms; converting it to a full-resolution BGR array
costs ~6 ms. Ingest needs pixels only for frames that survive decimation, and
the decimator answers from `media_ts` alone -- so its verdict is asked *before*
the conversion and most frames never become an array.

Sequential rather than seeking: frames reference each other, so producing the
frame at second 47 means decoding forward from its keyframe regardless.

A generator, so exactly one frame is in flight. `Frame.image` is borrowed and
released as soon as the frame is not needed; anything outliving the loop copies.

Rotation comes from OpenCV because PyAV 18.1 exposes the display matrix through
none of `side_data`, `rotation`, `display_matrix` or `metadata`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import av
import numpy as np

from ...shared.contracts.documents import Media
from ...shared.errors import FalconvarError

#: OpenCV auto-applies container rotation; PyAV does not, so the reader does.
#: PyAV 18.1 exposes the display matrix through none of `side_data`,
#: `rotation`, `display_matrix` or `metadata`, so OpenCV is opened once purely
#: to read the number. Checked again on 18.1 rather than assumed from
#: `falconvar`.
_ROTATIONS = {90: 0, 180: 1, 270: 2}       # cv2.ROTATE_* resolved lazily


class UnreadableSource(FalconvarError, RuntimeError):
    """The file cannot be opened, or carries no video stream."""


@dataclass
class Frame:
    """One frame the pipeline is going to look at.

    ``media_ts`` is the position on the media clock and the only clock a
    downstream decision may use. ``pts`` is the same position in the
    container's integer timebase and is the only thing that can *address* the
    frame later: seconds are a lossy rendering, and at timebases as fine as
    1/1200000 a rounded float lands on the wrong frame.

    ``index`` is a plain read counter over *every* frame, not over the kept
    ones -- it is how a store names a file and how recovery finds it again.

    ``image`` is BGR and is **borrowed**: it is released as soon as the frame
    is known not to be needed, so anything that outlives the loop must copy.
    """

    index: int
    media_ts: float
    pts: Optional[int] = None
    image: Optional[np.ndarray] = field(default=None, repr=False)
    is_keyframe: bool = False

    def release(self) -> None:
        """Drop the pixels. A method, so the one dangerous operation greps."""
        self.image = None


def rotation_of(path: str) -> float:
    """Container rotation in degrees, via OpenCV. 0.0 when unknown."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return 0.0
        return float(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0.0)
    finally:
        cap.release()


def read_frames(media: Media,
                keep: Callable[[float], bool],
                rotation: Optional[float] = None) -> Iterator[Frame]:
    """Yield only the frames ``keep`` accepts, with pixels attached.

    ``keep`` is handed a ``media_ts`` and nothing else -- deciding from the
    timestamp is what allows the decision to precede the conversion. A frame it
    declines is never turned into an array and costs the bare decode.
    """
    if not media.has_video:
        raise UnreadableSource(f"{media.path} has no video stream")

    try:
        container = av.open(media.path)
    except Exception as exc:                             # noqa: BLE001
        raise UnreadableSource(
            f"cannot open {media.path} ({type(exc).__name__})") from None

    degrees = int(rotation if rotation is not None else rotation_of(media.path))
    rotate = None
    if degrees in _ROTATIONS:
        import cv2
        rotate = (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                  cv2.ROTATE_90_COUNTERCLOCKWISE)[_ROTATIONS[degrees]]

    index = 0
    try:
        stream = container.streams.video[0]
        # Mandatory, not an optimisation: 7.15 ms/frame without it against
        # 3.97 with, which is slower than OpenCV.
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        fps = media.video.rate or 30.0

        for av_frame in container.decode(video=0):
            # No pixels touched yet. `pts * time_base` is exact rational
            # arithmetic; the float is only for comparison downstream.
            if av_frame.pts is not None and time_base is not None:
                pts = av_frame.pts
                media_ts = float(pts * time_base)
            else:
                pts = None
                media_ts = index / fps

            if keep(media_ts):
                image = av_frame.to_ndarray(format="bgr24")   # the 6.2 ms
                if rotate is not None:
                    import cv2
                    image = cv2.rotate(image, rotate)
                yield Frame(index=index, media_ts=media_ts, pts=pts,
                            image=image, is_keyframe=bool(av_frame.key_frame))
            index += 1
    finally:
        container.close()


__all__ = ["Frame", "UnreadableSource", "read_frames", "rotation_of"]
