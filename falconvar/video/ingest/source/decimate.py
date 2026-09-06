"""Rate reduction at the head of the pipeline.

Nothing downstream sees native frame rate. A 15 fps source becomes whatever
``per_second`` asks for, which is what keeps samplers and the VLM from having
to reason about fifteen near-identical frames a second.
"""

from __future__ import annotations

from typing import Optional

from .types import Frame


class Decimator:
    """Keeps the first frame seen in each slice of media time.

    ``per_second`` is how many frames survive per second of video:

        1  -> one frame per second        (buckets at 0.0, 1.0, 2.0 ...)
        4  -> four frames per second      (buckets at 0.0, 0.25, 0.5, 0.75 ...)
        0.5 -> one frame every two seconds

    Frames land uniformly because the buckets are uniform: the first frame at
    or after each boundary wins. On a 15 fps source, ``per_second=4`` keeps
    roughly every 4th frame -- roughly, not exactly, because a bucket takes
    whichever real frame opens it rather than interpolating one.

    Bucketing on media time rather than counting every Nth frame is identical
    on a clean file and self-correcting on a lossy one: the frame that lands
    in second 47 is the frame for second 47 however many went missing before
    it. Counting drifts permanently after a gap; bucketing snaps back within
    one bucket.
    """

    def __init__(self, per_second: float = 1.0) -> None:
        if per_second <= 0:
            raise ValueError("per_second must be positive")
        self.per_second = per_second
        self._last_bucket: Optional[int] = None

    def bucket_of(self, media_ts: float) -> int:
        return int(media_ts * self.per_second)

    def accepts(self, frame: Frame) -> bool:
        """True if this frame opens a new slice of media time."""
        bucket = self.bucket_of(frame.media_ts)
        if self._last_bucket is None or bucket > self._last_bucket:
            self._last_bucket = bucket
            return True
        return False

    def config(self) -> dict:
        return {"per_second": self.per_second}
