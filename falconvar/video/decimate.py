"""Rate reduction at the head of the pipeline.

Nothing downstream sees native frame rate. A 25 fps source becomes whatever
`per_second` asks for, which is what keeps samplers and a VLM from reasoning
about twenty-five near-identical frames a second.

**Buckets on media time, never every Nth frame.** Identical on a clean file and
self-correcting on a lossy one: the frame that lands in second 47 is the frame
for second 47 however many went missing before it. Counting drifts permanently
after a gap; bucketing snaps back within one bucket.

It answers from a timestamp alone, which is what lets the reader ask it
*before* converting pixels -- see `reader.py`.
"""

from __future__ import annotations

from typing import Any, Optional


class Decimator:
    """Keeps the first frame seen in each slice of media time.

        1    -> one frame per second      (buckets at 0.0, 1.0, 2.0 ...)
        4    -> four per second           (0.0, 0.25, 0.5, 0.75 ...)
        0.5  -> one every two seconds
    """

    def __init__(self, per_second: float = 1.0) -> None:
        if per_second <= 0:
            raise ValueError("per_second must be positive")
        self.per_second = per_second
        self._last_bucket: Optional[int] = None

    def bucket_of(self, media_ts: float) -> int:
        return int(media_ts * self.per_second)

    def accepts(self, media_ts: float) -> bool:
        """True if this timestamp opens a new slice of media time."""
        bucket = self.bucket_of(media_ts)
        if self._last_bucket is None or bucket > self._last_bucket:
            self._last_bucket = bucket
            return True
        return False

    def config(self) -> dict[str, Any]:
        return {"per_second": self.per_second}


__all__ = ["Decimator"]
