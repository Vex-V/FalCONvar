"""Random access into a video, for recovering frames a manifest points at.

Seeking is what ingestion deliberately avoids and retrieval unavoidably needs.
It is harder than it looks, because a video has none of the properties that
make random access cheap elsewhere:

  * Most frames are not self-contained. Only keyframes are, and they are
    sparse -- one every 80 frames on the reference footage. Reaching frame 345
    means starting at keyframe 320 and decoding 25 frames forward.
  * The seek primitive is approximate by design. FFmpeg seeks to a keyframe at
    or before a timestamp; landing on an exact frame is something built on top.
  * Whether "at or before" is honoured depends on the container's index. MP4
    has an exact sample table; MPEG-TS has none, so FFmpeg estimates a byte
    offset from average bitrate -- measured overshooting a target by 0.33 s
    after being asked to land 3 s before it.

So this verifies rather than trusts: seek, check that the first decoded frame
is at or before the target, back off and retry if not, and fall back to a scan
from the start when even that fails. Without the verify step MPEG-TS returns
the wrong frame 10 times out of 10.
"""

from __future__ import annotations

from typing import Iterator, Optional

import av
import cv2
import numpy as np

from .probe import UnusableSource
from .reader import ROTATIONS
from .types import SourceInfo

MAX_SEEK_ATTEMPTS = 5


class FrameFetcher:
    """Pulls individual frames out of a video by PTS, or by index as a fallback.

    Holds one open container across calls, because reopening per frame would
    re-parse the index every time. Targets are best requested in ascending
    order; nothing requires it, but a backwards jump costs a full seek.
    """

    def __init__(self, info: SourceInfo) -> None:
        self.info = info
        self._rotate = ROTATIONS.get(int(info.rotation))
        try:
            self._container = av.open(info.uri)
        except Exception as exc:
            raise UnusableSource(f"cannot open {info.uri}: {exc}") from None
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._time_base = self._stream.time_base
        self._start = self._stream.start_time or 0
        # One second in this stream's timebase, used as the backoff unit.
        self._second = int(1 / self._time_base) if self._time_base else 1
        self.seeks = 0
        self.retries = 0
        self.scans = 0

    def close(self) -> None:
        self._container.close()

    def __enter__(self) -> "FrameFetcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _finish(self, av_frame) -> np.ndarray:
        image = av_frame.to_ndarray(format="bgr24")
        return cv2.rotate(image, self._rotate) if self._rotate is not None else image

    def by_pts(self, target: int) -> Optional[np.ndarray]:
        """The frame whose PTS is ``target``, or None if it cannot be reached."""
        offset = 0
        for attempt in range(MAX_SEEK_ATTEMPTS):
            self._container.seek(
                max(self._start, target - offset),
                stream=self._stream,
                backward=True,
                any_frame=False,
            )
            self.seeks += 1
            overshot = False
            first = True
            for av_frame in self._container.decode(video=0):
                if av_frame.pts is None:
                    continue
                if first:
                    first = False
                    if av_frame.pts > target:
                        # The seek landed *past* what was asked for, which an
                        # indexless container will do. Decoding forward from
                        # here can never reach the target.
                        overshot = True
                        break
                if av_frame.pts == target:
                    return self._finish(av_frame)
                if av_frame.pts > target:
                    break                      # target absent from the stream
            if not overshot:
                break
            offset = self._second if offset == 0 else offset * 4
            self.retries += 1
        return self._scan_for(pts=target)

    def by_index(self, target: int) -> Optional[np.ndarray]:
        """The ``target``-th decoded frame. For sources with no usable PTS."""
        return self._scan_for(index=target)

    def _scan_for(
        self, pts: Optional[int] = None, index: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Last resort: decode from the beginning. Always correct, never fast."""
        self.scans += 1
        self._container.seek(self._start, stream=self._stream, backward=True)
        i = 0
        for av_frame in self._container.decode(video=0):
            if pts is not None and av_frame.pts == pts:
                return self._finish(av_frame)
            if index is not None and i == index:
                return self._finish(av_frame)
            i += 1
        return None

    def fetch(self, pts: Optional[int] = None, index: Optional[int] = None) -> Optional[np.ndarray]:
        """Prefer PTS; fall back to index when the source carries no timestamps."""
        if pts is not None and self.info.has_pts:
            return self.by_pts(pts)
        if index is not None:
            return self.by_index(index)
        raise ValueError("fetch needs a pts or an index")

    def stream_from(self, start_ts: float, count: int):
        """Yield up to ``count`` consecutive Frames starting at ``start_ts``.

        Seek once, then decode forward -- the asymmetry the whole pipeline is
        built around. Reading a window this way touches only the frames in it,
        where re-reading from the beginning would decode everything before it.
        """
        from .types import Frame

        if self.info.has_pts and self._time_base:
            target = int(start_ts / float(self._time_base)) + self._start
            self._container.seek(max(self._start, target), stream=self._stream,
                                 backward=True, any_frame=False)
        else:
            self._container.seek(self._start, stream=self._stream, backward=True)
        self.seeks += 1

        emitted = 0
        index = 0
        for av_frame in self._container.decode(video=0):
            if av_frame.pts is not None and self._time_base:
                media_ts = float(av_frame.pts * self._time_base)
            else:
                media_ts = index / self.info.fps if self.info.fps else 0.0
            index += 1
            if media_ts < start_ts:
                continue
            yield Frame(
                index=index - 1,
                media_ts=media_ts,
                pts=av_frame.pts,
                image=self._finish(av_frame),
                is_keyframe=bool(av_frame.key_frame),
            )
            emitted += 1
            if emitted >= count:
                return

    def fetch_many(self, targets: list[tuple[Optional[int], Optional[int]]]) -> Iterator:
        """Fetch (pts, index) pairs in ascending order, yielding (target, image)."""
        for pts, index in targets:
            yield (pts, index), self.fetch(pts=pts, index=index)

    def stats(self) -> dict:
        return {"seeks": self.seeks, "retries": self.retries, "scans": self.scans}
