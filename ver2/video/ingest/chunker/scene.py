"""Boundaries at picture cuts, with duration caps at both ends.

PySceneDetect's own machinery -- ``open_video``, ``SceneManager``,
``detect()`` -- is deliberately bypassed. It is pull-based and batch: it owns
the read loop, decodes the file itself, and returns a scene list only once the
whole video has been seen. That means a second decode pass, and it cannot work
on a source with no end. Only the detector algorithm is borrowed, driven one
frame at a time from :meth:`observe`, so nothing looks ahead and a live stream
works unchanged.
"""

from __future__ import annotations

from typing import Optional

from ..source import Frame
from .base import Chunker


class SceneChunker(Chunker):
    """Cuts where the picture cuts, bounded at both ends.

    Both caps are load-bearing rather than tidy-up. Measured on an animated
    documentary, the default threshold found three cuts in three and a half
    minutes and left a 111-second scene; even at threshold 10 the longest ran
    44 s. A chunk that long is useless to retrieval and too much for one VLM
    call, so ``max_duration_s`` splits it. In the other direction a burst of
    fast cuts would produce one-second chunks, so ``min_duration_s`` merges
    them by ignoring cuts that arrive too soon.

    Detection runs on a downscaled copy at ~0.6 ms/frame -- cuts are a global
    property of the frame and full resolution buys nothing but time. That cost
    is the whole budget for the full-rate tap: at 25 fps a real-time consumer
    has 40 ms per frame to share.

    One wart, unavoidable on a live source: a cut shortly before the end
    leaves a stub chunk, which still costs a describer call because every
    chunk is guaranteed a frame.
    """

    name = "scene"

    def __init__(
        self,
        threshold: float = 27.0,
        min_duration_s: float = 5.0,
        max_duration_s: float = 60.0,
        detector: str = "content",
        detect_width: int = 320,
        fps: float = 30.0,
    ) -> None:
        if min_duration_s <= 0 or max_duration_s <= 0:
            raise ValueError("durations must be positive")
        if max_duration_s < min_duration_s:
            raise ValueError("max_duration_s must be >= min_duration_s")
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.threshold = threshold
        self.min_duration_s = min_duration_s
        self.max_duration_s = max_duration_s
        self.detector_name = detector
        self.detect_width = detect_width
        self.fps = fps

        self._detector = self._build_detector()
        self._boundaries: list[float] = [0.0]
        self.cuts_seen = 0
        self.cuts_merged = 0
        self.splits_forced = 0

    def _build_detector(self):
        from scenedetect import AdaptiveDetector, ContentDetector

        if self.detector_name == "adaptive":
            return AdaptiveDetector(adaptive_threshold=self.threshold)
        return ContentDetector(threshold=self.threshold)

    def observe(self, frame: Frame) -> None:
        """Native rate, before decimation. A cut is invisible at 1 fps."""
        import cv2
        from scenedetect import FrameTimecode

        image = frame.image
        if image is None:
            # Nothing to detect on. Silent by necessity, but it means a
            # release moved ahead of this call -- the ordering is load-bearing.
            return
        if image.shape[1] > self.detect_width:
            scale = self.detect_width / image.shape[1]
            image = cv2.resize(
                image,
                (self.detect_width, max(1, int(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        cuts = self._detector.process_frame(
            FrameTimecode(frame.index, self.fps), image
        )
        for cut in cuts:
            self._add_cut(cut.frame_num / self.fps)

    def _add_cut(self, media_ts: float) -> None:
        self.cuts_seen += 1
        # Also rejects a cut reported out of order: the difference goes
        # negative, which keeps the boundary list sorted and the reverse scan
        # in chunk_id_of valid. ContentDetector buffers up to min_scene_len
        # frames, so late reports are normal.
        if media_ts - self._boundaries[-1] < self.min_duration_s:
            self.cuts_merged += 1
            return
        self._boundaries.append(media_ts)

    def _extend_to(self, media_ts: float) -> None:
        """Force a boundary when a scene has run past the cap."""
        while media_ts - self._boundaries[-1] >= self.max_duration_s:
            self._boundaries.append(self._boundaries[-1] + self.max_duration_s)
            self.splits_forced += 1

    def chunk_id_of(self, media_ts: float) -> int:
        self._extend_to(media_ts)
        # Boundaries are appended in order, so the last one at or before this
        # timestamp is this frame's chunk.
        for index in range(len(self._boundaries) - 1, -1, -1):
            if media_ts >= self._boundaries[index]:
                return index
        return 0

    def bounds_of(self, chunk_id: int) -> tuple[float, Optional[float]]:
        start = self._boundaries[chunk_id]
        end = (
            self._boundaries[chunk_id + 1]
            if chunk_id + 1 < len(self._boundaries)
            else None
        )
        return start, end

    def config(self) -> dict:
        return {
            "name": self.name,
            "detector": self.detector_name,
            "threshold": self.threshold,
            "min_duration_s": self.min_duration_s,
            "max_duration_s": self.max_duration_s,
            "cuts_seen": self.cuts_seen,
            "cuts_merged": self.cuts_merged,
            "splits_forced": self.splits_forced,
        }
