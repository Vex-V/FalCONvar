"""Fixed windows of media time."""

from __future__ import annotations

from typing import Optional

from .base import Chunker


class UniformChunker(Chunker):
    """A boundary every ``duration_s`` seconds.

    Boundaries are a pure function of media time, so they cannot drift: two
    runs of the same video put the same second in the same chunk even when one
    of them lost a third of its frames, and a live stream that joined late
    still produces windows of the same length.

    The trade is that boundaries land on a stopwatch rather than on anything
    in the picture, so a chunk can cover the tail of one scene and the head of
    the next. A scene-cut chunker replaces chunk_id_of() and nothing else.
    """

    name = "uniform"

    def __init__(self, duration_s: float = 20.0) -> None:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        self.duration_s = duration_s

    def chunk_id_of(self, media_ts: float) -> int:
        return int(media_ts / self.duration_s)

    def bounds_of(self, chunk_id: int) -> tuple[float, Optional[float]]:
        return chunk_id * self.duration_s, (chunk_id + 1) * self.duration_s

    def config(self) -> dict:
        return {"name": self.name, "duration_s": self.duration_s}
