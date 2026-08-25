"""Positional sampling: every Nth decimated frame within a chunk."""

from __future__ import annotations

from typing import Optional

from ..source import Frame
from .base import Sampler


class UniformSampler(Sampler):
    """Takes every ``every_n``-th decimated frame in each chunk.

    ``every_n`` counts decimated frames, not source frames, so what it means
    in seconds follows the decimator: at ``per_second=1`` every_n=3 is one
    frame every 3 seconds, at ``per_second=4`` it is one every 0.75 s.

    This decides on position alone, which makes it useless for retrieval and
    ideal as a baseline: its output is exactly predictable, so everything
    downstream can be tested without loading model weights.
    """

    name = "uniform"

    def __init__(
        self,
        every_n: int = 3,
        offset: int = 0,
        min_interval_s: float = 0.0,
        max_per_chunk: Optional[int] = None,
        sampler_id: Optional[str] = None,
    ) -> None:
        super().__init__(min_interval_s, max_per_chunk, sampler_id)
        if every_n < 1:
            raise ValueError("every_n must be >= 1")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        self.every_n = every_n
        self.offset = offset

    def propose(self, frame: Frame, chunk_local_index: int) -> bool:
        return (chunk_local_index - self.offset) % self.every_n == 0

    def config(self) -> dict:
        return {**self._base_config(), "every_n": self.every_n, "offset": self.offset}
