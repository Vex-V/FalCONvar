"""Positional sampling: every Nth decimated frame.

**A stride over what the sampler was actually offered**, counted in
`chunk_local_index` -- the frame's position in its chunk's decimated stream and
the only thing about position a sampler is handed. A sampler answers yes or no
about the frame in front of it; counting the frames it was offered keeps the
decision inside what flows past it, where reaching for a wall of media time to
divide does not.

The cadence in seconds is therefore a consequence of decimation rather than a
second setting beside it: `every_n=3` is one frame every 3 s at
`per_second=1` and one every 0.75 s at 4. That is the point -- `per_second`
already decides how much of the video anything downstream may look at, so a
positional sampler is a stride over what survived it. A cadence in seconds
regardless is still expressible, through `min_interval_s`, which the base class
enforces for every sampler alike.

The default is 1: every decimated frame. There is no separate `overview`
sampler -- `overview` is a *prompt*, and every sampler takes one.
"""

from __future__ import annotations

from typing import Any, Optional

from ..reader import Frame
from .base import Sampler


class UniformSampler(Sampler):
    """Every ``every_n``th decimated frame."""

    name = "uniform"

    def __init__(self, every_n: int = 1, min_interval_s: float = 0.0,
                 max_per_chunk: Optional[int] = None,
                 sampler_id: Optional[str] = None,
                 prompt: Optional[str] = None) -> None:
        every_n = int(every_n)
        if every_n < 1:
            raise ValueError("every_n must be >= 1; it is a frame stride")
        # The stride is the strategy and lives in `propose`; `min_interval_s`
        # stays what it is for every other sampler -- an independent ceiling,
        # unset by default. Nothing is lost by deciding here because this
        # sampler runs no model, so a frame it turns down costs what one the
        # base class never offered it costs.
        super().__init__(min_interval_s, max_per_chunk, sampler_id, prompt)
        self.every_n = every_n

    def propose(self, frame: Frame, chunk_local_index: int) -> bool:
        # Index 0 is kept by this and by the every-chunk-keeps-a-frame
        # guarantee alike, so the two agree rather than one covering the other.
        return chunk_local_index % self.every_n == 0

    def config(self) -> dict[str, Any]:
        return {**self._base_config(), "every_n": self.every_n}


__all__ = ["UniformSampler"]
