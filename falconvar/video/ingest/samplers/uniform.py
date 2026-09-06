"""Positional sampling: every Nth decimated frame.

Two things, and they are deliberately separate -- ``prompt`` now lives on the
base class, so the second of them is every sampler's, not this one's:

**When.** Every ``every_n`` frames of the decimated stream -- the stride is
counted in ``chunk_local_index``, which is the frame's position in its chunk's
decimated stream and the only thing about position a sampler is handed. A
sampler sees the stream one frame at a time and answers yes or no about *that
frame*; counting the frames it was actually offered keeps the decision inside
what flows downstream, where reaching for a wall of media time to divide by
does not.

The cadence in seconds is therefore a consequence of decimation, not a second
setting beside it: ``every_n=3`` is one frame every 3 s at ``per_second=1`` and
one every 0.75 s at 4. That is the point rather than a caveat -- ``per_second``
already decides how much of the video anything downstream may look at, and a
positional sampler is a stride over what survived it. Wanting a cadence in
seconds regardless means saying so in ``min_interval_s``, which is enforced in
the base class for every sampler alike.

**What to ask.** ``prompt`` is inherited from `Sampler`, because pairing a
question with a strategy is not this sampler's privilege -- see `base.py`. It
matters most here all the same, since this is the sampler with no opinion of
its own about content: ``uniform:text`` reads the screen on a stride whether or
not the writing changed, ``uniform:overview`` asks for a few sentences.

That is worth having because the change samplers answer a different question
from the one it sounds like. `text` fires *when the writing changes* and pays
EasyOCR on every decimated frame to find out -- 98.1% of that sampler's cost.
If what you want is simply "read the screen every so often", this does it for
no inference at all at ingest time, and lets the VLM do the reading.

The stride is recorded in ``config()``, so it lands in the manifest and
therefore inside ``manifest_fingerprint``; so does the decimator's
``per_second``, which is what makes the stride mean something. Changing either
invalidates describe's resume the way editing `vlm/prompts.py` does, rather
than silently reusing answers to a question no longer being asked.
"""

from __future__ import annotations

from typing import Optional

from ..source import Frame
from .base import Sampler


class UniformSampler(Sampler):
    """Every ``every_n``th decimated frame."""

    name = "uniform"

    def __init__(
        self,
        every_n: int = 1,
        min_interval_s: float = 0.0,
        max_per_chunk: Optional[int] = None,
        sampler_id: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> None:
        every_n = int(every_n)
        if every_n < 1:
            raise ValueError("every_n must be >= 1; it is a frame stride")
        # No cadence is folded into min_interval_s any more. The stride is the
        # strategy and lives in `propose`; the interval stays what it is for
        # every other sampler -- an independent ceiling on frequency, unset by
        # default. Nothing is lost by deciding in `propose` here because this
        # sampler runs no model: `describe` returns None, so a frame it turns
        # down costs the same as one the base class never offered it.
        super().__init__(min_interval_s, max_per_chunk, sampler_id, prompt)
        self.every_n = every_n

    def propose(self, frame: Frame, chunk_local_index: int) -> bool:
        # Position in the decimated stream alone. Index 0 is kept by this and
        # by the base class's every-chunk-keeps-a-frame guarantee alike, so the
        # two agree rather than one covering for the other.
        return chunk_local_index % self.every_n == 0

    def config(self) -> dict:
        return {**self._base_config(), "every_n": self.every_n}
