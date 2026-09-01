"""Positional sampling: a frame every so many seconds, for a question you name.

Two things, and they are deliberately separate:

**When.** A frame every ``every_s`` seconds of media time. Not every Nth
decimated frame, which is what this used to count -- that made the cadence a
function of the decimator, so ``every_n=3`` meant one frame every 3 s at
``per_second=1`` and one every 0.75 s at 4. Media time is the only clock a
decision here may use, and a sampler counting frames was the last place that
was not true.

**What to ask.** ``prompt`` names the question the describe stage should put to
these frames. Left unset it is the scene question, and this is the positional
baseline it has always been. Set, it is how any question gets run on a clock:
``prompt="text"`` reads the screen every N seconds whether or not the writing
changed, ``prompt="overview"`` asks for a few sentences and nothing else.

That is worth having because the change samplers answer a different question
from the one it sounds like. `text` fires *when the writing changes* and pays
EasyOCR on every decimated frame to find out -- 98.1% of that sampler's cost.
If what you want is simply "read the screen every ten seconds", this does it
for no inference at all at ingest time, and lets the VLM do the reading.

The prompt is recorded in ``config()``, so it lands in the manifest and
therefore inside ``manifest_fingerprint``: changing it invalidates describe's
resume the way editing `vlm/prompts.py` does, rather than silently reusing
answers to a question no longer being asked.
"""

from __future__ import annotations

from typing import Optional

from ..source import Frame
from .base import Sampler


class UniformSampler(Sampler):
    """A frame every ``every_s`` seconds, for the question ``prompt`` names."""

    name = "uniform"

    def __init__(
        self,
        every_s: float = 3.0,
        prompt: Optional[str] = None,
        min_interval_s: float = 0.0,
        max_per_chunk: Optional[int] = None,
        sampler_id: Optional[str] = None,
    ) -> None:
        if every_s <= 0:
            raise ValueError("every_s must be positive")
        # The cadence *is* a minimum interval, so it is enforced by the base
        # class rather than reimplemented: `propose` says yes to everything and
        # the rate limit does the spacing. That also means a rate-limited frame
        # costs no inference, which is the whole reason the limit lives there.
        super().__init__(max(min_interval_s, every_s), max_per_chunk,
                         sampler_id or prompt)
        self.every_s = every_s
        self.prompt = prompt

    def propose(self, frame: Frame, chunk_local_index: int) -> bool:
        # Position alone. Everything about the timing is the interval above.
        return True

    def config(self) -> dict:
        return {**self._base_config(), "every_s": self.every_s,
                "prompt": self.prompt}


class OverviewSampler(UniformSampler):
    """A frame every ``every_s`` seconds, asking for a few sentences of prose.

    Nothing but `UniformSampler` with the question fixed, registered under its
    own name so `--sampler overview` reads as the thing it is rather than as a
    configuration of something else.
    """

    name = "overview"

    def __init__(self, every_s: float = 5.0, **kwargs) -> None:
        kwargs.pop("prompt", None)
        super().__init__(every_s=every_s, prompt="overview", **kwargs)
        self._sampler_id = kwargs.get("sampler_id") or "overview"
