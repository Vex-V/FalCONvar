"""The sampler contract.

A sampler sees the decimated stream one frame at a time and answers yes or no.
It cannot look ahead, revisit a frame it declined, or buffer the chunk.

Three things live here rather than in each strategy:

    min_interval_s  smallest gap between two kept frames
    max_per_chunk   ceiling on frames kept from one chunk
    prompt          which question describe should ask about these frames

The rate limits short-circuit before `propose` runs, so a rate-limited frame
costs no inference. Every chunk keeps at least one frame.

`prompt` is on the base class, so any strategy pairs with any question as
`name:prompt` -- `uniform:text` reads the screen on a stride, `yolo:overview`
asks for prose about frames the people detector chose. Unpaired, the question
is the sampler's own name. Nothing here reads the prompt: it is an opaque
string, validated by whoever built the sampler, because a sampler knowing the
prompt registry would be an edge from ingest to describe.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..reader import Frame


class Sampler(ABC):
    """Base class for frame-selection strategies."""

    name: str = "base"

    def __init__(self, min_interval_s: float = 0.0,
                 max_per_chunk: Optional[int] = None,
                 sampler_id: Optional[str] = None,
                 prompt: Optional[str] = None) -> None:
        if min_interval_s < 0:
            raise ValueError("min_interval_s must be >= 0")
        if max_per_chunk is not None and max_per_chunk < 1:
            raise ValueError("max_per_chunk must be >= 1; every chunk keeps a frame")
        self.min_interval_s = min_interval_s
        self.max_per_chunk = max_per_chunk
        self._sampler_id = sampler_id
        self.prompt = prompt
        self._kept_in_chunk = 0
        self._last_kept_ts: Optional[float] = None

    @property
    def sampler_id(self) -> str:
        """The manifest key, and the key every later stage inherits.

        `name:prompt` for a pairing, the bare name otherwise. Keying by the
        question alone was the earlier rule and it breaks the moment any
        sampler can ask anything: `yolo:overview` and `clip:overview` would
        both be `overview`, colliding on one key while holding different
        frames. Keying by the strategy alone loses the question, which is the
        more useful half when reading a search result.
        """
        if self._sampler_id:
            return self._sampler_id
        if self.prompt and self.prompt != self.name:
            return f"{self.name}:{self.prompt}"
        return self.name

    def reset(self, chunk_id: int) -> None:
        """Called once when a chunk opens, before any frame is offered.

        Samplers forget everything at a boundary, so a chunk's sampling never
        depends on the chunk before it.
        """
        self._kept_in_chunk = 0
        self._last_kept_ts = None
        self.on_reset(chunk_id)

    def on_reset(self, chunk_id: int) -> None:
        """Subclass hook for clearing strategy state at a boundary."""

    def accepts(self, frame: Frame, chunk_local_index: int) -> bool:
        """Final decision for one frame. Do not override -- implement propose."""
        if self.max_per_chunk is not None and self._kept_in_chunk >= self.max_per_chunk:
            # Chunk is full. Skipping here rather than inside the strategy is
            # what makes the cap free instead of merely quiet.
            return False

        first_of_chunk = self._last_kept_ts is None
        if (not first_of_chunk
                and frame.media_ts - self._last_kept_ts < self.min_interval_s):
            return False

        # The strategy still sees every frame it is allowed to see, so its own
        # state stays coherent; the guarantee is layered on top.
        keep = self.propose(frame, chunk_local_index) or first_of_chunk
        if keep:
            self._kept_in_chunk += 1
            self._last_kept_ts = frame.media_ts
        return keep

    @abstractmethod
    def propose(self, frame: Frame, chunk_local_index: int) -> bool:
        """The strategy's opinion, before rate limits and the chunk guarantee.

        ``chunk_local_index`` counts decimated frames in the current chunk from
        0. It is the only thing about position a sampler is handed.
        """

    # The two halves of a change sampler, split so a calibrator can cache the
    # expensive one and replay the cheap one at many thresholds. Running a
    # model once per frame and comparing thousands of times is what makes a
    # threshold sweep affordable -- the same split `boundaries/scenes.py` uses.

    def describe(self, frame: Frame) -> Any:
        """The model's output for this frame. None for positional samplers."""
        return None

    def compare(self, current: Any, reference: Any) -> Optional[float]:
        """Similarity in [0, 1]. Lower means more changed."""
        return None

    def last_score(self) -> Optional[float]:
        """Whatever the last decision was based on, for the manifest.

        Content-driven samplers record it so a threshold can be retuned by
        reading a run's output rather than decoding the video again.
        """
        return None

    def _base_config(self) -> dict[str, Any]:
        # `prompt` omitted when unset rather than written as null, so a sampler
        # that pairs no question produces exactly the config it always did and
        # `manifest_fingerprint` does not move for every video already ingested.
        config: dict[str, Any] = {
            "id": self.sampler_id,
            "name": self.name,
            "min_interval_s": self.min_interval_s,
            "max_per_chunk": self.max_per_chunk,
        }
        if self.prompt is not None:
            config["prompt"] = self.prompt
        return config

    def config(self) -> dict[str, Any]:
        """Serialised into the manifest so a run can be reproduced."""
        return self._base_config()


__all__ = ["Sampler"]
