"""Whole-frame appearance change, via CLIP cosine similarity."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..source import Frame
from .base import Sampler

if TYPE_CHECKING:
    from .components.embedders import FrameEmbedder


class ClipChangeSampler(Sampler):
    """Samples once the scene has changed enough to be worth describing.

    Each decimated frame is embedded and compared by cosine similarity. A
    frame is kept when similarity falls *below* ``threshold`` -- high
    similarity means nothing happened, so nothing is sampled. Within a chunk a
    static scene therefore yields the opening frame and nothing more.

    Two comparison modes, and the difference matters:

      ``reference``   -- compare against the last frame that was *kept*.
                         Change accumulates, so a slow pan eventually trips
                         the threshold. This is what deduplication wants.
      ``consecutive`` -- compare against the previous frame it evaluated.
                         Only detects instantaneous change; a slow pan never
                         trips it however far the scene travels.

    ``min_interval_s`` suppresses frames before they are embedded, so in
    ``consecutive`` mode "the previous frame" means the previous frame the
    sampler was allowed to look at, not the previous decimated frame.

    Threshold is video-dependent and must be measured. On the reference
    footage consecutive-frame similarity sits at p50 0.989 with sensor noise
    manufacturing false samples above ~0.97, leaving a usable window of
    roughly 0.94-0.97; 0.96 is the default.
    """

    name = "clip"

    def __init__(
        self,
        embedder: Optional["FrameEmbedder"] = None,
        threshold: float = 0.96,
        mode: str = "reference",
        min_interval_s: float = 0.0,
        max_per_chunk: Optional[int] = None,
        sampler_id: Optional[str] = None,
    ) -> None:
        super().__init__(min_interval_s, max_per_chunk, sampler_id)
        if mode not in ("reference", "consecutive"):
            raise ValueError("mode must be 'reference' or 'consecutive'")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a cosine similarity in [0, 1]")
        if embedder is None:
            from .components.embedders import CLIPEmbedder

            embedder = CLIPEmbedder()
        self.embedder = embedder
        self.threshold = threshold
        self.mode = mode
        self._reference: Optional[np.ndarray] = None
        self._last_score: Optional[float] = None

    def on_reset(self, chunk_id: int) -> None:
        # Chunks are independent: the reference never crosses a boundary, so
        # every chunk opens with a frame and its sampling is reproducible
        # regardless of what came before.
        self._reference = None
        self._last_score = None

    def last_score(self) -> Optional[float]:
        return self._last_score

    def describe(self, frame: Frame) -> np.ndarray:
        if frame.image is None:
            raise ValueError("ClipChangeSampler needs pixels; frame.image is None")
        return self.embedder.embed_one(frame.image)

    def compare(self, current: np.ndarray, reference: np.ndarray) -> float:
        # Embeddings are L2-normalised, so the dot product is the cosine.
        return float(np.dot(current, reference))

    def propose(self, frame: Frame, chunk_local_index: int) -> bool:
        embedding = self.describe(frame)

        if self._reference is None:
            self._last_score = None
            self._reference = embedding
            return True

        similarity = self.compare(embedding, self._reference)
        self._last_score = similarity

        if self.mode == "consecutive":
            # The reference is "the previous frame", so it moves every time
            # regardless of what is decided below.
            self._reference = embedding

        keep = similarity < self.threshold
        if keep and self.mode == "reference":
            self._reference = embedding
        return keep

    def config(self) -> dict:
        return {
            **self._base_config(),
            "threshold": self.threshold,
            "mode": self.mode,
            "embedder": self.embedder.config(),
        }
