"""Chunker contract.

A chunk is the unit a describer summarises and retrieval returns, so where the
boundaries fall decides what a search result means.

A chunker maps media time to a chunk id and back to bounds. It is asked only
about decimated frames, and it must answer from media time alone -- never from
how many frames arrived -- so two runs of the same video agree even if one of
them lost frames.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..source import Frame


class Chunker(ABC):
    """Decides which chunk a moment in media time belongs to."""

    name: str = "base"

    def observe(self, frame: Frame) -> None:
        """Called for every frame at native rate, before decimation.

        Only strategies that need temporal density implement this -- a scene
        cut is indistinguishable from ordinary motion once decimated to 1 fps.
        The budget here is hard: at 25 fps a real-time consumer has 40 ms per
        frame to share, so nothing model-sized belongs in it.
        """

    @abstractmethod
    def chunk_id_of(self, media_ts: float) -> int: ...

    @abstractmethod
    def bounds_of(self, chunk_id: int) -> tuple[float, Optional[float]]:
        """Start and end of a chunk. End is None while it is still open."""

    def config(self) -> dict:
        return {"name": self.name}
