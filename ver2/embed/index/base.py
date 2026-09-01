"""What a vector index has to do, and nothing more.

Four calls. ``ensure`` makes the store ready for one embedder's width;
``existing`` says which units are already indexed *with the text they
currently have*; ``upsert`` writes; ``search`` ranks. Everything else --
which embedder, how to fuse, how to group hits into chunks -- is the caller's
business, so a second backend is a new file rather than a new concept.

``existing`` returns hashes rather than keys because that is the question worth
asking. A pair being present says nothing about whether its description has
since been rewritten by a better model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

from ..units import Unit


@dataclass
class Hit:
    """One matching description, with everything needed to act on it."""

    video_id: str
    chunk_id: int
    sampler: str
    score: float
    content: str
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    frame_indexes: Optional[list[int]] = None
    vector_rank: Optional[int] = None
    text_rank: Optional[int] = None

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.video_id, self.chunk_id, self.sampler)


class VectorIndex(Protocol):
    def ensure(self, embedder: dict[str, Any]) -> None:
        """Make the store ready for this embedder's vectors."""
        ...

    def existing(self, video_id: str, embedder: dict[str, Any]) -> dict[tuple[int, str], str]:
        """(chunk_id, sampler) -> text_hash already indexed for this embedder."""
        ...

    def upsert(self, units: Sequence[Unit], vectors: Sequence[Sequence[float]],
               embedder: dict[str, Any]) -> None:
        """Write or replace one batch."""
        ...

    def search(self, vector: Sequence[float], embedder: dict[str, Any],
               query_text: Optional[str] = None, video_id: Optional[str] = None,
               limit: int = 20, sampler: Optional[str] = None) -> list[Hit]:
        """Rank units against a query vector, and against its text if supported.

        ``sampler`` narrows the search to one question's answers. It is a
        filter over one shared space, not a space of its own -- everything in
        the index came from the same embedder and stays comparable.
        """
        ...
