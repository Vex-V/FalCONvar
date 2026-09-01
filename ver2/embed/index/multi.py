"""Writing one set of vectors to several indexes.

The same fan-out the manifest and description sinks use: first index
authoritative, the rest best-effort. See ``ver2.fanout.FanOut``.

``existing`` is the intersection, for the reason it is everywhere else here --
a copy that fell behind has to be able to catch up, and asking only the primary
means it never does. A unit counts as indexed only if every index holds it with
the same text hash.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ver2.fanout import FanOut

from ..units import Unit
from .base import Hit


class MultiIndex(FanOut):
    """Fans vectors out to several indexes. A ``VectorIndex`` itself."""

    def ensure(self, embedder: dict[str, Any]) -> None:
        self.primary.ensure(embedder)
        for index in list(self.secondary):
            self._try(index, lambda i: i.ensure(embedder), "ensure")

    def existing(self, video_id: str, embedder: dict[str, Any]) -> dict[tuple[int, str], str]:
        done = dict(self.primary.existing(video_id, embedder))
        for index in list(self.secondary):
            before = dict(done)
            def narrow(i, acc=done):
                theirs = i.existing(video_id, embedder)
                for key in list(acc):
                    if theirs.get(key) != acc[key]:
                        acc.pop(key)
            self._try(index, narrow, "existing")
            if index not in self.secondary:
                done = before
        return done

    def upsert(self, units: Sequence[Unit], vectors: Sequence[Sequence[float]],
               embedder: dict[str, Any]) -> None:
        self.primary.upsert(units, vectors, embedder)
        for index in list(self.secondary):
            self._try(index, lambda i: i.upsert(units, vectors, embedder), "upsert")

    def search(self, vector: Sequence[float], embedder: dict[str, Any],
               query_text: Optional[str] = None, video_id: Optional[str] = None,
               limit: int = 20, sampler: Optional[str] = None) -> list[Hit]:
        # Searching is a read: one backend answers, and which one is a choice
        # the caller makes explicitly rather than a fan-out.
        return self.primary.search(vector, embedder, query_text, video_id,
                                   limit, sampler)
