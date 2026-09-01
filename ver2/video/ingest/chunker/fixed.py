"""Boundaries decided somewhere else.

The other chunkers derive boundaries as the decode pass runs -- uniform from
arithmetic, scene from frame content. This one is handed them. That is what
lets a run's grid come from the soundtrack: voice activity and speaker changes
are found by a whole-file audio pass, and the video pipeline then has to
honour a list it did not compute.

It is deliberately the dumbest chunker here. Everything interesting about an
audio-derived boundary -- the silence threshold, the minimum and maximum chunk
length, whether a pause is a breath or an edge -- was decided before the list
was built, and re-deciding any of it here would put the same policy in two
places. See `ver2/timeline.py` and `ver2/audio/segment`.

Because the spans are known up front, ``bounds_of`` never returns an open end.
The last chunk of a `Timeline` already stops where the media does, so the
correction the streaming chunkers apply at ``finish`` has nothing to do here.
"""

from __future__ import annotations

from typing import Optional

from ....timeline import Timeline
from .base import Chunker


class FixedChunker(Chunker):
    """A chunk grid supplied as data. A ``Chunker``."""

    name = "fixed"

    def __init__(self, timeline: Timeline) -> None:
        if not timeline.spans:
            raise ValueError("a fixed chunker needs at least one span")
        self.timeline = timeline

    def chunk_id_of(self, media_ts: float) -> int:
        index = self.timeline.index_at(media_ts)
        if index is not None:
            return index
        # Media time outside the grid. It happens at the tail when the audio
        # stream is fractionally shorter than the video one -- containers do
        # not promise equal durations -- so the nearest edge chunk owns it
        # rather than the frame being dropped or a chunk invented for it.
        first, last = self.timeline.spans[0], self.timeline.spans[-1]
        return 0 if media_ts < first[0] else len(self.timeline.spans) - 1

    def bounds_of(self, chunk_id: int) -> tuple[float, Optional[float]]:
        start, end = self.timeline.spans[chunk_id]
        return start, end

    def config(self) -> dict:
        return {"name": self.name,
                "policy": self.timeline.policy,
                "derived_from": self.timeline.derived_from,
                "timeline_fingerprint": self.timeline.fingerprint(),
                "chunks": len(self.timeline.spans),
                "params": self.timeline.params}
