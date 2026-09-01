"""What the pipeline needs from a manifest destination, and nothing more.

The pipeline does not care whether a manifest becomes a file, a set of
Postgres rows or a stream -- only that a run can be started, that finished
chunks can be handed over as they close, and that the run can be declared
complete at the end.

Three calls, and **every sink gets the same arguments in all three**. A sink is
constructed with whatever its destination needs -- a path, a URL and a key --
and learns nothing else that way. Everything about the *run* arrives through
``begin``, identically for all of them. That is what makes writing to two
destinations at once a fan-out rather than a special case: the pipeline emits
one stream of facts, and each sink decides what storing them means.

``begin`` is separate from construction because ``source`` and ``config`` do
not exist until the video has been probed, and a caller assembling sinks from
command-line flags has no probe yet.

``finish`` takes ``chunks`` and ``config`` because the pipeline learns two
things only after the last frame: the final chunk's real ``end_ts``, and the
chunker's closing counters. Passing them in means a sink never has to expose
mutable attributes for the pipeline to reach into.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence


class ManifestSink(Protocol):
    """A destination for a manifest, written as the run proceeds."""

    def begin(
        self, video_id: str, source: dict[str, Any], config: dict[str, Any]
    ) -> None:
        """Start a run. Called once, after the probe, before any chunk."""
        ...

    def chunk_closed(
        self, chunk: dict[str, Any], stats: Optional[dict] = None
    ) -> None:
        """Record one finished chunk. Called as each chunk closes."""
        ...

    def finish(
        self,
        stats: Optional[dict] = None,
        chunks: Optional[Sequence[dict[str, Any]]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Mark the run complete, applying any late corrections."""
        ...
