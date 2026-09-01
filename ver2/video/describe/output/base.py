"""What the reader needs from a description destination, and nothing more.

The same three-call shape ingest's ``ManifestSink`` uses, for the same reason:
every sink hears identical arguments, is constructed with only what its own
destination needs, and decides for itself what storing a description means.
Writing to a file and to Postgres is then a fan-out, not a branch in the loop.

``existing`` is the one addition, and it is what makes this stage resumable.
Describing is the expensive step -- a crash at chunk 90 must not redo the
first 89 -- so a sink is asked what it already holds before any work starts.
The reader trusts the primary's answer, because the primary is the copy that
is authoritative everywhere else too.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol


class DescriptionSink(Protocol):
    def begin(self, video_id: str, manifest: dict[str, Any], model: dict[str, Any]) -> None:
        """Start a run over one manifest. Called once, before any description."""
        ...

    def existing(self) -> set[tuple[int, str]]:
        """The (chunk_id, sampler) pairs already described for this video."""
        ...

    def described(self, record: dict[str, Any]) -> None:
        """Record one finished (chunk, sampler) description."""
        ...

    def finish(self, stats: Optional[dict] = None) -> dict[str, Any]:
        """Mark the pass complete."""
        ...
