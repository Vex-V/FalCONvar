"""Where an aggregate goes.

One call, like the transcript sink and unlike the manifest sink: an aggregator
produces its whole result at once, so there is no partial state a reader could
see and nothing a `begin`/`finish` pair would express.

The unit is one `(video_id, aggregate_id)` record, written as it lands rather
than collected -- so an expensive aggregator failing after a cheap one leaves
the cheap one's result on disk.
"""

from __future__ import annotations

from typing import Any, Protocol


class AggregateSink(Protocol):
    def write(self, record: dict[str, Any]) -> dict[str, Any]:
        """Store one finished aggregate and return what was stored."""
        ...
