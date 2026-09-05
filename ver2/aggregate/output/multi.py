"""One aggregate, several destinations. See ``ver2.fanout.FanOut``."""

from __future__ import annotations

from typing import Any

from ver2.fanout import FanOut


class MultiAggregateSink(FanOut):
    """Fans aggregates out to several sinks. An ``AggregateSink`` itself."""

    def write(self, record: dict[str, Any]) -> dict[str, Any]:
        written = self.primary.write(record)
        for sink in list(self.secondary):
            self._try(sink, lambda s: s.write(record),
                      f"aggregate {record.get('aggregate_id')}")
        return written
