"""Writing one stream of facts to several destinations.

Used by both stages and owned by neither, so neither has to reach into the
other for it. Ingest fans a manifest out to a file and to Postgres; describe
fans descriptions out the same way. What differs is the calls; what is shared
is what happens when one destination fails.

The first sink is the primary and its failures are real failures. The rest are
best-effort: one that raises is reported once and dropped for the remainder of
the run, rather than raising on every record or taking down work that is
expensive to redo. The consequence -- one copy complete, another partial -- is
why both stages carry a completeness signal that a reader can check rather
than assume.
"""

from __future__ import annotations

import sys
from typing import Any


class FanOut:
    """Primary plus best-effort secondaries. Subclasses add the calls."""

    def __init__(self, primary: Any, *secondary: Any) -> None:
        self.primary = primary
        self.secondary = list(secondary)

    def _try(self, sink: Any, call, what: str) -> None:
        try:
            call(sink)
        except Exception as exc:                       # noqa: BLE001
            name = type(sink).__name__
            self.secondary.remove(sink)
            print(
                f"  warning: {name} failed on {what} and was dropped for the "
                f"rest of this run ({type(exc).__name__}: {exc}). The primary "
                f"sink is unaffected; its copy stays authoritative.",
                file=sys.stderr,
            )
