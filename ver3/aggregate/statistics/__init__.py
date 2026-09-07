"""The free tier: arithmetic over documents already produced.

No model, no network, no GPU. Everything here is a count, a ratio or a span,
and each answers something similarity answers approximately or not at all --
"how much of this is speech", "who dominated", "which chunks nothing described".

One file per aggregator, because the tier is the only thing they share.
"""

from __future__ import annotations

from .coverage import CoverageAggregator
from .speakers import SpeakersAggregator
from .stats import StatsAggregator

__all__ = ["CoverageAggregator", "SpeakersAggregator", "StatsAggregator"]
