"""The `fold` kind: batch, summarise each batch, summarise the result.

A kind, not an aggregator. `summary` is a definition that names this one, and
so is any custom prompt of kind `fold`.
"""

from .driver import FoldAggregator, fold

__all__ = ["FoldAggregator", "fold"]
