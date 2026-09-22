"""`sentiment` -- tone per chunk, and where it turns.

Imported only when asked for by name; see `ner`.
"""

from .driver import SentimentAggregator

__all__ = ["SentimentAggregator"]
