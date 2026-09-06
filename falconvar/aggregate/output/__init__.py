"""Where aggregates go: files beside the video, and a shared copy."""

from .base import AggregateSink
from .document import AGGREGATE_VERSION, AggregateDocuments
from .multi import MultiAggregateSink


def __getattr__(name: str):
    if name == "SupabaseAggregates":
        from .supabase import SupabaseAggregates
        return SupabaseAggregates
    raise AttributeError(name)


__all__ = ["AGGREGATE_VERSION", "AggregateDocuments", "AggregateSink",
           "MultiAggregateSink", "SupabaseAggregates"]
