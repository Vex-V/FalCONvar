"""Where descriptions go.

The document is the record and Postgres is the queryable copy, exactly as with
manifests. Both hear the same four calls; neither knows the other exists.
"""

from .base import DescriptionSink
from .document import DESCRIPTION_VERSION, DescriptionDocument, fingerprint
from .multi import MultiDescriptionSink


def __getattr__(name: str):
    # supabase-py is optional; a file-only run must not pay for it.
    if name == "SupabaseDescriptions":
        from .supabase import SupabaseDescriptions
        return SupabaseDescriptions
    raise AttributeError(name)


__all__ = [
    "DESCRIPTION_VERSION",
    "DescriptionDocument",
    "DescriptionSink",
    "MultiDescriptionSink",
    "SupabaseDescriptions",
    "fingerprint",
]
