"""Where transcripts go: a file, and a shared copy beside it."""

from .base import TranscriptSink
from .document import TRANSCRIPT_VERSION, TranscriptDocument, build_document
from .multi import MultiTranscriptSink


def __getattr__(name: str):
    # supabase-py is optional; a file-only run must not pay for it.
    if name == "SupabaseTranscript":
        from .supabase import SupabaseTranscript
        return SupabaseTranscript
    raise AttributeError(name)


__all__ = ["TRANSCRIPT_VERSION", "MultiTranscriptSink", "SupabaseTranscript",
           "TranscriptDocument", "TranscriptSink", "build_document"]
