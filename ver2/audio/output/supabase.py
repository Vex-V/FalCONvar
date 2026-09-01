"""A transcript as Postgres rows.

Two tables, written in the order a reader would want them: the header first,
so a consumer polling `transcripts` never sees chunks belonging to a video it
knows nothing about.

**Replaced wholesale, not resumed.** The description sink asks what it already
holds because describing is per-chunk and a crash at chunk 90 must not redo
the first 89. Audio has no such state -- transcription is one whole-file pass
that either produced everything or nothing -- so a run always writes the
complete document, and there is no `existing()` to intersect.

The one thing wholesale replacement has to handle is a grid that shrank. A
re-run under a different policy can produce fewer chunks than the last, and an
upsert alone would leave the surplus rows behind, still keyed to this video,
still matching a `chunk_id` the timeline no longer has. They are deleted after
the new rows land rather than before, so a failure mid-write leaves the old
copy intact instead of a hole.
"""

from __future__ import annotations

from typing import Any, Optional

from ver2 import db


class SupabaseTranscript:
    """Writes a transcript document to `transcripts` + `transcript_chunks`."""

    name = "supabase"

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None,
                 client: Any = None) -> None:
        self.client = client or db.client_from_env(url, key, write=True)

    def write(self, document: dict[str, Any]) -> dict[str, Any]:
        video_id = document["video_id"]
        timeline = document.get("timeline") or {}

        self.client.table("audio_transcripts").upsert({
            "video_id": video_id,
            "language": document.get("language"),
            "language_probability": document.get("language_probability"),
            "speakers": document.get("speakers") or [],
            "model": document.get("model") or {},
            "audio": (document.get("source") or {}).get("audio") or {},
            "stats": document.get("stats") or {},
            "timeline": timeline,
            "timeline_fingerprint": document.get("timeline_fingerprint"),
            "segments": document.get("segments") or [],
        }, on_conflict="video_id").execute()

        chunks = document.get("chunks") or []
        if chunks:
            self.client.table("audio_chunks").upsert([{
                "video_id": video_id,
                "chunk_id": c["chunk_id"],
                "start_ts": c["start_ts"],
                "end_ts": c["end_ts"],
                "text": c.get("text") or "",
                "word_count": c.get("word_count") or 0,
                "structured": c.get("structured") or {},
                "timeline_fingerprint": document.get("timeline_fingerprint"),
            } for c in chunks], on_conflict="video_id,chunk_id").execute()

        # Only now, so a failure above leaves the previous copy whole.
        (self.client.table("audio_chunks").delete()
         .eq("video_id", video_id).gte("chunk_id", len(chunks)).execute())
        return document
