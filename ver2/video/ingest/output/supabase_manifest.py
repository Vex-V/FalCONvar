"""The manifest as Postgres rows instead of a JSON document.

Same information, different shape. ``videos`` holds the one-per-run header;
``chunks`` holds one row per chunk with its ``samplers`` payload stored
verbatim as ``jsonb``, so every frame record keeps its ``index``, ``media_ts``,
``pts``, ``chunk_local_index`` and ``score`` unchanged.

The file writer rewrites the whole document whenever a chunk closes, because a
JSON file cannot be safely appended to while someone reads it. That cost is
quadratic in chunks -- 137 full rewrites on a 23-minute video. Here a closing
chunk is one INSERT, and a reader following along polls
``where chunk_id > $last`` instead of re-parsing a growing document.

``complete`` is the same signal it is in the file: false while the run is in
flight, true once ``finish`` lands. A consumer uses it to tell "no more chunks
yet" from "no more chunks ever".

Writes need the **secret** key, which bypasses RLS. The publishable key is
read-only by policy and is what a manifest consumer holds; see SUPABASE.md.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ver2 import db

from .manifest import MANIFEST_VERSION


class SupabaseManifestWriter:
    """Streams a manifest into Postgres as the run proceeds. A ``ManifestSink``."""

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        client: Any = None,
    ) -> None:
        # Construction takes only what is specific to reaching Postgres. Every
        # fact about the run arrives through begin(), the same way it does for
        # every other sink -- so connecting can fail before a frame is decoded.
        self.video_id = ""
        self.source: dict[str, Any] = {}
        self.config: dict[str, Any] = {}
        self.stats: dict[str, Any] = {}
        self.chunks: list[dict[str, Any]] = []
        self.complete = False

        self.client = client or db.client_from_env(url, key, write=True)

    def begin(
        self, video_id: str, source: dict[str, Any], config: dict[str, Any]
    ) -> None:
        """Claim the video_id, so a reader sees an incomplete run from the start."""
        self.video_id = video_id
        self.source = source
        self.config = config
        # Replace, not insert: re-ingesting a video should supersede its
        # manifest rather than fail. The cascade on chunks takes the old rows.
        self.client.table("video_manifests").delete().eq("video_id", video_id).execute()
        self.client.table("video_manifests").insert({
            "video_id": video_id,
            "complete": False,
            "manifest_version": MANIFEST_VERSION,
            "source": source,
            "config": config,
            "stats": {},
        }).execute()

    def document(self) -> dict[str, Any]:
        return {
            "manifest_version": MANIFEST_VERSION,
            "video_id": self.video_id,
            "complete": self.complete,
            "source": self.source,
            "config": self.config,
            "stats": self.stats,
            "chunks": self.chunks,
        }

    def _row(self, chunk: dict[str, Any]) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "chunk_id": chunk["chunk_id"],
            "start_ts": chunk["start_ts"],
            "end_ts": chunk["end_ts"],
            "decimated_frames": chunk["decimated_frames"],
            "samplers": chunk["samplers"],
        }

    def chunk_closed(self, chunk: dict[str, Any], stats: Optional[dict] = None) -> None:
        """One INSERT. The row is readable the moment it lands."""
        self.chunks.append(chunk)
        if stats is not None:
            self.stats = stats
        self.client.table("video_chunks").insert(self._row(chunk)).execute()

    def finish(
        self,
        stats: Optional[dict] = None,
        chunks: Optional[Sequence[dict[str, Any]]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if stats is not None:
            self.stats = stats
        if config is not None:
            self.config = config
        if chunks is not None:
            # The late corrections are real changes to rows already written:
            # the last chunk's end_ts, and any chunk the pipeline restated.
            self.chunks = list(chunks)
            for chunk in self.chunks:
                self.client.table("video_chunks").upsert(
                    self._row(chunk), on_conflict="video_id,chunk_id"
                ).execute()
            # A restated run can hold FEWER chunks than were streamed. Rows go
            # in as each chunk closes, but the pipeline may fold a too-short
            # final chunk into the one before it once it knows where the media
            # really ends -- so the last row written can name a chunk that no
            # longer exists. Observed on test2: 60.325 s at 20 s closed four
            # chunks, the last 0.325 s long, and the merge left `chunk_id = 3`
            # orphaned in Postgres while the file held three. The file writer
            # rewrites its whole document and never had the problem; a sink
            # that appends has to clean up after a shrink.
            #
            # After the upserts, never before: a failure above then leaves the
            # previous copy whole rather than a hole.
            (self.client.table("video_chunks").delete()
             .eq("video_id", self.video_id)
             .gte("chunk_id", len(self.chunks)).execute())
        self.complete = True
        self.client.table("video_manifests").update({
            "complete": True,
            "stats": self.stats,
            "config": self.config,
        }).eq("video_id", self.video_id).execute()
        return self.document()
