"""Descriptions as Postgres rows.

One row per ``(video_id, chunk_id, sampler)``, which is the unit a describer
call covers. Upserted rather than inserted: re-describing a pair with a better
model should replace the answer, not fail.

**No foreign key to `videos`.** That is deliberate and is the whole reason this
is a separate table. Ingest replaces a video's manifest wholesale -- the sink
deletes the `videos` row on every run, and `chunks` cascades with it -- and
re-ingesting is cheap where describing is not. A cascade would let a 20-second
re-ingest destroy hours of inference.

The cost of no cascade is that stale rows can outlive the manifest that
justified them, so each row carries the fingerprint of the manifest it was
produced from. Staleness becomes a comparison a reader can make, rather than
something silently destroyed or silently trusted.
"""

from __future__ import annotations

from typing import Any, Optional

from ver2 import db


class SupabaseDescriptions:
    """Streams descriptions into Postgres. A ``DescriptionSink``."""

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        client: Any = None,
    ) -> None:
        self.video_id = ""
        self.model: dict[str, Any] = {}
        self.manifest_fingerprint = ""
        self.written = 0

        self.client = client or db.client_from_env(url, key, write=True)

    def begin(self, video_id: str, manifest: dict[str, Any], model: dict[str, Any]) -> None:
        from .document import fingerprint

        self.video_id = video_id
        self.model = model
        self.manifest_fingerprint = fingerprint(manifest)

    def existing(self) -> set[tuple[int, str]]:
        """What this video already has described, for this manifest and model.

        Rows from a different manifest are not counted: they describe frames
        this run is not being asked about. Neither are rows from a different
        describer -- switching from the stub to a real model, or between
        models, must not be skipped as "already described", which would report
        success and do nothing.
        """
        rows = (self.client.table("descriptions")
                .select("chunk_id,sampler,description,model,manifest_fingerprint")
                .eq("video_id", self.video_id)
                .eq("manifest_fingerprint", self.manifest_fingerprint)
                .execute().data)
        return {(r["chunk_id"], r["sampler"]) for r in rows
                if r.get("description") is not None and r.get("model") == self.model}

    def described(self, record: dict[str, Any]) -> None:
        self.client.table("descriptions").upsert({
            "video_id": self.video_id,
            "chunk_id": record["chunk_id"],
            "sampler": record["sampler"],
            "frame_indexes": record["frame_indexes"],
            "frame_count": record["frame_count"],
            "description": record["description"],
            "structured": record.get("structured") or {},
            "model": self.model,
            "elapsed_s": record.get("elapsed_s"),
            "manifest_fingerprint": self.manifest_fingerprint,
        }, on_conflict="video_id,chunk_id,sampler").execute()
        self.written += 1

    def finish(self, stats: Optional[dict] = None) -> dict[str, Any]:
        # Nothing to close: every row was durable the moment it was written,
        # and completeness is a property of the rows, not a flag on them.
        return {"video_id": self.video_id, "written": self.written,
                "stats": stats or {}}
