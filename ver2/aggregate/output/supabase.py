"""Aggregates as Postgres rows.

One row per `(video_id, aggregate_id)`, the payload as `jsonb`.

**One table rather than one per aggregator**, and this is the opposite call
from the transcript tables on purpose. There, `descriptions` and `transcripts`
were kept apart because `model` would have meant two different things and half
the columns would have been null -- different provenance, different lifecycle.
Here every row is the *same kind of thing*: a video-level document derived from
the same inputs, differing only in the shape of its payload. `jsonb` is exactly
for that, and a new aggregator becomes a row rather than a migration.

No foreign key to `video_manifests`, for the reason nothing else has one:
re-ingesting costs seconds where an LLM aggregate costs money, and a cascade
would let the cheap operation destroy the expensive one. `inputs_fingerprint`
replaces it -- staleness is a comparison.
"""

from __future__ import annotations

from typing import Any, Optional

from ver2 import db


class SupabaseAggregates:
    """Upserts into `video_aggregates`. An ``AggregateSink``."""

    name = "supabase"

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None,
                 client: Any = None) -> None:
        self.client = client or db.client_from_env(url, key, write=True)

    def write(self, record: dict[str, Any]) -> dict[str, Any]:
        self.client.table("video_aggregates").upsert({
            "video_id": record["video_id"],
            "aggregate_id": record["aggregate_id"],
            "tier": record.get("tier"),
            "depends_on": record.get("depends_on") or [],
            "inputs_fingerprint": record.get("inputs_fingerprint"),
            "config": record.get("config") or {},
            "elapsed_s": record.get("elapsed_s"),
            "payload": record["payload"],
        }, on_conflict="video_id,aggregate_id").execute()
        return record

    def existing(self, video_id: str) -> dict[str, str]:
        rows = (self.client.table("video_aggregates")
                .select("aggregate_id,inputs_fingerprint")
                .eq("video_id", video_id).execute().data)
        return {r["aggregate_id"]: r["inputs_fingerprint"] or "" for r in rows}
