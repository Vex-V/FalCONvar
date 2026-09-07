"""Documents into rows.

One mapping per document, and nothing else knows the table names. `sinks.write`
takes a callable rather than importing this, so a file-only run never touches
`db` and a component that has no row representation simply cannot be asked for
the `supabase` backend.

**The grid is written once.** `timelines` holds the policy and `chunks` holds
the spans; the manifest and the transcript store neither, and join on
`chunk_id`. That is the whole reason this file exists rather than each
component shipping its own INSERT.

**Order matters within a document.** A parent row before its children, and a
delete of orphaned chunks *after* the upserts, so a failure leaves the previous
copy whole rather than a hole.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from . import db


def _writer(fn: Callable[[str, dict[str, Any], Any], None]
            ) -> Callable[[str, dict[str, Any]], None]:
    """Wrap a mapping so `sinks.write` can call it with one client."""
    def write(video_id: str, document: dict[str, Any]) -> None:
        fn(video_id, document, db.client())
    return write


def _media(video_id: str, document: dict[str, Any], api: Any) -> None:
    db.upsert("videos", [{
        "video_id": video_id,
        "path": document["path"],
        "container": document["container_format"],
        "duration_s": document.get("duration_s"),
        "has_video": document.get("video") is not None,
        "has_audio": document.get("audio") is not None,
        "video_stream": document.get("video"),
        "audio_stream": document.get("audio"),
    }], api)


def _timeline(video_id: str, document: dict[str, Any], api: Any) -> None:
    db.upsert("timelines", [{
        "video_id": video_id,
        "policy": document["policy"],
        "derived_from": document["derived_from"],
        "params": document.get("params", {}),
        "fingerprint": document["fingerprint"],
        "duration_s": document["duration_s"],
        "chunk_count": document["chunk_count"],
    }], api)
    chunks = document.get("chunks", [])
    db.upsert("chunks", [{
        "video_id": video_id, "chunk_id": c["chunk_id"],
        "start_ts": c["start_ts"], "end_ts": c["end_ts"],
    } for c in chunks], api)
    # After the upserts. A grid that shrank leaves chunks nobody can play, and
    # everything keyed by chunk_id cascades from these.
    db.delete_stale_chunks("chunks", video_id, len(chunks), api)


def _cuts(video_id: str, document: dict[str, Any], api: Any) -> None:
    db.upsert("cuts", [{
        "video_id": video_id,
        "source": document["source"],
        "detector": document["detector"],
        "params": document.get("params", {}),
        "cut_times": document.get("cuts", []),
        "scores": document.get("scores"),
        "stats": document.get("stats", {}),
    }], api)


def _raw_transcript(video_id: str, document: dict[str, Any], api: Any) -> None:
    db.upsert("transcripts", [{
        "video_id": video_id,
        "model": document.get("model", {}),
        "track": document.get("track", {}),
        "stats": document.get("stats", {}),
        "segments": document.get("segments", []),
        "words": document.get("words", []),
        "turns": document.get("turns", []),
    }], api)


def _transcript(video_id: str, document: dict[str, Any], api: Any) -> None:
    # The header row already exists from `raw_transcript`; this adds the grid
    # it was cut on and the per-chunk rows.
    db.upsert("transcripts", [{
        "video_id": video_id,
        "timeline_fingerprint": document.get("timeline_fingerprint"),
        "model": document.get("model", {}),
        "stats": document.get("stats", {}),
    }], api)
    chunks = document.get("chunks", [])
    db.upsert("transcript_chunks", [{
        "video_id": video_id, "chunk_id": c["chunk_id"],
        "text": c.get("text", ""), "word_count": c.get("word_count", 0),
        "structured": c.get("structured", {}), "turns": c.get("turns", []),
    } for c in chunks], api)


def _manifest(video_id: str, document: dict[str, Any], api: Any) -> None:
    db.upsert("manifests", [{
        "video_id": video_id,
        "timeline_fingerprint": document["timeline_fingerprint"],
        "manifest_fingerprint": document["manifest_fingerprint"],
        "source": document.get("source", {}),
        "config": document.get("config", {}),
        "stats": document.get("stats", {}),
    }], api)

    # One row per (chunk, sampler) rather than a jsonb blob on the chunk, so
    # "which chunks did yolo pick frames in" is a query.
    by_id = {s["id"]: s for s in document.get("config", {}).get("samplers", [])}
    rows = []
    for chunk in document.get("chunks", []):
        for sampler_id, block in chunk.get("samplers", {}).items():
            config = by_id.get(sampler_id, {})
            rows.append({
                "video_id": video_id, "chunk_id": chunk["chunk_id"],
                "sampler_id": sampler_id,
                "sampler": config.get("name", sampler_id.split(":")[0]),
                "question": config.get("prompt") or sampler_id,
                "frame_count": block.get("frame_count", 0),
                "frames": block.get("frames", []),
            })
    db.upsert("chunk_samplers", rows, api)


def _descriptions(video_id: str, document: dict[str, Any], api: Any) -> None:
    rows = []
    for chunk in document.get("chunks", []):
        for sampler_id, block in chunk.get("samplers", {}).items():
            rows.append({
                "video_id": video_id, "chunk_id": chunk["chunk_id"],
                "sampler_id": sampler_id,
                "question": block.get("question", sampler_id),
                "frame_indexes": block.get("frame_indexes", []),
                "frame_count": block.get("frame_count", 0),
                "description": block.get("description"),
                "structured": block.get("structured", {}),
                "model": document.get("model", {}),
                "elapsed_s": block.get("elapsed_s"),
                "timeline_fingerprint": document.get("timeline_fingerprint"),
                "manifest_fingerprint": document.get("manifest_fingerprint"),
            })
    db.upsert("descriptions", rows, api)
    # No stale-chunk delete here, and that is deliberate: these cost inference,
    # so they do not cascade from the grid and are not deleted by a re-ingest.
    # Staleness is the fingerprint a reader compares.


def _aggregate(video_id: str, document: dict[str, Any], api: Any) -> None:
    db.upsert("aggregates", [{
        "video_id": video_id,
        "aggregate_id": document["aggregate_id"],
        "tier": document.get("tier", "free"),
        "payload": document.get("payload", {}),
        "inputs_fingerprint": document.get("inputs_fingerprint", ""),
    }], api)


#: artifact name -> the function that writes its rows. A document absent from
#: here has no Postgres representation, and `sinks.write` refuses the
#: `supabase` backend for it rather than silently writing nothing.
WRITERS: dict[str, Callable[[str, dict[str, Any]], None]] = {
    "media": _writer(_media),
    "timeline": _writer(_timeline),
    "cuts": _writer(_cuts),
    "raw_transcript": _writer(_raw_transcript),
    "transcript": _writer(_transcript),
    "manifest": _writer(_manifest),
    "descriptions": _writer(_descriptions),
    "aggregate": _writer(_aggregate),
}


def writer_for(artifact: str) -> Optional[Callable[[str, dict[str, Any]], None]]:
    return WRITERS.get(artifact)


__all__ = ["WRITERS", "writer_for"]
