"""Embedding a video's summary, so a question can find the video.

`chunk_embeddings` answers *which twenty seconds*. This answers *which video* --
a different unit, so a different table and a different result shape. A chunk
that mentions reactors is not the same thing as a video that is about them, and
folding the two into one ranking would put a whole-video "moment" beside real
ones, competing on a scale it does not share.

**Only the summary.** Everything else `aggregate` produces is statistical, and a
vector of a count answers nothing: "how many people" is a number to read, not a
direction in space.

**pgvector only.** The lexical half is what makes this work -- a corpus of ten
video summaries is far too small for dense retrieval to separate on its own,
and BM25 over a topic list is exactly the right tool. Qdrant is dense-only, so
there is nothing here for it to do that Postgres does not do better.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ver2 import db

from .units import embedder_key, text_hash


def _render(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The text to embed, and the structure to keep beside it.

    Key points and topics are folded into the embedded text for the same reason
    the structured fields are folded into a description's: every summary of
    every video repeats the same shape, and the distinctive content is in the
    lists. Measured on descriptions, summary-only scored 0.528 MRR against
    0.705 for both halves.
    """
    parts = [payload.get("summary", "").strip()]
    if payload.get("key_points"):
        parts.append("key points: " + "; ".join(payload["key_points"]))
    if payload.get("topics"):
        parts.append("topics: " + ", ".join(payload["topics"]))
    structured = {"key_points": payload.get("key_points") or [],
                  "topics": payload.get("topics") or []}
    return "\n\n".join(p for p in parts if p), structured


def index_summary(video_id: str, embedder: Any, out_root: Path = Path("out"),
                  client: Any = None, force: bool = False) -> Optional[dict[str, Any]]:
    """Embed one video's summary aggregate. Returns None when there is none."""
    path = Path(out_root) / video_id / "aggregates" / "summary.json"
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document.get("payload") or {}
    content, structured = _render(payload)
    if not content:
        return None

    config = embedder.config()
    key = embedder_key(config)
    digest = text_hash(content)
    client = client or db.client_from_env(write=True)

    if not force:
        held = (client.table("video_embeddings").select("text_hash")
                .eq("video_id", video_id).eq("kind", "summary")
                .eq("embedder", key).execute().data)
        # The same hash comparison the chunk index uses, for the same reason:
        # the question is not "do I have a vector for this video" but "do I
        # have a vector for this text".
        if held and held[0]["text_hash"] == digest:
            return {"video_id": video_id, "embedder": key, "embedded": 0,
                    "skipped": 1, "text_hash": digest}

    vector = embedder.embed_documents([content])[0]
    client.table("video_embeddings").upsert({
        "video_id": video_id, "kind": "summary", "embedder": key,
        "dims": len(vector),
        "embedding": "[" + ",".join(f"{float(v):.7g}" for v in vector) + "]",
        "content": content, "structured": structured, "text_hash": digest,
        "inputs_fingerprint": document.get("inputs_fingerprint"),
    }, on_conflict="video_id,kind,embedder").execute()
    return {"video_id": video_id, "embedder": key, "embedded": 1, "skipped": 0,
            "text_hash": digest, "chars": len(content)}


def search(query: str, embedder: Any, limit: int = 10,
           client: Any = None) -> list[dict[str, Any]]:
    """Rank videos by how well their summaries answer the question."""
    client = client or db.client_from_env(write=True)
    vector = embedder.embed_query(query)
    rows = client.rpc("search_videos", {
        "p_embedder": embedder_key(embedder.config()),
        "p_query_vector": "[" + ",".join(f"{float(v):.7g}" for v in vector) + "]",
        "p_query_text": query,
        "p_limit": limit,
    }).execute().data or []
    return [{"video_id": r["video_id"], "kind": r["kind"],
             "summary": r["content"], "structured": r["structured"],
             "vector_rank": r.get("vector_rank"), "text_rank": r.get("text_rank"),
             "score": float(r["score"])} for r in rows]
