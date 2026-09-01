"""Reading the chunk stream as it is written.

Following a live ingest polls Postgres, not the manifest file, and the
difference is structural rather than a preference. The file sink rewrites the
whole document every time a chunk closes -- it must, since JSON cannot be
appended to while someone reads it -- so a file watcher re-parses a growing
document and diffs it to work out what is new. The rows are an append: one
INSERT per chunk, and ``where chunk_id > $last`` returns exactly the new ones.

``videos.complete`` is what ends the loop. It exists to separate "no more
chunks yet" from "no more chunks ever", which is precisely the question a
follower has to answer every time it finds nothing new.
"""

from __future__ import annotations

import time
from typing import Any, Iterator, Optional

from ver2 import db

POLL_SECONDS = 2.0


#: Connecting is shared; only what is asked for afterwards is this module's.
client_from_env = db.client_from_env


def _as_chunk(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": row["chunk_id"],
        "start_ts": float(row["start_ts"]),
        "end_ts": float(row["end_ts"]),
        "decimated_frames": row["decimated_frames"],
        "samplers": row["samplers"],
    }


def _rows_after(client: Any, video_id: str, last: int) -> list[dict[str, Any]]:
    return (client.table("chunks")
            .select("chunk_id,start_ts,end_ts,decimated_frames,samplers")
            .eq("video_id", video_id).gt("chunk_id", last)
            .order("chunk_id").execute().data)


def follow_chunks(
    client: Any, video_id: str, poll_s: float = POLL_SECONDS
) -> Iterator[dict[str, Any]]:
    """Chunks in order as they land, ending when ingest reports itself complete.

    Completeness is checked only after a poll returns nothing. Checking it
    first would race: a chunk can be inserted between reading the flag and
    reading the rows, and that chunk would never be described.

    **The most recent chunk is held back by one.** A chunk's ``end_ts`` is
    provisional while it is the last one: the pipeline corrects the final
    chunk to where the video actually ends -- 60.333 s rather than the grid's
    80.0 on the reference file -- and only restates the rows at ``finish``.
    Describing it as it lands hands the model a window twenty seconds longer
    than the footage. Holding one back means a chunk is only released once
    another has proved it is not the last, and the true last one is re-read
    after ingest completes, which is the moment the correction exists.

    The cost is that descriptions lag one chunk behind ingestion. The
    alternative is telling the model something false about every video.
    """
    last = -1
    pending: Optional[dict[str, Any]] = None
    while True:
        rows = _rows_after(client, video_id, last)
        for row in rows:
            last = row["chunk_id"]
            if pending is not None:
                yield pending                  # a later chunk exists, so it is final
            pending = _as_chunk(row)
        if rows:
            continue
        done = (client.table("videos").select("complete")
                .eq("video_id", video_id).execute().data)
        if done and done[0]["complete"]:
            if pending is not None:
                fresh = _rows_after(client, video_id, pending["chunk_id"] - 1)
                yield _as_chunk(fresh[0]) if fresh else pending
            return
        time.sleep(poll_s)
