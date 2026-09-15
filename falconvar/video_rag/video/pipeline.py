"""One decode pass, feeding every sampler from it.

Per frame: decimate on media time, ask the grid which chunk this is, reset the
samplers at a boundary, offer the frame to each, release the pixels.

The grid is an input and is never edited -- this asks `timeline.nearest(ts)`
and nothing else. No boundary is derived, corrected or merged here, which is
why nothing needs a streaming chunker.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

from ...shared.contracts.documents import Manifest, Media, Timeline
from .decimate import Decimator
from .reader import read_frames
from .samplers import Sampler
from .store import FrameStore


def ingest(media: Media, timeline: Timeline,
           samplers: Sequence[Sampler],
           per_second: float = 1.0,
           store: Optional[FrameStore] = None,
           store_scope: str = "sampled",
           on_chunk: Optional[Callable[[int, dict[str, Any]], None]] = None
           ) -> Manifest:
    """Decode once, offer every decimated frame to every sampler."""
    if not samplers:
        raise ValueError("name at least one sampler")
    ids = [s.sampler_id for s in samplers]
    if len(set(ids)) != len(ids):
        # The manifest keys frames by sampler id, so a collision would silently
        # drop one sampler's results into another's.
        raise ValueError(f"sampler ids must be unique, got {ids}")
    if store_scope not in ("sampled", "decimated"):
        raise ValueError("store_scope must be 'sampled' or 'decimated'")

    decimator = Decimator(per_second)
    config = {
        "decimator": decimator.config(),
        "samplers": [s.config() for s in samplers],
        "frame_store": ({**store.config(), "scope": store_scope}
                        if store else None),
    }

    chunks: list[dict[str, Any]] = [
        {"chunk_id": i, "decimated_frames": 0, "samplers": {}}
        for i in range(len(timeline))
    ]
    current: Optional[int] = None
    chunk_local_index = 0
    decimated = sampled = 0
    started = time.perf_counter()

    for frame in read_frames(media, decimator.accepts):
        decimated += 1
        # The grid says which chunk this is. `nearest` rather than `index_at`:
        # media time can fall past the grid at the tail, because the audio
        # stream is routinely a few milliseconds shorter than the video one, and
        # the nearest edge chunk should own those frames rather than them being
        # dropped or a chunk being invented.
        chunk_id = timeline.nearest(frame.media_ts)

        if chunk_id != current:
            if current is not None and on_chunk is not None:
                on_chunk(current, chunks[current])
            current = chunk_id
            for sampler in samplers:
                sampler.reset(chunk_id)
            chunk_local_index = 0

        chunk = chunks[chunk_id]
        chunk["decimated_frames"] += 1
        # "decimated" keeps every frame the samplers were offered, so a
        # threshold can be retuned later without decoding the video again.
        if store is not None and store_scope == "decimated":
            store.write(frame.index, frame.image)

        for sampler in samplers:
            if not sampler.accepts(frame, chunk_local_index):
                continue
            sampled += 1
            record: dict[str, Any] = {
                "index": frame.index,
                "media_ts": round(frame.media_ts, 3),
                "chunk_local_index": chunk_local_index,
            }
            # The address a fetcher can use. Seconds are a lossy rendering of
            # this; at a 1/1200000 timebase a rounded float lands elsewhere.
            if frame.pts is not None:
                record["pts"] = frame.pts
            if store is not None and store_scope == "sampled":
                store.write(frame.index, frame.image)
            score = sampler.last_score()
            if score is not None:
                record["score"] = round(score, 4)
            block = chunk["samplers"].setdefault(
                sampler.sampler_id, {"frame_count": 0, "frames": []})
            block["frames"].append(record)
            block["frame_count"] += 1

        chunk_local_index += 1
        # A describer wanting the whole window at once would hold frames until
        # the chunk closes. Nothing does, so pixels go back immediately and
        # peak memory stays at one frame.
        frame.release()

    if current is not None and on_chunk is not None:
        on_chunk(current, chunks[current])

    elapsed = time.perf_counter() - started
    stats = {
        "frames_decimated": decimated,
        "frames_sampled": sampled,
        "chunks": len(chunks),
        "chunks_with_frames": sum(1 for c in chunks if c["decimated_frames"]),
        "elapsed_s": round(elapsed, 3),
        **({"stored_frames": store.written,
            "stored_mb": round(store.bytes_written / 1024 / 1024, 2)}
           if store else {}),
    }

    return Manifest(
        video_id=media.video_id,
        timeline_fingerprint=timeline.fingerprint(),
        source={"path": media.path, "container": media.container_format,
                "duration_s": media.duration_s,
                **(media.video.as_dict() if media.video else {})},
        config=config,
        stats=stats,
        chunks=chunks,
    )


__all__ = ["ingest"]
