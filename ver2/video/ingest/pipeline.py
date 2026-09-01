"""One decode pass, feeding every stage from it.

    probe -> read -> chunker.observe -> decimate -> chunk -> sampler(s)

One pass rather than one per sampler: a live source can only be consumed once,
and re-decoding a file per sampler multiplies the most expensive stage for no
gain. The CLI that drives this lives in driver.py; nothing here knows about
argument parsing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from ver2.video.ingest import chunker as chunker_mod
from ver2.video.ingest import samplers as samplers_mod
from ver2.timeline import MIN_TAIL_FRACTION
from ver2.video.ingest.chunker import Chunker
from ver2.video.ingest.output import MANIFEST_VERSION, FrameStore, ManifestSink
from ver2.video.ingest.samplers import Sampler
from ver2.video.ingest.source import Decimator, probe, read_frames


@dataclass
class Chunk:
    """One window of media time, and what each sampler kept from it."""

    chunk_id: int
    start_ts: float
    end_ts: float
    decimated: int = 0
    # media_ts of the last frame that landed here. The only end a still-open
    # chunk has, since the chunker cannot know where it stops until it does.
    last_ts: float = 0.0
    # sampler id -> list of {index, media_ts, chunk_local_index, score?}
    picks: dict[str, list[dict]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "start_ts": round(self.start_ts, 3),
            "end_ts": round(self.end_ts, 3),
            "decimated_frames": self.decimated,
            "samplers": {
                sampler_id: {"frame_count": len(frames), "frames": frames}
                for sampler_id, frames in self.picks.items()
            },
        }



def _merge_tail(chunks: list["Chunk"], min_tail_s: float) -> None:
    """Fold a too-short final chunk into its predecessor, in place.

    The frames were already chosen -- the samplers reset at the boundary that
    is being removed, so the merged chunk holds one extra mandatory
    first-frame-of-chunk pick. That is the honest cost of deciding this after
    the pass rather than before it, and an extra frame is not a defect: the
    manifest still names exactly the frames that were kept, and the store
    still holds them. Nothing is invented and nothing is dropped.

    ``chunk_local_index`` is **shifted, not renumbered**. It is the frame's
    position in the chunk's decimated stream, not a sequence number over the
    picks -- a uniform sampler at stride 3 yields 0, 3, 6, 9 for four picks --
    so the tail's indices continue from however many decimated frames the host
    already had. Renumbering them 0..n would silently redefine the field.
    """
    if len(chunks) < 2:
        return
    tail = chunks[-1]
    if tail.end_ts - tail.start_ts >= min_tail_s:
        return
    host = chunks[-2]
    offset = host.decimated
    for sampler_id, picks in tail.picks.items():
        for pick in picks:
            pick["chunk_local_index"] += offset
        host.picks.setdefault(sampler_id, []).extend(picks)
        host.picks[sampler_id].sort(key=lambda p: p["media_ts"])
    host.decimated += tail.decimated
    host.end_ts = tail.end_ts
    host.last_ts = max(host.last_ts, tail.last_ts)
    chunks.pop()


@dataclass
class Result:
    manifest: dict
    chunks: list[Chunk]
    frames_read: int
    frames_decimated: int
    frames_sampled: int
    elapsed_s: float


def ingest(
    uri: str,
    per_second: float = 1.0,
    chunking: str = "uniform",
    chunking_options: Optional[dict[str, Any]] = None,
    chunk_duration_s: float = 20.0,
    sampler_specs: Sequence[Sampler] = (),
    video_id: Optional[str] = None,
    sink: Optional[ManifestSink] = None,
    store: Optional[FrameStore] = None,
    store_scope: str = "sampled",
) -> Result:
    """Run one decode pass, feeding every sampler from it.

    One pass rather than one per sampler: a live source can only be consumed
    once, and re-decoding a file per sampler multiplies the most expensive
    stage for no gain.
    """
    info = probe(uri)
    video_id = video_id or Path(uri).stem
    decimator = Decimator(per_second=per_second)

    options = dict(chunking_options or {})
    if chunking == "scene":
        # Scene detection needs the source rate to turn frame numbers into
        # media time, and the duration cap doubles as the chunk length.
        options.setdefault("max_duration_s", chunk_duration_s)
        options.setdefault("fps", info.fps if info.fps_trusted else 30.0)
    elif chunking == "fixed":
        # The grid is the argument. Injecting a chunk length here would be a
        # second opinion about boundaries that were already decided.
        pass
    else:
        options.setdefault("duration_s", chunk_duration_s)
    chunker: Chunker = chunker_mod.build(chunking, **options)

    samplers = list(sampler_specs) or [samplers_mod.build("uniform")]
    # The manifest keys frames by sampler id, so a collision would silently
    # drop one sampler's results into another's.
    ids = [s.sampler_id for s in samplers]
    if len(set(ids)) != len(ids):
        raise ValueError(f"sampler ids must be unique, got {ids}")

    # One config dict for the whole run. The chunker's entry is restated at
    # the end, once its counters are final.
    config = {
        "decimator": decimator.config(),
        "chunker": chunker.config(),
        "samplers": [s.config() for s in samplers],
        "frame_store": (
            {**store.config(), "scope": store_scope} if store else None
        ),
    }

    # The caller assembles its own sinks; this knows only that a sink hears
    # three calls, and nothing at all about where any of them writes.
    if sink is not None:
        sink.begin(video_id, info.as_dict(), config)

    chunks: list[Chunk] = []
    current: Optional[Chunk] = None
    chunk_local_index = 0
    read = decimated = sampled = 0
    started = time.perf_counter()

    def stats() -> dict:
        return {
            "frames_read": read,
            "frames_decimated": decimated,
            "frames_sampled": sampled,
            "chunks": len(chunks),
            "elapsed_s": round(time.perf_counter() - started, 3),
            **({"stored_frames": store.written,
                "stored_mb": round(store.bytes_written / 1024 / 1024, 2)} if store else {}),
        }

    def close(chunk: Chunk) -> None:
        # The chunker only knows a chunk's end once the chunk is over, which
        # for scene cuts is not the same as when it was opened. The last
        # chunk has no next boundary at all, so its own last frame is the
        # only end it has -- and on a live source that stays true.
        _, end = chunker.bounds_of(chunk.chunk_id)
        chunk.end_ts = end if end is not None else chunk.last_ts
        chunks.append(chunk)
        if sink is not None:
            sink.chunk_closed(chunk.as_dict(), stats())

    for frame in read_frames(info):
        read += 1

        # Native rate, before decimation and before any release: a scene cut
        # is indistinguishable from ordinary motion once decimated to 1 fps.
        chunker.observe(frame)

        if decimator.accepts(frame):
            decimated += 1
            chunk_id = chunker.chunk_id_of(frame.media_ts)

            if current is None or chunk_id != current.chunk_id:
                if current is not None:
                    close(current)
                start, end = chunker.bounds_of(chunk_id)
                current = Chunk(chunk_id, start, end if end is not None else start)
                # Samplers forget everything at a boundary, so a chunk's
                # sampling never depends on the chunk before it.
                for sampler in samplers:
                    sampler.reset(chunk_id)
                chunk_local_index = 0

            current.decimated += 1
            current.last_ts = frame.media_ts
            # "decimated" keeps every frame the samplers were offered, so a
            # threshold can be retuned later without decoding the video again.
            if store is not None and store_scope == "decimated":
                store.write(frame.index, frame.image)
            for sampler in samplers:
                if not sampler.accepts(frame, chunk_local_index):
                    continue
                sampled += 1
                record = {
                    "index": frame.index,
                    "media_ts": round(frame.media_ts, 3),
                    "chunk_local_index": chunk_local_index,
                }
                # The address a fetcher can use. Seconds are a lossy rendering
                # of this; at a 1/1200000 timebase a rounded float lands on
                # the wrong frame.
                if frame.pts is not None:
                    record["pts"] = frame.pts
                if store is not None and store_scope == "sampled":
                    store.write(frame.index, frame.image)
                score = sampler.last_score()
                if score is not None:
                    record["score"] = round(score, 4)
                current.picks.setdefault(sampler.sampler_id, []).append(record)

            chunk_local_index += 1

        # A describer that wants the whole window at once would hold sampled
        # frames until the chunk closes. Nothing does yet, so pixels go back
        # immediately and peak memory stays at one frame.
        frame.release()

    if current is not None:
        close(current)

    # The last chunk ends where the video does, not where the boundary grid
    # would put it and not at its last decimated frame. Only a file knows this;
    # a live source keeps the last-frame answer set in close().
    # Not when the grid was handed in. A `fixed` run is honouring boundaries
    # another pass computed, and both of the corrections below would silently
    # edit them: the audio stream is routinely a few milliseconds shorter than
    # the video one (205.264 s against 205.280 s on the reference file), so
    # "correcting" the last chunk to the video's end makes the manifest and the
    # timeline disagree about a boundary neither of them chose. A supplied grid
    # has also already had its own minimum applied, and re-applying a different
    # one here would merge a tail the other pass deliberately kept.
    if chunking != "fixed" and chunks and info.frame_count and info.fps_trusted:
        duration = info.frame_count / info.fps
        if duration > chunks[-1].start_ts:
            chunks[-1].end_ts = duration
        # And once the true end is known, a stub of a last chunk is folded into
        # the one before it. Where the media stops relative to a grid that knew
        # nothing about it is arbitrary -- 97.99 s at 20 s leaves a usable
        # 17.99 s, 100.4 s leaves 0.4 s -- and a 0.4 s chunk still costs a
        # describer call per sampler, still keeps a frame because every chunk
        # keeps at least one, and is still a moment nobody can play.
        _merge_tail(chunks, chunk_duration_s * MIN_TAIL_FRACTION)

    elapsed = time.perf_counter() - started
    final = stats()
    # The final chunker counters (cuts seen, splits forced) are only known now.
    config["chunker"] = chunker.config()
    if sink is not None:
        # Re-state chunks so a late end_ts correction lands in the document.
        document = sink.finish(
            final, chunks=[c.as_dict() for c in chunks], config=config
        )
    else:
        # No sink: the document exists only in the returned Result. Same shape
        # a sink would have produced, so a caller cannot tell them apart.
        document = {
            "manifest_version": MANIFEST_VERSION,
            "video_id": video_id,
            "complete": True,
            "source": info.as_dict(),
            "config": config,
            "stats": final,
            "chunks": [c.as_dict() for c in chunks],
        }
    return Result(document, chunks, read, decimated, sampled, elapsed)
