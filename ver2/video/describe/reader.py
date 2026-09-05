"""The loop: manifest in, descriptions out.

One pass over the chunks, and inside each chunk one call per sampler. The
sampler is the unit because a person detector's picks and a scene-change
detector's picks are answers to different questions -- what the frames show is
only half of it; which question they were an answer to is the other half, and
the manifest is the only thing that knows.

Frames are read once per chunk regardless. Samplers overlap heavily -- 25 of
80 frames on the reference run were chosen by two of them -- so the pixels are
cached by index for as long as the chunk is open and released at its boundary.
The model sees a shared frame twice, which is the point; the disk does not.

Nothing here decides where descriptions go, or what a description is. It reads
frames, calls a describer, hands the result to a sink, and keeps count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

from .describers import Describer
from .input import FrameSource
from .output import DescriptionSink
# A leaf: prompts.py imports nothing. Resolving each chunk's questions here
# rather than in a describer keeps every describer seeing the same answer.
from .vlm import prompts


@dataclass
class Result:
    document: dict
    chunks_seen: int
    described: int
    skipped: int
    frames: dict
    elapsed_s: float


def chunks_of(manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """The manifest's chunks, in chunk order. The finished-ingest case."""
    yield from sorted(manifest["chunks"], key=lambda c: c["chunk_id"])


def describe(
    manifest: dict[str, Any],
    describer: Describer,
    sink: Optional[DescriptionSink] = None,
    samplers: Optional[Sequence[str]] = None,
    feed: Optional[Iterator[dict[str, Any]]] = None,
    limit: Optional[int] = None,
    on_described: Optional[Any] = None,
) -> Result:
    """Describe every (chunk, sampler) the manifest names.

    ``feed`` supplies the chunks. Defaulting to the manifest's own list is the
    batch case; a follower passes a generator that yields chunks as they land
    and returns when ingest reports itself complete. The loop cannot tell the
    difference, which is the reason following costs nothing structurally.
    """
    video_id = manifest["video_id"]
    model = describer.config()
    started = time.perf_counter()

    # Before anything is read or written, and before a single call is paid for.
    # A manifest naming a question this module does not have would otherwise
    # fall through to the scene question and produce a complete-looking
    # document answering something nobody asked -- the manifest is written by
    # a stage that cannot see this vocabulary, so it is checked on arrival.
    unknown = sorted({q for entry in (manifest.get("config") or {}).get("samplers", [])
                      if (q := entry.get("prompt")) and q not in prompts.QUESTIONS})
    if unknown:
        raise ValueError(
            f"manifest names unknown question(s) {', '.join(unknown)}; "
            f"known: {', '.join(prompts.QUESTIONS)}")

    # Before the sink, so a missing or incomplete frame store fails without
    # having written anything. A run that cannot read pixels should leave no
    # document behind claiming it tried.
    source = FrameSource(manifest)

    already: set[tuple[int, str]] = set()
    if sink is not None:
        sink.begin(video_id, manifest, model)
        already = sink.existing()

    described = skipped = chunks_seen = 0
    with source:
        for chunk in (feed if feed is not None else chunks_of(manifest)):
            chunks_seen += 1
            wanted = [s for s in chunk["samplers"]
                      if samplers is None or s in samplers]
            for sampler in wanted:
                if (chunk["chunk_id"], sampler) in already:
                    skipped += 1
                    continue
                images = source.images_for(chunk, sampler)
                context = {
                    "video_id": video_id,
                    "chunk_id": chunk["chunk_id"],
                    "start_ts": chunk["start_ts"],
                    "end_ts": chunk["end_ts"],
                    "sampler": sampler,
                    "sampler_config": _config_for(manifest, sampler),
                    # What else is being *asked* about this chunk. A general
                    # describer gives up the fields a specialist here owns,
                    # rather than answering them a second time.
                    #
                    # Questions, not sampler ids. Ownership in `prompts.OWNER`
                    # is keyed by question, and the two are only equal while no
                    # sampler is paired with someone else's question. With
                    # `yolo:overview` on the chunk, passing ids would have
                    # `clip` drop `people` for a call that was asked `overview`
                    # and owns no keys at all -- the field would leave the
                    # document with nothing malformed to notice.
                    "chunk_questions": _questions_on(manifest, chunk),
                }
                call_started = time.perf_counter()
                answer = describer.describe(images, context)
                record = {
                    "chunk_id": chunk["chunk_id"],
                    "start_ts": chunk["start_ts"],
                    "end_ts": chunk["end_ts"],
                    "sampler": sampler,
                    "frame_count": len(images),
                    "frame_indexes": [f.index for f in images],
                    "description": answer.summary,
                    "structured": answer.fields,
                    "elapsed_s": round(time.perf_counter() - call_started, 3),
                }
                if sink is not None:
                    sink.described(record)
                if on_described is not None:
                    on_described(record)
                described += 1
                if limit is not None and described >= limit:
                    break
            # The chunk is finished with; its pixels are not needed again.
            source.release()
            if limit is not None and described >= limit:
                break
        frames = source.stats.as_dict()

    elapsed = time.perf_counter() - started
    stats = {
        "chunks": chunks_seen,
        "described": described,
        "skipped": skipped,
        "elapsed_s": round(elapsed, 3),
        **frames,
    }
    document = sink.finish(stats) if sink is not None else {"stats": stats}
    return Result(document, chunks_seen, described, skipped, frames, elapsed)


def _questions_on(manifest: dict[str, Any], chunk: dict[str, Any]) -> list[str]:
    """Every question asked about this chunk, resolved the same way each call
    resolves its own -- so what a schema gives up and what is actually asked
    can never disagree."""
    return [prompts.question_for({"sampler": sampler,
                                  "sampler_config": _config_for(manifest, sampler)})
            for sampler in chunk["samplers"]]


def _config_for(manifest: dict[str, Any], sampler: str) -> dict:
    """The sampler's own settings, so a prompt can say what it was looking for."""
    for entry in (manifest.get("config") or {}).get("samplers", []):
        if entry.get("id") == sampler:
            return entry
    return {}
