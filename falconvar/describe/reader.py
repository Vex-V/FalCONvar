"""`describe()` -- one call per (chunk, sampler).

**Ownership is resolved over questions, never sampler ids.** Which keys a
call's schema may fill is narrowed by the other questions asked about the same
chunk, so exactly one call answers each key and merging is a plain union.
`prompts.OWNER` is keyed by question, and the two are only equal while no
sampler is paired with someone else's question -- with `yolo:overview` present,
passing ids would have `clip` give up `people` to a call whose schema owns
nothing, and the field would leave the document with everything still
well-formed.

**Resume is keyed on the manifest, the describer, and the prompts.** A stored
description counts as done only if all three match. Without the model check,
describing with the stub and then switching to a real one skips every pair and
reports success having done nothing -- the most expensive kind of silent no-op,
since the output looks complete. The `model` block therefore carries a hash of
every instruction and schema in `prompts.py`: editing a prompt changes the
output but not the model id.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

from ..shared.documents import Descriptions, Manifest, Timeline
from . import prompts
from .base import Describer, Description
from .frames import FrameSource


def _model_block(describer: Describer) -> dict[str, Any]:
    return {**describer.config(), "prompts": prompts.version()}


def describe(manifest: Manifest, timeline: Timeline, describer: Describer,
             source: FrameSource,
             samplers: Optional[Sequence[str]] = None,
             existing: Optional[Descriptions] = None,
             limit: Optional[int] = None,
             on_described: Optional[Callable[[int, str, Description], None]] = None
             ) -> Descriptions:
    """Describe every (chunk, sampler) the manifest names."""
    model = _model_block(describer)
    started = time.perf_counter()

    # A stored answer counts only if the manifest AND the model block match.
    done: set[tuple[int, str]] = set()
    kept: dict[int, dict[str, Any]] = {}
    if existing is not None:
        same_manifest = existing.manifest_fingerprint == manifest.fingerprint()
        same_model = existing.model == model
        if same_manifest and same_model:
            done = existing.done()
            kept = {c["chunk_id"]: c for c in existing.chunks}

    chunks: list[dict[str, Any]] = []
    described = skipped = 0

    for chunk in manifest.chunks:
        chunk_id = chunk["chunk_id"]
        start_ts, end_ts = timeline.bounds_of(chunk_id)
        # Resolved once per chunk, and this is the list every call's schema is
        # narrowed against.
        questions = prompts.questions_on(manifest, chunk)
        by_id = {s["id"]: s for s in manifest.config.get("samplers", [])}

        out = {"chunk_id": chunk_id, "samplers": {}}
        previous = kept.get(chunk_id, {}).get("samplers", {})

        for sampler_id in chunk.get("samplers", {}):
            if samplers is not None and sampler_id not in samplers:
                continue
            if (chunk_id, sampler_id) in done:
                out["samplers"][sampler_id] = previous[sampler_id]
                skipped += 1
                continue
            if limit is not None and described >= limit:
                continue

            images = source.images_for(chunk_id, sampler_id)
            if not images:
                continue
            context = {
                "video_id": manifest.video_id,
                "chunk_id": chunk_id,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "sampler": sampler_id,
                "sampler_config": by_id.get(sampler_id, {}),
                "chunk_questions": questions,
            }
            call_started = time.perf_counter()
            answer = describer.describe(images, context)
            described += 1
            out["samplers"][sampler_id] = {
                "question": prompts.question_for(context),
                "frame_count": len(images),
                "frame_indexes": [f.index for f in images],
                "description": answer.summary,
                "structured": answer.fields,
                "elapsed_s": round(time.perf_counter() - call_started, 3),
            }
            if on_described is not None:
                on_described(chunk_id, sampler_id, answer)

        # Narrowing means the keys are already disjoint, so this is a union.
        out["structured"] = prompts.merge(
            {sid: block.get("structured", {})
             for sid, block in out["samplers"].items()})
        chunks.append(out)

    return Descriptions(
        video_id=manifest.video_id,
        timeline_fingerprint=manifest.timeline_fingerprint,
        manifest_fingerprint=manifest.fingerprint(),
        model=model,
        chunks=chunks,
        stats={"described": described, "skipped": skipped,
               "chunks": len(chunks),
               "elapsed_s": round(time.perf_counter() - started, 3)},
    )


__all__ = ["describe"]
