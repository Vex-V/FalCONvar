"""`describe()` -- one call per (chunk, sampler).

**Ownership is resolved over questions, never sampler ids.** Which keys a
call's schema may fill is narrowed by the other questions asked about the same
chunk, so exactly one call answers each key and merging is a plain union.
The owner map is keyed by question, and the two are only equal while no
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


def _model_block(describer: Describer, questions: Sequence[str]) -> dict[str, Any]:
    """What a stored answer must match to count as done.

    `prompts` is a map, `{question: hash}`, not one hash over the vocabulary.
    A single hash meant that adding a question -- which cannot change what any
    existing answer should say -- invalidated every description of every video,
    and the next run silently paid to rebuild them all.
    """
    return {**describer.config(), "prompts": prompts.versions(questions)}


def _resumable(existing: Optional[Descriptions], manifest: Manifest,
               model: dict[str, Any]) -> set[tuple[int, str]]:
    """The (chunk, sampler) pairs whose stored answer is still current.

    Three conditions, checked separately because they fail for different
    reasons: the manifest must be the one that chose these frames, the
    describer must be the same model, and **each pair's own question** must
    hash the same as it does now. The third is per pair, which is the whole
    point -- editing the `text` instruction re-describes `uniform:text` and
    leaves `clip` alone.

    A stored `prompts` that is a bare string is the pre-map format. Its hash
    covered the whole vocabulary and cannot be reduced to a per-question one,
    so every pair is re-described once. Keeping them instead would mean
    claiming a provenance this cannot check.
    """
    if existing is None:
        return set()
    if existing.manifest_fingerprint != manifest.fingerprint():
        return set()

    def describer_half(block: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in (block or {}).items() if k != "prompts"}

    if describer_half(existing.model) != describer_half(model):
        return set()

    stored = (existing.model or {}).get("prompts")
    if not isinstance(stored, dict):
        return set()
    current = model.get("prompts") or {}

    return {(chunk["chunk_id"], sampler_id)
            for chunk in existing.chunks
            for sampler_id, block in (chunk.get("samplers") or {}).items()
            if (q := block.get("question")) is not None
            and stored.get(q) is not None and stored.get(q) == current.get(q)}


def describe(manifest: Manifest, timeline: Timeline, describer: Describer,
             source: FrameSource,
             samplers: Optional[Sequence[str]] = None,
             existing: Optional[Descriptions] = None,
             limit: Optional[int] = None,
             on_described: Optional[Callable[[int, str, Description], None]] = None
             ) -> Descriptions:
    """Describe every (chunk, sampler) the manifest names."""
    # Every question this manifest asks, resolved before the first call so the
    # model block is complete whether or not a chunk is reached.
    asked = sorted({q for chunk in manifest.chunks
                    for q in prompts.questions_on(manifest, chunk)})
    model = _model_block(describer, asked)
    started = time.perf_counter()

    done = _resumable(existing, manifest, model)
    kept: dict[int, dict[str, Any]] = (
        {c["chunk_id"]: c for c in existing.chunks} if existing is not None else {})

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
