"""`describe()` -- one call per (chunk, sampler).

**Every (chunk, sampler) pairing is independent.** A call's schema is a
function of its question alone, so two questions that share a field both answer
it and both answers are kept under their own sampler id. Nothing is reconciled
and nothing is dropped.

**Resume is keyed on the manifest, the describer, and the prompts.** A stored
description counts as done only if all three match. Without the describer
check, running with the stub and then switching to a real one skips every pair
and reports success having done nothing -- the most expensive kind of silent
no-op, since the output looks complete. The prompts need their own key for the
same reason: editing an instruction changes the answer but not the model id.

**The prompt key is a hash per question, not one over the vocabulary.** A
single hash also invalidated answers that could not have changed, so adding a
question re-described every chunk of every video and the next run silently paid
for it. `model.prompts` is `{question: hash}` and `_resumable` compares each
pair against its own question.
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
    model = _model_block(describer, prompts.questions_in(manifest))
    started = time.perf_counter()

    done = _resumable(existing, manifest, model)
    kept: dict[int, dict[str, Any]] = (
        {c["chunk_id"]: c for c in existing.chunks} if existing is not None else {})

    chunks: list[dict[str, Any]] = []
    described = skipped = 0

    for chunk in manifest.chunks:
        chunk_id = chunk["chunk_id"]
        start_ts, end_ts = timeline.bounds_of(chunk_id)
        by_id = {s["id"]: s for s in manifest.config.get("samplers", [])}

        out = {"chunk_id": chunk_id, "samplers": {}}
        previous = kept.get(chunk_id, {}).get("samplers", {})

        # One run of a sampler, one set of frames, one call per question asked
        # about them. The frames are read once however many questions there are.
        for run_id in chunk.get("samplers", {}):
            config = by_id.get(run_id, {})
            name = config.get("name") or run_id
            images = None

            for question in prompts.questions_of(config, run_id):
                sampler_id = prompts.answer_id(name, question)
                if samplers is not None and sampler_id not in samplers:
                    continue
                if (chunk_id, sampler_id) in done:
                    out["samplers"][sampler_id] = previous[sampler_id]
                    skipped += 1
                    continue
                if limit is not None and described >= limit:
                    continue

                if images is None:
                    images = source.images_for(chunk_id, run_id)
                if not images:
                    break

                context = {
                    "video_id": manifest.video_id,
                    "chunk_id": chunk_id,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "sampler": sampler_id,
                    "question": question,
                    "sampler_config": config,
                }
                call_started = time.perf_counter()
                answer = describer.describe(images, context)
                described += 1
                out["samplers"][sampler_id] = {
                    "question": question,
                    "frame_count": len(images),
                    "frame_indexes": [f.index for f in images],
                    "description": answer.summary,
                    "structured": answer.fields,
                    "elapsed_s": round(time.perf_counter() - call_started, 3),
                }
                if on_described is not None:
                    on_described(chunk_id, sampler_id, answer)

        # No chunk-level rollup. It flattened every sampler's answer into one
        # record, which forced a rule for who wins a shared key -- and there is
        # no non-arbitrary one, because the user asked both questions. Nothing
        # read it: units, both index writers and every aggregator work from the
        # per-sampler blocks below.
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
