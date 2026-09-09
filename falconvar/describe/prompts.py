"""What to ask about a chunk, and what shape to expect back.

Logic only. The instructions, response schemas and shapes are data, in
`prompts.json` beside this file and `data/prompts.json` for anything added
since -- `library.py` merges the two.

A question resolves to a **shape**, and the shape decides the schema. Nothing
else does: a (sampler, question) pairing is independent of every other pairing
on the chunk, so `schema_for` is a function of the question alone and two
questions that share a field both answer it.

Any sampler may be paired with any question. The vocabulary is validated before
a run rather than fallen through: an unknown name would otherwise quietly get
the general question and bill for it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from . import library


def __getattr__(name: str) -> Any:
    """`SYSTEM` reads through to the data, so call sites stay unchanged.

    It is one string prepended to every request, and it lives in the same file
    as everything else it is versioned with.
    """
    if name == "SYSTEM":
        return library.load()["system"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def questions() -> list[str]:
    """The valid question names, for a CLI to validate against and an API to
    publish. Read from the data, so adding a question adds it here."""
    return library.questions()


def schema_for(question: str) -> dict[str, Any]:
    """The strict response schema for one call. A function of the question.

    Every property is required and `additionalProperties` is false, so the
    model answers all of them: a strict schema is a guarantee where a prompt
    saying "focus on X" is only a request.
    """
    shape = library.shape_of(question)
    fields = dict(shape.get("fields") or {})
    summaries = library.load()["summaries"]
    summary = summaries.get(shape.get("summary", "standard")) or summaries["standard"]
    properties = {"summary": summary, **fields}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def version_of(question: str) -> str:
    """A hash of one question: its instruction, its shape, and the preamble.

    Recorded per question so that resume treats a prompt change the way it
    treats a model change, **without** an edit to one question invalidating
    answers given to another. A single hash over the whole vocabulary made
    adding a question re-describe every chunk of every video -- a bill for work
    already done, paid silently because the output looks the same either way.

    The system preamble is folded into every question's hash because it is
    prepended to every request: changing it does change every answer.
    """
    payload = json.dumps({
        "system": library.load()["system"],
        "instruction": library.instruction_of(question),
        "shape": library.shape_of(question),
        "summaries": library.load()["summaries"],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def versions(names: Sequence[str]) -> dict[str, str]:
    """`{question: hash}` for the questions a run actually asks."""
    return {name: version_of(name) for name in sorted(set(names))}


def question_for(context: dict[str, Any]) -> str:
    """Which question these frames are for.

    The sampler id is the default, and normally the right answer: the manifest
    records *why* a frame was kept, and why it was kept is the best guide to
    what to ask about it.

    But a sampler may name a question in its own config, and that is what lets
    a positional sampler run any question on a stride -- `uniform` keeps every
    Nth decimated frame and says which question they are for, without having to
    be the sampler that would normally ask it. Reaching for the sampler *class*
    instead would mean constructing an EasyOCR or CLIP object purely to borrow
    its name, and would never extend to a question no sampler corresponds to.

    Read from the manifest rather than from a table, so a run is reproducible
    from the document it produced -- and so the choice sits inside
    `manifest_fingerprint`, which means changing it correctly invalidates
    describe's resume. A question absent from the vocabulary falls back to the
    general one, exactly as an unregistered sampler does.
    """
    config = context.get("sampler_config") or {}
    return config.get("prompt") or context.get("sampler") or ""


def questions_in(manifest) -> list[str]:
    """Every question this manifest asks, from its sampler config.

    Read off the config rather than by walking chunks: the samplers are
    declared once, and this only needs to know which questions exist so their
    hashes can go in the resume key.
    """
    return sorted({question_for({"sampler": s.get("id", ""), "sampler_config": s})
                   for s in manifest.config.get("samplers", [])})


def span_of(context: dict[str, Any]) -> str:
    return f"{context['start_ts']:.1f}s to {context['end_ts']:.1f}s"


def vocabulary_of(context: dict[str, Any]) -> str:
    """The open-vocabulary list the detector was given, as prose."""
    detector = (context.get("sampler_config") or {}).get("detector") or {}
    words = detector.get("vocabulary") or []
    return ", ".join(words) if words else "no vocabulary recorded"


def for_sampler(question: str, context: dict[str, Any], frame_count: int) -> str:
    """The instruction for one (chunk, sampler) call, given its question.

    No "focus on X" line: the schema this sampler is given has no field for
    anything else, which enforces what a sentence could only ask for.
    """
    return library.instruction_of(question).format(
        n=frame_count,
        span=span_of(context),
        vocabulary=vocabulary_of(context),
    )


def frame_label(index: int, media_ts: float, position: int, total: int) -> str:
    """What precedes each image, so the model can order and refer to them.

    Timestamps rather than "image 1, image 2": the gaps between sampled frames
    are uneven by design, and a model told only the ordering will assume they
    are evenly spaced and narrate a smooth progression that did not happen.
    """
    return f"Frame {position} of {total} -- t={media_ts:.2f}s (index {index}):"
