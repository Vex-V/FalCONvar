"""What to ask about a chunk, given which sampler chose its frames.

The manifest records *why* each frame was kept, and that is the most useful
thing it knows. A person-change sampler fired because the people in shot
changed; a scene-change sampler fired because the whole frame did; a text
sampler fired because the writing on screen changed. Asking the same question
of all of them throws that away and gets back the same generic caption three
times, which is worse than useless in a retrieval index -- three near-identical
embeddings for one moment.

Two kinds of question, and the difference matters:

**The scene question is general.** It covers setting, people, objects, actions
and what changed, and it is the fallback: `clip` asks it, `uniform` asks it,
and so does any sampler with no entry of its own. A run with only a uniform
sampler still produces a usable record rather than a stub.

**A specialist question is narrow and structured.** The people sampler returns
one object *per person* -- appearance, clothing, role, action -- because a flat
list of people and a flat list of actions does not say who is doing what, and
that relation is most of the value. Same for objects and for text.

**Which keys the scene question may fill depends on who else ran.** If the
people sampler is also on this chunk, the scene schema has no `people` field at
all, so the scene call cannot spend tokens on -- or disagree about -- something
already being answered better elsewhere. Exclusive ownership, decided per
chunk, is what lets a chunk's answers be merged with nothing to reconcile.

Prompts and schemas are data. A new sampler adds an entry; an unknown one falls
back to the scene question. Nothing here imports anything.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

#: Prepended to every request. Kept short -- the per-sampler prompt carries the
#: actual instruction, and a long preamble competes with it.
SYSTEM = (
    "You describe frames sampled from surveillance and video footage for a "
    "retrieval index. Report only what is visible. Do not speculate about "
    "intent, identity, or anything outside the frame. If the frames are "
    "ambiguous or too dark to read, say so plainly instead of guessing. The "
    "summary field is prose with no preamble; the other fields are terse."
)

SCENE = (
    "These {n} frames span {span}. Describe the segment as a whole: where it "
    "is, who and what is present, what happens, and what changes between "
    "frames."
)

BY_SAMPLER: dict[str, str] = {
    # Positional, not content-driven: it fired on a clock, so it is the one
    # sampler whose frames carry no claim of having changed.
    "uniform": (
        "These {n} frames are an even time-sampling of {span}, taken on a "
        "fixed interval rather than because anything changed. Describe the "
        "segment as a whole, and say plainly if nothing changes across it."
    ),
    # Whole-frame appearance change, above a CLIP similarity threshold.
    "clip": (
        "These {n} frames span {span} and were kept because the scene's "
        "overall appearance changed between them. Describe the segment, then "
        "what changed from frame to frame -- camera view, lighting, layout, "
        "who or what entered or left."
    ),
    # PersonChangeSampler: YOLO restricted to COCO class 0.
    "yolo": (
        "These {n} frames span {span} and were kept because the people in "
        "shot changed -- their number, position, or arrangement. Give one "
        "entry per distinct person, and track each across the frames rather "
        "than listing them again per frame. Appearance and clothing are what "
        "make a person findable again, so be specific about both. Never guess "
        "at a name, an age or an identity."
    ),
    # Open-vocabulary detection: the vocabulary is the whole point.
    "objects": (
        "These {n} frames span {span} and were kept because the objects in "
        "shot changed. The detector was looking for: {vocabulary}. Give one "
        "entry per distinct object, leading with those, and say what each is "
        "being used for and by whom."
    ),
    # Deliberately unstructured. Not a fallback and not a narrowed scene
    # question -- a short prose answer and nothing else, for when the whole
    # apparatus of fields is more than the moment is worth.
    "overview": (
        "These {n} frames span {span}. Say what is happening, in 4 to 5 "
        "sentences of plain prose. Do not list, do not label, do not itemise."
    ),
    # Text regions changed; the value is the words, verbatim.
    "text": (
        "These {n} frames span {span} and were kept because the text on "
        "screen changed. Give one entry per distinct piece of text, "
        "transcribed exactly as written including case and punctuation, with "
        "where it appears. Mark anything unreadable as [unreadable] rather "
        "than guessing at it."
    ),
}


# --------------------------------------------------------------------------- #
# the response shape
# --------------------------------------------------------------------------- #

#: Prose, always required. It is what a person reads, what gets embedded and
#: what full-text search indexes -- and the only field where relations survive.
#: A list of people and a list of actions does not say who did what; a sentence
#: does, which is why the summary is the text retrieval runs on.
#:
#: **The length instruction is quantitative because a qualitative one failed.**
#: "several sentences" produced 479-495 characters once the structured fields
#: arrived, against 1185-2577 in the prose-only era -- the model sizes the
#: summary in proportion to how many other fields it has been asked for, and
#: 480 characters genuinely is several sentences. A word count is something it
#: can check itself, and saying the fields *index* rather than replace the
#: summary reverses the assumption that made it economise.
SUMMARY = {
    "type": "string",
    "description": (
        "The full description of this segment as prose, at least 150 words. "
        "This is the text the search index is built from, so it must contain "
        "everything worth finding: the other fields index this text, they do "
        "not replace it. Anything you put in them should also be said here, "
        "in sentences, with who is doing what."
    ),
}

#: `overview` overrides the length instruction above, and the override is the
#: point of the sampler. SUMMARY asks for at least 150 words because it is the
#: only text that gets embedded and its length is a retrieval parameter; an
#: overview is asked for when a paragraph is more than the moment deserves,
#: so demanding 150 words of it would defeat the reason for choosing it.
OVERVIEW_SUMMARY = {
    "type": "string",
    "description": (
        "What is happening here, in 4 to 5 sentences of plain prose. No "
        "preamble, no lists, no labels -- just the description."
    ),
}

#: Per-sampler overrides of the summary field. Everything absent gets SUMMARY.
SUMMARY_BY_SAMPLER: dict[str, Any] = {"overview": OVERVIEW_SUMMARY}

#: The general question's fields. `people`, `objects` and `visible_text` are
#: coarse here -- plain phrases -- because a specialist answers them properly
#: when one is present, and this is the fallback for when none is.
SCENE_FIELDS: dict[str, Any] = {
    "setting": {
        "type": "string",
        "description": "Where this takes place: place, camera position, lighting.",
    },
    "people": {
        "type": "array", "items": {"type": "string"},
        "description": ("Each person visible, by appearance only -- 'woman in a "
                        "grey sweater'. Never a name or an identity."),
    },
    "objects": {
        "type": "array", "items": {"type": "string"},
        "description": "Salient objects visible, as short noun phrases.",
    },
    "visible_text": {
        "type": "array", "items": {"type": "string"},
        "description": ("Text legible in the frames, transcribed verbatim. "
                        "Empty if there is none."),
    },
    "actions": {
        "type": "array", "items": {"type": "string"},
        "description": "Discrete actions occurring across the frames.",
    },
    "changes": {
        "type": "array", "items": {"type": "string"},
        "description": ("What differs across the frames, in order. Empty if the "
                        "window is static."),
    },
    "tags": {
        "type": "array", "items": {"type": "string"},
        "description": "Short keywords for filtering, lowercase.",
    },
}

#: Specialists. Each returns objects rather than strings, because the binding
#: between an entity and what it did is the part a flat list destroys -- and
#: recovering it later is impossible, not merely expensive.
SPECIALIST_FIELDS: dict[str, dict[str, Any]] = {
    "yolo": {
        "people": {
            "type": "array",
            "description": "One entry per distinct person visible.",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["appearance", "clothing", "role", "action"],
                "properties": {
                    "appearance": {
                        "type": "string",
                        "description": ("Build, hair and other lasting physical "
                                        "features. No names, no identity."),
                    },
                    "clothing": {
                        "type": "string",
                        "description": ("Every visible garment with its colour -- "
                                        "the most reliable way to recognise the "
                                        "same person again."),
                    },
                    "role": {
                        "type": "string",
                        "description": ("Apparent role from context alone: "
                                        "'cashier', 'customer', 'passer-by'."),
                    },
                    "action": {
                        "type": "string",
                        "description": ("What this person does across these "
                                        "frames, including movement."),
                    },
                },
            },
        },
    },
    "objects": {
        "objects": {
            "type": "array",
            "description": "One entry per distinct object worth indexing.",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["object", "appearance", "context"],
                "properties": {
                    "object": {
                        "type": "string",
                        "description": "Short name, e.g. 'shopping basket'.",
                    },
                    "appearance": {
                        "type": "string",
                        "description": "Colour, material, size, state.",
                    },
                    "context": {
                        "type": "string",
                        "description": ("What it is doing or being used for here, "
                                        "and by whom."),
                    },
                },
            },
        },
    },
    # Empty on purpose: `overview` is a specialist in the sense that it has its
    # own schema, and that schema is `summary` alone. Because it owns no keys
    # it also takes none away from the scene question, so running it beside
    # `clip` costs the scene answer nothing.
    "overview": {},
    "text": {
        "visible_text": {
            "type": "array",
            "description": "One entry per distinct piece of text visible.",
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["text", "context"],
                "properties": {
                    "text": {
                        "type": "string",
                        "description": ("Exactly as written, including case and "
                                        "punctuation."),
                    },
                    "context": {
                        "type": "string",
                        "description": ("What this text is and where it appears, "
                                        "e.g. 'a label on a cupboard'."),
                    },
                },
            },
        },
    },
}

#: Which sampler owns a key outright when it is present on the same chunk. The
#: scene question gives these up rather than answering them a second time.
OWNER: dict[str, str] = {
    key: sampler
    for sampler, fields in SPECIALIST_FIELDS.items()
    for key in fields
}


def owned_by(sampler: str, siblings: Sequence[str] = ()) -> list[str]:
    """The keys this sampler fills on a chunk where ``siblings`` also ran."""
    if sampler in SPECIALIST_FIELDS:
        return list(SPECIALIST_FIELDS[sampler])
    taken = {key for key, owner in OWNER.items()
             if owner in siblings and owner != sampler}
    return [key for key in SCENE_FIELDS if key not in taken]


def schema_for(sampler: str, siblings: Sequence[str] = ()) -> dict[str, Any]:
    """The strict response schema for one call.

    ``siblings`` is every sampler on this chunk. It narrows the scene schema:
    a field a specialist is answering is removed rather than asked for twice.
    """
    if sampler in SPECIALIST_FIELDS:
        fields = SPECIALIST_FIELDS[sampler]
    else:
        keep = owned_by(sampler, siblings)
        fields = {key: SCENE_FIELDS[key] for key in keep}
    properties = {"summary": SUMMARY_BY_SAMPLER.get(sampler, SUMMARY), **fields}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def merge(structured_by_sampler: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """One chunk's answers, flattened into a single record.

    Narrowing means the keys are already disjoint, so this is a plain union.
    The specialist-wins rule below only matters for documents written before a
    schema changed, or where a run was assembled from two different sets of
    samplers.
    """
    merged: dict[str, Any] = {}
    for sampler, fields in sorted(structured_by_sampler.items()):
        for key, value in (fields or {}).items():
            if key in merged and OWNER.get(key) != sampler:
                continue                    # a specialist already answered it
            merged[key] = value
    return merged


def version() -> str:
    """A hash of every instruction and schema in this module.

    Recorded in the describer's config so that resume treats a prompt change
    the way it treats a model change. Without it, editing a prompt and
    re-running reports "10 skipped, already described" and does nothing --
    which is exactly what happened while measuring the summary-length fix, and
    is indistinguishable from success.
    """
    import hashlib
    import json as _json

    payload = _json.dumps({
        "system": SYSTEM,
        "scene": SCENE,
        "by_sampler": BY_SAMPLER,
        "summary": SUMMARY,
        "summary_by_sampler": SUMMARY_BY_SAMPLER,
        "scene_fields": SCENE_FIELDS,
        "specialist_fields": SPECIALIST_FIELDS,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def question_for(context: dict[str, Any]) -> str:
    """Which question these frames are for.

    The sampler id is the default, and normally the right answer: the manifest
    records *why* a frame was kept, and why it was kept is the best guide to
    what to ask about it.

    But a sampler may name a question in its own config, and that is what lets
    a positional sampler run any question on a clock -- `uniform` keeps a frame
    every N seconds and says which question they are for, without having to be
    the sampler that would normally ask it. Reaching for the sampler *class*
    instead would mean constructing an EasyOCR or CLIP object purely to borrow
    its name, and would never extend to a question no sampler corresponds to.

    Read from the manifest rather than from a table, so a run is reproducible
    from the document it produced -- and so the choice sits inside
    `manifest_fingerprint`, which means changing it correctly invalidates
    describe's resume. A question named here but absent from BY_SAMPLER falls
    back to the scene question, exactly as an unregistered sampler does.
    """
    config = context.get("sampler_config") or {}
    return config.get("prompt") or context.get("sampler") or ""


def span_of(context: dict[str, Any]) -> str:
    return f"{context['start_ts']:.1f}s to {context['end_ts']:.1f}s"


def vocabulary_of(context: dict[str, Any]) -> str:
    """The open-vocabulary list the detector was given, as prose."""
    detector = (context.get("sampler_config") or {}).get("detector") or {}
    words = detector.get("vocabulary") or []
    return ", ".join(words) if words else "no vocabulary recorded"


def for_sampler(sampler: str, context: dict[str, Any], frame_count: int) -> str:
    """The instruction for one (chunk, sampler) call.

    No "focus on X" line: the schema this sampler is given has no field for
    anything else, which enforces what a sentence could only ask for.
    """
    template = BY_SAMPLER.get(sampler, SCENE)
    return template.format(
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
