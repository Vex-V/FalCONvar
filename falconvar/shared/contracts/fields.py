"""The field builder: a compact field spec in, JSON Schema out.

`{"severity": {"type": "text", "one_of": [...]}}` becomes the schema a
structured model call is sent. Generated rather than accepted: the call goes
out with `strict: true`, whose subset is narrow, and a raw schema arriving over
HTTP could express something the API refuses -- failing *after* the frames are
read, with the call about to be paid for.

**Shared because an answer is not only a describe answer.** A describe shape
and an aggregate prompt both declare their fields this way and both compile
through here, which is why `check_fields` was already written apart from
`check_shape`. Keeping it in `describe` meant `aggregates` reached a video_rag
component to build its own schemas.

The caps travel with the checker that enforces them: structured answers run
~3x longer than prose, and the `people` schema truncated into unparseable JSON
at 700 output tokens.
"""

from __future__ import annotations

import re
from typing import Any


#: A field name has to survive being a JSON Schema property, a `jsonb` key and
#: a named part of a rendered unit, so it is as narrow as a question name.
FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


#: What a custom field may be. Two primitives; `of` turns a list into a list of
#: objects. That is the whole range the built-in shapes already span -- `scene`
#: is flat lists, `people`/`objects`/`text` are the nested form -- so a custom
#: shape is the shipped vocabulary exposed, not a new mechanism beside it.
FIELD_TYPES = ("text", "list")


#: Caps. Structured answers run ~3x longer than prose and the `people` schema
#: already truncated mid-string at `max_output_tokens=700`, coming back as
#: unparseable JSON. An unbounded shape is a paid call for an answer that
#: cannot be read, so the limit is refused at write time rather than
#: discovered at call time.
MAX_FIELDS = 12


MAX_NESTED_KEYS = 8


MAX_ENUM = 24


def _compile_field(spec: dict[str, Any]) -> dict[str, Any]:
    """One field spec -> the JSON Schema fragment a call will be sent.

    Generated rather than accepted. The schema goes to the API with
    `strict: true`, and that subset is narrow -- every property required,
    `additionalProperties` false, no unions, a shallow nesting cap. A builder
    cannot express something the API would refuse; a raw schema arriving over
    HTTP can, and would fail *after* the frames are read with the call about to
    be paid for. That is the same failure `check` already prevents for a
    `{typo}` placeholder.
    """
    about = str(spec.get("about") or "").strip()
    one_of = list(spec.get("one_of") or [])

    if spec.get("type") == "text":
        leaf: dict[str, Any] = {"type": "string", "description": about}
        if one_of:
            leaf["enum"] = one_of
        return leaf

    nested = spec.get("of")
    if nested:
        keys = list(nested)
        # One bound object per entity, never parallel lists: a list of people
        # beside a list of actions does not say who did what, and cannot be
        # made to afterwards.
        return {
            "type": "array", "description": about,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": keys,
                "properties": {k: {"type": "string",
                                   "description": str(nested[k]).strip()}
                               for k in keys},
            },
        }

    items: dict[str, Any] = {"type": "string"}
    if one_of:
        items["enum"] = one_of
    return {"type": "array", "description": about, "items": items}


def compile_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """`{name: JSON Schema fragment}`, in the order the fields were written."""
    return {name: _compile_field(f) for name, f in fields.items()}


def check_fields(fields: dict[str, Any]) -> list[str]:
    """Everything wrong with a field builder, as messages.

    Apart from `check_shape` because a describe shape is not the only answer
    built from fields: an aggregate prompt's is too, with no prose summary.
    """
    problems: list[str] = []
    if len(fields) > MAX_FIELDS:
        problems.append(f"{len(fields)} fields; {MAX_FIELDS} max -- a longer "
                        "answer truncates rather than failing")

    for name, field in fields.items():
        where = f"field {name!r}"
        if not FIELD_NAME.match(str(name)):
            problems.append(f"{where} must match {FIELD_NAME.pattern}")
        if not isinstance(field, dict):
            problems.append(f"{where} must be an object with a `type`")
            continue

        kind = field.get("type")
        if kind not in FIELD_TYPES:
            problems.append(f"{where}: unknown type {kind!r}; "
                            f"known: {', '.join(FIELD_TYPES)}")
        if not str(field.get("about") or "").strip():
            # The description is what the model is actually steered by, so an
            # unlabelled field is a paid call for a key nobody explained.
            problems.append(f"{where} needs an `about` describing what to put "
                            "in it -- it is what the model is steered by")

        one_of = field.get("one_of")
        if one_of is not None:
            if not isinstance(one_of, list) or not one_of:
                problems.append(f"{where}: `one_of` must be a non-empty list")
            elif len(one_of) > MAX_ENUM:
                problems.append(f"{where}: {len(one_of)} choices; {MAX_ENUM} max")
            elif not all(isinstance(v, str) and v.strip() for v in one_of):
                problems.append(f"{where}: `one_of` values must be strings")

        nested = field.get("of")
        if nested is None:
            continue
        if kind == "text":
            problems.append(f"{where}: `of` needs type 'list' -- it makes each "
                            "entry an object, so there must be entries")
        if one_of is not None and nested:
            problems.append(f"{where}: `one_of` and `of` are exclusive; a "
                            "vocabulary constrains a value, `of` replaces it")
        if not isinstance(nested, dict) or not nested:
            problems.append(f"{where}: `of` must be a non-empty "
                            "{key: description} map")
            continue
        if len(nested) > MAX_NESTED_KEYS:
            problems.append(f"{where}: {len(nested)} keys; {MAX_NESTED_KEYS} max")
        for key, description in nested.items():
            if not FIELD_NAME.match(str(key)):
                problems.append(f"{where}: key {key!r} must match "
                                f"{FIELD_NAME.pattern}")
            if not str(description or "").strip():
                problems.append(f"{where}: key {key!r} needs a description")
        if identity is not None and not (isinstance(identity, list) and identity
                                         and all(k in nested for k in identity)):
            problems.append(f"{where}: `identity` must be a non-empty list of "
                            f"keys from `of` ({', '.join(nested)})")
    return problems


def problems(fields: Any) -> list[str]:
    """`check_fields`, plus the guard that it is a field map at all.

    Separate because `check_shape` has already established that much by the
    time it calls `check_fields`, while a definition arriving over HTTP has
    not.
    """
    if not isinstance(fields, dict) or not fields:
        return ["`fields` must be a non-empty {name: {type, about}} map"]
    return check_fields(fields)


__all__ = ["FIELD_NAME", "FIELD_TYPES", "MAX_ENUM", "MAX_FIELDS",
           "MAX_NESTED_KEYS", "check_fields", "compile_fields", "problems"]
