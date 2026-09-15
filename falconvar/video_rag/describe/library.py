"""The prompt vocabulary: built-in questions, plus whatever a user has added.

Two files, and the split is the point:

    falconvar/video_rag/describe/prompts.json   built in, shipped, read-only at runtime
    data/prompts.json                 custom, written by the API

Custom entries layer on top and may not shadow a built-in, so a request can
never change what a shipped question asks -- the alternative is a deployment
whose `yolo` means something different from every other one, with nothing in
the repo saying so.

A question is an instruction and a **shape**. The shape is where the response
schema lives, and a question either names a shipped shape or brings its own --
`yolo` is not a special case in the code, it is the `people` shape.

A shape is just a set of fields. The one marked `fallback` is what an
unrecognised question resolves to; it has no other privilege. Two questions
whose shapes share a field both answer it, and both answers are kept, because
a (sampler, question) pairing is independent of every other pairing on the
chunk.

**A custom shape is built, never accepted.** `fields` is a small builder --
`text` or `list`, plus `of` for a list of objects and `one_of` for a fixed
vocabulary -- and `compile_shape` generates the JSON Schema from it. The schema
reaches the model API with `strict: true`, whose subset is narrow, so a raw
schema arriving over HTTP could express something the API refuses and the
failure would land after the frames are read with the call about to be paid
for. The builder spans exactly the range the shipped shapes already use:
verified by rebuilding all five from their own field lists, identical.

A custom shape is stored under its question's name and dies with it. It may
not take a shipped shape's name -- `people` and `prose` are shape names that
are *not* question names, so the question-level shadow guard does not cover
them -- and it can never claim `fallback`, because exactly one shape is what
every unrecognised question resolves to and it is a shipped one.

**Which keys identify an entry is not a shape's business.** It used to be
declared beside the shapes, for entity linking; a link profile in
`falconvar/aggregates/definitions.json` owns it now, so the same answers can be
linked by different keys without touching what was asked.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Optional

from ...shared import paths

BUILTIN_PATH = Path(__file__).with_name("prompts.json")

#: A question name has to survive being a `--sampler` half, a manifest key, a
#: filename fragment and a JSON Schema `name`, so it is deliberately narrow.
#: A colon would split a `name:question` pair in two.
NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

#: `{n}` and `{span}` are substituted for every question; `{vocabulary}` only
#: has a value when the sampler recorded one, so it renders as a plain note
#: rather than failing. Anything else is a typo that would raise at call time,
#: which is after the frames have been read and the money is about to be spent.
PLACEHOLDERS = {"n", "span", "vocabulary"}

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

_lock = threading.Lock()
_cache: Optional[dict[str, Any]] = None


class PromptError(ValueError):
    """A prompt the vocabulary will not accept, with the reason."""


class Protected(PromptError):
    """The question exists and is built in, so it cannot be changed.

    Distinct from a rejected or unknown one because the answer to a caller is
    different: not "fix this" and not "no such thing", but "this one is not
    yours to edit".
    """


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PromptError(f"{path} is not valid JSON: {exc}") from None


# ------------------------------------------------------- building a shape

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


def compile_shape(spec: dict[str, Any]) -> dict[str, Any]:
    """A field spec -> a shape, in the form the built-ins are already written.

    So `shape_of`, `schema_for` and `fields_of` need no idea that a shape was
    supplied rather than shipped.

    `fallback` is never set. Exactly one shape is what an unrecognised question
    resolves to, and it is a shipped one -- a custom shape that could claim it
    would change what every unknown question means.
    """
    fields = spec.get("fields") or {}
    return {
        "summary": spec.get("summary") or "standard",
        "fallback": False,
        "fields": {name: _compile_field(f) for name, f in fields.items()},
    }


def check_shape(spec: Any) -> list[str]:
    """Everything wrong with a proposed shape, as messages.

    Returned rather than raised, so a caller reports all of them at once.
    """
    problems: list[str] = []
    if not isinstance(spec, dict):
        return ["shape must be an object with a `fields` map"]

    summary = spec.get("summary") or "standard"
    known_summaries = load()["summaries"]
    if summary not in known_summaries:
        problems.append(f"unknown summary {summary!r}; "
                        f"known: {', '.join(sorted(known_summaries))}")

    fields = spec.get("fields")
    if not isinstance(fields, dict) or not fields:
        problems.append("a shape needs at least one field; use the `prose` "
                        "shape for a summary-only question")
        return problems
    return problems + check_fields(fields)


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


def load(refresh: bool = False) -> dict[str, Any]:
    """The merged vocabulary: built-ins, then custom layered on top.

    Cached, because every describe call asks for it. `refresh` is what the API
    uses after a write, so a running server does not serve a stale list -- the
    file is small enough that re-reading it is cheaper than reasoning about
    when a cache is wrong.
    """
    global _cache
    with _lock:
        if _cache is not None and not refresh:
            return _cache

        builtin = _read(BUILTIN_PATH)
        if not builtin:
            raise PromptError(f"{BUILTIN_PATH} is missing; the package is "
                              "incomplete and no question can be resolved")

        merged = {
            "system": builtin["system"],
            "summaries": dict(builtin["summaries"]),
            "shapes": dict(builtin["shapes"]),
            "questions": {name: {**q, "builtin": True}
                          for name, q in builtin["questions"].items()},
        }

        custom = _read(paths.PROMPTS)

        # Shapes before questions: a question resolves its shape by name, so
        # the shape has to exist by the time the question is merged.
        #
        # Stored as a spec and compiled here rather than stored compiled, so
        # the file holds what a person wrote and can hand-edit. The generated
        # JSON Schema is derivable from it, and two copies of one thing is two
        # things that drift.
        for name, spec in (custom.get("shapes") or {}).items():
            # A custom shape may not redefine a shipped one -- and this is not
            # covered by the question-level guard: `people` and `prose` are
            # shape names that are *not* question names, so a question called
            # `people` would otherwise silently rewrite the people schema.
            if name in merged["shapes"]:
                continue
            try:
                merged["shapes"][name] = compile_shape(spec)
                declared = {f: list(s["identity"])
                            for f, s in (spec.get("fields") or {}).items()
                            if isinstance(s, dict) and s.get("identity")}
                if declared:
                    merged["identity"][name] = declared
            except Exception:                              # noqa: BLE001
                # A hand-edited shape that will not compile is dropped, and the
                # question falls back to the general shape exactly as an
                # unknown shape name already does. Refusing at write time is
                # where the error belongs; this file is editable by hand too.
                continue

        for name, entry in (custom.get("questions") or {}).items():
            # A custom entry that collides with a built-in is dropped rather
            # than applied. Refusing at write time is where the error belongs,
            # but the file is hand-editable too, and a shadowed built-in is the
            # one failure that would change a shipped question's meaning
            # silently.
            if name in merged["questions"]:
                continue
            merged["questions"][name] = {**entry, "builtin": False}

        _cache = merged
        return merged


# --------------------------------------------------------------------- reading

def questions() -> list[str]:
    """Every question name, built in or added."""
    return sorted(load()["questions"])


def question(name: str) -> dict[str, Any]:
    entry = load()["questions"].get(name)
    if entry is None:
        raise PromptError(f"unknown question {name!r}; "
                          f"known: {', '.join(questions())}")
    return entry


def shapes() -> dict[str, Any]:
    return load()["shapes"]


def builtin_shapes() -> set[str]:
    """Which shapes ship in the package.

    Read from the file rather than marked on the shape itself: `version_of`
    hashes the shape dict, so a `builtin` key inside it would change every
    hash and re-describe every chunk of every video once.
    """
    return set(_read(BUILTIN_PATH).get("shapes") or {})


def shape_of(name: str) -> dict[str, Any]:
    """The shape a question answers in, falling back to the general one.

    An unknown question resolves to the fallback shape rather than raising,
    because this is called per (chunk, sampler) after the manifest is written:
    the vocabulary is checked before a run, and failing here would mean failing
    with the frames already read.
    """
    entry = load()["questions"].get(name)
    shape = entry.get("shape") if entry else None
    return load()["shapes"].get(shape) or _fallback_shape()


def _fallback_shape() -> dict[str, Any]:
    for shape in load()["shapes"].values():
        if shape.get("fallback"):
            return shape
    raise PromptError("no shape is marked `fallback`; the general question "
                      "has nothing to resolve to")


def instruction_of(name: str) -> str:
    entry = load()["questions"].get(name)
    if entry and entry.get("instruction"):
        return entry["instruction"]
    fallback = next((q for q in load()["questions"].values()
                     if (load()["shapes"].get(q.get("shape")) or {}).get("fallback")),
                    None)
    return (fallback or {}).get("instruction", "")


def fields_of(name: str) -> list[str]:
    """The structured keys this question answers, from its shape.

    A property of the question alone. It used to depend on which other
    questions were asked about the same chunk -- the fallback shape gave up any
    key a specialist owned -- and that coupling was the source of three silent
    faults: sampler ids passed where questions were meant, `yolo:overview`
    taking a key from a call that answered none, and two fallback questions
    overlapping on everything with no rule for which won. It bought a few
    percent of output tokens in one of four possible pairings. Pairings are
    independent now, and overlap is answered twice and kept twice.
    """
    return list(shape_of(name).get("fields") or {})


# ------------------------------------------------------------------ validating

def check(name: str, entry: dict[str, Any],
          shape_spec: Optional[dict[str, Any]] = None) -> list[str]:
    """Everything wrong with a proposed question, as messages.

    Returned rather than raised so a caller reports all of them at once, the
    same way `workflow.validate` does.

    ``shape_spec`` is checked instead of `entry["shape"]` when the question
    brings its own shape rather than naming a shipped one.
    """
    problems: list[str] = []
    if not NAME.match(name or ""):
        problems.append(f"name {name!r} must match {NAME.pattern} -- lowercase, "
                        "no colon, since `sampler:question` splits on one")

    instruction = (entry.get("instruction") or "").strip()
    if not instruction:
        problems.append("instruction is required")
    elif len(instruction) > 4000:
        problems.append(f"instruction is {len(instruction)} characters; 4000 max")
    else:
        # A `{typo}` would raise at call time -- after the frames are read and
        # with the request about to be paid for.
        try:
            unknown = {f for _, f, _, _ in __import__("string").Formatter()
                       .parse(instruction) if f} - PLACEHOLDERS
        except ValueError as exc:
            problems.append(f"instruction has malformed braces: {exc}")
        else:
            if unknown:
                problems.append(
                    f"instruction uses unknown placeholder(s) "
                    f"{', '.join(sorted(unknown))}; known: "
                    f"{', '.join(sorted(PLACEHOLDERS))}")

    if shape_spec is not None:
        # A shape supplied with the question. Its name is the question's, so
        # the shipped shape names have to be refused here too.
        #
        # Against the *built-ins*, not against every shape: a custom shape is
        # stored under its question's name, so checking the merged set would
        # make a question unable to edit the shape it already owns.
        if name in builtin_shapes():
            problems.append(f"{name!r} is a built-in shape; a question that "
                            "defines its own shape cannot take that name")
        problems.extend(check_shape(shape_spec))
        return problems

    shape = entry.get("shape")
    if shape not in load()["shapes"]:
        problems.append(f"unknown shape {shape!r}; "
                        f"known: {', '.join(sorted(load()['shapes']))}")
    return problems


# -------------------------------------------------------------------- writing

def _write_custom(doc: dict[str, Any]) -> None:
    paths.PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.PROMPTS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    tmp.replace(paths.PROMPTS)          # atomic: a torn file is a broken run


def add(name: str, instruction: str, shape: str = "scene",
        about: str = "", fields: Optional[dict[str, Any]] = None,
        summary: str = "standard") -> dict[str, Any]:
    """Add or replace a custom question. Built-ins are refused.

    ``fields`` makes the question bring its own shape instead of naming a
    shipped one, and the shape is then stored under the question's own name.
    Ownership of a key like `people` still stays with the shipped shapes: a
    custom shape may not take a shipped shape's name, and no custom shape can
    claim `fallback`.
    """
    shape_spec = (None if fields is None
                  else {"summary": summary, "fields": fields})
    entry = {"shape": name if shape_spec is not None else shape,
             "instruction": instruction.strip(),
             **({"about": about.strip()} if about.strip() else {})}

    if load()["questions"].get(name, {}).get("builtin"):
        raise Protected(
            f"{name!r} is a built-in question and cannot be replaced. Built-ins "
            "live in the package so that every deployment's `yolo` means the "
            "same thing; pick another name.")

    problems = check(name, entry, shape_spec)
    if problems:
        raise PromptError("; ".join(problems))

    with _lock:
        doc = _read(paths.PROMPTS) or {"document": "prompts", "version": 1,
                                       "questions": {}}
        doc.setdefault("questions", {})[name] = entry
        shapes = doc.setdefault("shapes", {})
        if shape_spec is not None:
            shapes[name] = shape_spec
        else:
            # Switching a question from its own shape back to a shipped one
            # leaves the old shape addressed by nothing.
            shapes.pop(name, None)
        if not shapes:
            doc.pop("shapes")
        _write_custom(doc)
    load(refresh=True)
    return {**entry, "builtin": False}


def remove(name: str) -> None:
    """Delete a custom question, and the shape it brought with it.

    The shape goes because it is keyed by the question's name and nothing else
    can reference it -- a shipped shape is never touched, since a question
    naming one does not own it.
    """
    if load()["questions"].get(name, {}).get("builtin"):
        raise Protected(f"{name!r} is built in and cannot be deleted")
    with _lock:
        doc = _read(paths.PROMPTS)
        if not (doc.get("questions") or {}).pop(name, None):
            raise PromptError(f"no custom question {name!r}")
        (doc.get("shapes") or {}).pop(name, None)
        if "shapes" in doc and not doc["shapes"]:
            doc.pop("shapes")
        _write_custom(doc)
    load(refresh=True)
