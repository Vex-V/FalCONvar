"""Aggregate definitions, as data: prompts, and link profiles.

    falconvar/aggregates/definitions/definitions.json   built in, shipped, read-only at runtime
    data/aggregates.json                    custom, written by the API

The split `describe` makes for its questions, for the same reason: a custom
entry may not shadow a built-in, so every deployment's `summary` means what the
repo says it means.

**A prompt** is what an llm aggregate asks; its kind is how it is asked:

    fold    batch the chunks, summarise each batch, summarise the summaries
    spans   contiguous ranges over the whole video, each citing chunk ids
    items   discrete things, each citing the chunk it happened in

The kinds are the only code. `summary`, `chapters` and `events` are one entry
of each, and a custom prompt is another entry rather than another class. There
is no free-form kind: an answer with no citations is a fold.

**A link profile** names a field, the keys that identify an entry of it, and
what to write about each entity once linked. Who is who is decided by
`linking` -- embeddings under rules -- and the model only writes the account.
With `check: flag` it also names observations that contradict the rest; those
stay in the entity, flagged, and the account is written without them. Dropping
them was measured and lost true links on both labelled videos.

**Identity lives here, not in the describe shapes.** It used to sit beside a
shape; a profile owning it means the same `people` answers can be linked by
clothing in one profile and by appearance in another, without touching what
was asked.

**An answer is built, never accepted.** `fields` is describe's builder and the
schema is generated from it, because a raw schema over HTTP could express
something a strict API refuses, failing with the call about to be paid for. The
keys a kind writes itself -- `chunk_id`, `first_chunk`, `last_chunk`, `doubts`
-- cannot be declared.

Everything a model is told is in the JSON, the kinds' own wording included, so
`version_of` covers all of it and editing any of it rebuilds what it wrote.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ...shared import paths
from .. import inputs as inputs_mod
from ...shared.errors import FalconvarError

BUILTIN_PATH = Path(__file__).with_name("definitions.json")
SECTIONS = ("prompts", "profiles")
#: A profile's aggregate id is this plus its name: `entities:people`.
PROFILE_PREFIX = "entities:"

NAME = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
FIELD_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
KINDS = ("fold", "spans", "items")
#: Names a definition cannot take: the aggregators written as code, and the
#: prefix every profile id carries.
RESERVED = frozenset({"stats", "speakers", "coverage", "ner", "sentiment", "entities"})
#: Keys a kind writes into an answer itself.
OWNED: dict[str, frozenset[str]] = {
    "fold": frozenset(), "spans": frozenset({"first_chunk", "last_chunk"}),
    "items": frozenset({"chunk_id"}), "link": frozenset({"doubts"}),
}
CHECKS = ("flag", "off")
MAX_INSTRUCTION = 4000
MAX_NARRATIVES = 50

PROMPT_KEYS = frozenset({"kind", "about", "instruction", "fields", "inputs",
                         "key", "fold_instruction"})
PROFILE_DEFAULTS: dict[str, Any] = {
    "identity": [], "story": [], "transcript": False, "from": "*",
    "threshold": None, "rule": "max", "mutual": True, "check": "flag",
    "min_appearances": 2, "max_narratives": 12,
}
PROFILE_KEYS = frozenset({"about", "field", "instruction", "fields", *PROFILE_DEFAULTS})

_lock = threading.Lock()
_cache: Optional[dict[str, Any]] = None


class DefinitionError(FalconvarError, ValueError):
    """A definition that will not be accepted, or does not exist.

    Carries the problems as a list, because a message may itself contain the
    separator a joined string would be split on.
    """

    def __init__(self, message: str, problems: Optional[list[str]] = None) -> None:
        super().__init__(message)
        self.problems = problems or [message]


class ProtectedDefinition(DefinitionError):
    """Built in, so not yours to change."""


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DefinitionError(f"{path} is not valid JSON: {exc}") from None


def _with_defaults(section: str, entry: dict[str, Any]) -> dict[str, Any]:
    return {**PROFILE_DEFAULTS, **entry} if section == "profiles" else dict(entry)


def load(refresh: bool = False) -> dict[str, Any]:
    """Built-ins, then custom layered on top. Cached; `refresh` after a write.

    A custom entry that shadows a built-in or fails its check is dropped and
    listed under `problems` rather than raised: the file is hand-editable, and
    one typo taking down `/capabilities` takes every generated form with it.
    """
    global _cache
    with _lock:
        if _cache is not None and not refresh:
            return _cache
        builtin = _read(BUILTIN_PATH)
        if not builtin:
            raise DefinitionError(f"{BUILTIN_PATH} is missing; the package is incomplete")
        merged: dict[str, Any] = {
            "system": builtin["system"],
            "kinds": builtin["kinds"],
            "prompts": {n: {**e, "builtin": True} for n, e in builtin["prompts"].items()},
            "profiles": {n: {**_with_defaults("profiles", e), "builtin": True}
                         for n, e in builtin["profiles"].items()},
            "problems": {},
        }
        custom = _read(paths.AGGREGATE_DEFINITIONS)
        for section in SECTIONS:
            for name, entry in (custom.get(section) or {}).items():
                where = f"{section}/{name}"
                if name in merged[section]:
                    merged["problems"][where] = ["shadows a built-in; ignored"]
                    continue
                if not isinstance(entry, dict):
                    merged["problems"][where] = ["not an object"]
                    continue
                entry = _with_defaults(section, entry)
                found = CHECKERS[section](name, entry)
                if found:
                    merged["problems"][where] = found
                    continue
                merged[section][name] = {**entry, "builtin": False}
        _cache = merged
        return merged


# --------------------------------------------------------------------- reading

def system() -> str:
    return load()["system"]


def kind_text(kind: str) -> dict[str, str]:
    """What a kind appends to every instruction, from the data."""
    return dict(load()["kinds"].get(kind) or {})


def prompts() -> dict[str, dict[str, Any]]:
    return load()["prompts"]


def profiles() -> dict[str, dict[str, Any]]:
    return load()["profiles"]


def get(section: str, name: str) -> dict[str, Any]:
    if section not in SECTIONS:
        raise DefinitionError(f"section must be one of {', '.join(SECTIONS)}")
    entry = load()[section].get(name)
    if entry is None:
        raise DefinitionError(f"unknown {section[:-1]} {name!r}; known: "
                              f"{', '.join(sorted(load()[section])) or 'none'}")
    return entry


def ids() -> list[str]:
    """Every definition's aggregate id: prompts by name, profiles prefixed."""
    return [*sorted(prompts()), *(PROFILE_PREFIX + p for p in sorted(profiles()))]


def locate(definition_id: str) -> tuple[str, str]:
    """`entities:people` -> ("profiles", "people"); `summary` -> ("prompts", "summary")."""
    if definition_id.startswith(PROFILE_PREFIX):
        return "profiles", definition_id[len(PROFILE_PREFIX):]
    return "prompts", definition_id


def version_of(section: str, name: str) -> str:
    """A hash of everything the model is told for this definition: the entry,
    its kind's own wording and the system preamble."""
    entry = {k: v for k, v in get(section, name).items() if k != "builtin"}
    kind = entry.get("kind", "link")
    payload = json.dumps({"system": system(), "kind": kind_text(kind), "entry": entry},
                         sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def default_selection(definition_id: str) -> str:
    section, name = locate(definition_id)
    entry = get(section, name)
    if section == "profiles":
        return entry.get("from") or "*"
    return entry.get("inputs") or inputs_mod.DEFAULT


@dataclass(frozen=True)
class Selection:
    """What one link runs over: which answers, which field, which keys."""

    sources: tuple[inputs_mod.Source, ...]
    field: str
    #: The identity keys. Empty links the field's whole value.
    keys: tuple[str, ...]


def selection(name: str, override: Optional[inputs_mod.Input] = None) -> Selection:
    """A profile's selection, narrowed by an input if one is given.

    `yolo[people.clothing]` against `people` reads only yolo's answers and
    links on clothing alone. An input may narrow the identity keys, never
    widen them: a key the profile does not name is a different profile.
    """
    entry = get("profiles", name)
    field = entry["field"]
    identity = tuple(entry.get("identity") or ())
    chosen = override or inputs_mod.parse(entry.get("from") or "*")[0]
    keys = identity
    asked = [f for s in chosen.sources for f in s.fields]
    if override is not None and asked:
        if not identity:
            raise inputs_mod.InputError(
                f"profile {name!r} links whole `{field}` values; its input takes no fields")
        allowed = {f"{field}.{k}" for k in identity}
        wrong = [f for f in asked if f not in allowed]
        if wrong:
            raise inputs_mod.InputError(
                f"{', '.join(wrong)}: profile {name!r} identifies by "
                f"{', '.join(sorted(allowed))}; an input may narrow that, not widen it")
        keys = tuple(dict.fromkeys(f.split(".", 1)[1] for f in asked))
    return Selection(tuple(inputs_mod.Source(s.head) for s in chosen.sources),
                     field, keys)


# ------------------------------------------------------------------ validating

def _check_name(name: str) -> list[str]:
    problems = []
    if not NAME.match(name or ""):
        problems.append(f"name {name!r} must match {NAME.pattern}")
    if name in RESERVED:
        problems.append(f"{name!r} is reserved: {', '.join(sorted(RESERVED))}")
    return problems


def _check_text(value: Any, what: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [f"`{what}` is required"]
    if len(value) > MAX_INSTRUCTION:
        return [f"`{what}` is {len(value)} characters; {MAX_INSTRUCTION} max"]
    return []


def _check_fields(fields: Any, owned: frozenset[str]) -> list[str]:
    from ...shared.contracts.fields import problems as field_problems
    problems = field_problems(fields)
    if isinstance(fields, dict):
        taken = sorted(set(fields) & owned)
        if taken:
            problems.append(f"{', '.join(taken)}: written by the kind itself, "
                            "so a definition cannot declare it")
    return problems


def _check_selection(text: Any, vocabulary: Optional[dict[str, Any]],
                     single: bool = False) -> list[str]:
    try:
        parsed = inputs_mod.parse(str(text))
    except inputs_mod.InputError as exc:
        return [str(exc)]
    problems = []
    if single and (len(parsed) != 1 or any(s.fields for s in parsed[0].sources)):
        problems.append("`from` is one input of sources without fields; the "
                        "profile's `field` and `identity` say what is read")
    if vocabulary is not None:
        problems += inputs_mod.check(parsed, vocabulary)
    return problems


def check_prompt(name: str, entry: dict[str, Any],
                 vocabulary: Optional[dict[str, Any]] = None) -> list[str]:
    """Everything wrong with a proposed prompt. `vocabulary` checks its
    `inputs` against what exists; None checks syntax alone."""
    problems = _check_name(name)
    unknown = set(entry) - PROMPT_KEYS - {"builtin"}
    if unknown:
        problems.append(f"unknown key(s) {', '.join(sorted(unknown))}; a prompt "
                        f"takes {', '.join(sorted(PROMPT_KEYS))}")
    kind = entry.get("kind")
    if kind not in KINDS:
        problems.append(f"kind must be one of {', '.join(KINDS)}")
    problems += _check_text(entry.get("instruction"), "instruction")
    if entry.get("fold_instruction") is not None:
        problems += (["`fold_instruction` is for kind fold"] if kind != "fold"
                     else _check_text(entry["fold_instruction"], "fold_instruction"))
    if entry.get("key") is not None:
        if kind not in ("spans", "items"):
            problems.append("`key` names the list a spans or items answer is under")
        elif not FIELD_NAME.match(str(entry["key"])):
            problems.append(f"`key` must match {FIELD_NAME.pattern}")
    problems += _check_fields(entry.get("fields"), OWNED.get(kind, frozenset()))
    if entry.get("inputs") is not None:
        problems += _check_selection(entry["inputs"], vocabulary)
    return problems


def check_profile(name: str, entry: dict[str, Any],
                  vocabulary: Optional[dict[str, Any]] = None) -> list[str]:
    """Everything wrong with a proposed link profile (defaults applied)."""
    problems = _check_name(name)
    unknown = set(entry) - PROFILE_KEYS - {"builtin"}
    if unknown:
        problems.append(f"unknown key(s) {', '.join(sorted(unknown))}; a profile "
                        f"takes {', '.join(sorted(PROFILE_KEYS))}")
    field = entry.get("field")
    identity, story = entry.get("identity"), entry.get("story")
    if not isinstance(field, str) or not FIELD_NAME.match(field):
        problems.append("`field` names what is linked: a list field such as "
                        "`people`, a text field, `summary` or `transcript`")
    for what, keys in (("identity", identity), ("story", story)):
        if not isinstance(keys, list) or not all(
                isinstance(k, str) and FIELD_NAME.match(k) for k in keys):
            problems.append(f"`{what}` must be a list of entry keys")
    if problems:
        return problems

    if identity and field in (inputs_mod.PROSE, "transcript"):
        problems.append(f"`{field}` has no entries, so no `identity` keys")
    if story and not identity:
        problems.append("`story` picks entry keys, so it needs `identity`")
    threshold = entry.get("threshold")
    if threshold is not None and not (isinstance(threshold, (int, float))
                                      and 0 < threshold < 1):
        problems.append("`threshold` is a cosine similarity between 0 and 1")
    if not identity and threshold is None:
        problems.append(
            "a profile without `identity` links whole values, and nothing in one "
            "answer is provably different from anything else, so no threshold "
            "can be read off the video -- give `threshold`")
    from ..entities.linking import RULES
    if entry.get("rule") not in RULES:
        problems.append(f"`rule` must be one of {', '.join(RULES)}")
    if entry.get("check") not in CHECKS:
        problems.append(f"`check` must be one of {', '.join(CHECKS)}")
    for key in ("mutual", "transcript"):
        if not isinstance(entry.get(key), bool):
            problems.append(f"`{key}` is true or false")
    appearances, narratives = entry.get("min_appearances"), entry.get("max_narratives")
    if not isinstance(appearances, int) or appearances < 1:
        problems.append("`min_appearances` is a whole number, at least 1")
    if not isinstance(narratives, int) or not 0 <= narratives <= MAX_NARRATIVES:
        problems.append(f"`max_narratives` is a whole number, 0 to {MAX_NARRATIVES}")
    problems += _check_text(entry.get("instruction"), "instruction")
    problems += _check_fields(entry.get("fields"), OWNED["link"])
    if field != "transcript":
        problems += _check_selection(entry.get("from") or "*", vocabulary, single=True)

    if vocabulary is not None and field not in (inputs_mod.PROSE, "transcript"):
        shapes = list(vocabulary["questions"].values())
        if identity:
            wanted = set(identity) | set(story)
            if not any(isinstance(s.get(field), list) and wanted <= set(s[field])
                       for s in shapes):
                problems.append(f"no question's shape has a `{field}` list whose "
                                f"entries carry {', '.join(sorted(wanted))}")
        elif not any(field in s and s[field] is None for s in shapes):
            problems.append(f"no question's shape has a `{field}` field holding a value")
    return problems


CHECKERS = {"prompts": check_prompt, "profiles": check_profile}


# -------------------------------------------------------------------- writing

def add(section: str, name: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Add or replace a custom definition. Built-ins are refused."""
    if section not in SECTIONS:
        raise DefinitionError(f"section must be one of {', '.join(SECTIONS)}")
    if load()[section].get(name, {}).get("builtin"):
        raise ProtectedDefinition(f"{name!r} is a built-in {section[:-1]} and cannot be "
                        "replaced; pick another name")
    clean = {k: v for k, v in entry.items() if v is not None and k != "builtin"}
    from ...video_rag import driver as video_rag
    problems = CHECKERS[section](name, _with_defaults(section, clean),
                                 video_rag.vocabulary())
    if problems:
        raise DefinitionError("; ".join(problems), problems)
    with _lock:
        doc = _read(paths.AGGREGATE_DEFINITIONS) or {
            "document": "aggregate_definitions", "version": 1}
        doc.setdefault(section, {})[name] = clean
        _write(doc)
    load(refresh=True)
    return get(section, name)


def remove(section: str, name: str) -> None:
    """Delete a custom definition. Answers it already wrote are untouched."""
    if load()[section].get(name, {}).get("builtin"):
        raise ProtectedDefinition(f"{name!r} is built in and cannot be deleted")
    with _lock:
        doc = _read(paths.AGGREGATE_DEFINITIONS)
        if not (doc.get(section) or {}).pop(name, None):
            raise DefinitionError(f"no custom {section[:-1]} {name!r}")
        _write(doc)
    load(refresh=True)


def _write(doc: dict[str, Any]) -> None:
    target = paths.AGGREGATE_DEFINITIONS
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(target)                  # atomic: a torn file is a broken run


__all__ = ["CHECKS", "DefinitionError", "KINDS", "PROFILE_PREFIX", "ProtectedDefinition",
           "Selection", "add", "check_profile", "check_prompt", "default_selection",
           "get", "ids", "kind_text", "load", "locate", "profiles", "prompts",
           "remove", "selection", "system", "version_of"]
