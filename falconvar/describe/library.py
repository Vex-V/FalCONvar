"""The prompt vocabulary: built-in questions, plus whatever a user has added.

Two files, and the split is the point:

    falconvar/describe/prompts.json   built in, shipped, read-only at runtime
    data/prompts.json                 custom, written by the API

Custom entries layer on top and may not shadow a built-in, so a request can
never change what a shipped question asks -- the alternative is a deployment
whose `yolo` means something different from every other one, with nothing in
the repo saying so.

A question is an instruction and a **shape**. The shape is where the response
schema lives, so adding a question is writing prose rather than JSON Schema,
and the shapes the built-ins use are the same ones a custom question picks --
`yolo` is not a special case in the code, it is the `people` shape.

A shape either *is* the fallback or owns its fields outright:

    fallback   the general question. Its fields are offered only where no
               sibling on the chunk owns them.
    exact      its fields, always, and it owns those keys against the
               fallback. `prose` is the degenerate case: no fields at all.

Ownership is derived from a shape's fields rather than declared beside them,
because a declaration is a second list to keep in step with the first.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Optional

from ..shared import paths

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

        for name, entry in (_read(paths.PROMPTS).get("questions") or {}).items():
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


def owns(name: str) -> list[str]:
    """The keys this question claims outright. Derived from its shape."""
    shape = shape_of(name)
    return [] if shape.get("fallback") else list(shape.get("fields") or {})


def owner_map() -> dict[str, str]:
    """key -> the question that owns it.

    Sorted so a collision resolves the same way every run rather than by dict
    order. Two questions owning one key is legal but not useful; `check()`
    reports it, and the fallback gives the key up to whichever wins here.
    """
    out: dict[str, str] = {}
    for name in sorted(load()["questions"]):
        for key in owns(name):
            out.setdefault(key, name)
    return out


# ------------------------------------------------------------------ validating

def check(name: str, entry: dict[str, Any]) -> list[str]:
    """Everything wrong with a proposed question, as messages.

    Returned rather than raised so a caller reports all of them at once, the
    same way `workflow.validate` does.
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
        about: str = "") -> dict[str, Any]:
    """Add or replace a custom question. Built-ins are refused."""
    entry = {"shape": shape, "instruction": instruction.strip(),
             **({"about": about.strip()} if about.strip() else {})}

    if load()["questions"].get(name, {}).get("builtin"):
        raise Protected(
            f"{name!r} is a built-in question and cannot be replaced. Built-ins "
            "live in the package so that every deployment's `yolo` means the "
            "same thing; pick another name.")

    problems = check(name, entry)
    if problems:
        raise PromptError("; ".join(problems))

    with _lock:
        doc = _read(paths.PROMPTS) or {"document": "prompts", "version": 1,
                                       "questions": {}}
        doc.setdefault("questions", {})[name] = entry
        _write_custom(doc)
    load(refresh=True)
    return {**entry, "builtin": False}


def remove(name: str) -> None:
    """Delete a custom question. Built-ins are refused."""
    if load()["questions"].get(name, {}).get("builtin"):
        raise Protected(f"{name!r} is built in and cannot be deleted")
    with _lock:
        doc = _read(paths.PROMPTS)
        if not (doc.get("questions") or {}).pop(name, None):
            raise PromptError(f"no custom question {name!r}")
        _write_custom(doc)
    load(refresh=True)
