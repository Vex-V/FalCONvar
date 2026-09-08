"""JSON Schema generated from the document dataclasses.

The dataclasses are the single source of truth; `db/json/*.schema.json` is
generated and checked in, and `--check` fails on a stale diff so the dataclass,
the schema and the SQL cannot drift apart silently.

`validate` is structural unless `deep`: presence, the document tag, and
top-level types. That catches a document written by the wrong component or an
older version, and costs nothing on a manifest holding tens of thousands of
frame records.
"""

from __future__ import annotations

import dataclasses
import json
import typing
from pathlib import Path
from typing import Any, Optional, get_args, get_origin

from . import documents, paths

#: Written to `db/json/`. Relative to the checkout, not the data root: these
#: are source, not output.
SCHEMA_DIR = paths.REPO_ROOT / "db" / "json"

_PRIMITIVES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _json_type(annotation: Any) -> dict[str, Any]:
    """One dataclass field's type, as a JSON Schema fragment."""
    if annotation in _PRIMITIVES:
        return {"type": _PRIMITIVES[annotation]}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is typing.Union:
        # `Optional[X]` is `Union[X, None]`. Rendered as a nullable type
        # rather than dropped, because "absent" and "null" are different facts
        # in these documents -- an unknown duration is not a zero one.
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1:
            schema = _json_type(inner[0])
            kind = schema.get("type")
            if isinstance(kind, str):
                schema["type"] = [kind, "null"]
            return schema
        return {}
    if origin in (list, tuple):
        item = _json_type(args[0]) if args else {}
        return {"type": "array", "items": item or True}
    if origin is dict:
        return {"type": "object"}
    if annotation is Any:
        return {}
    return {}


def schema_for(cls: type) -> dict[str, Any]:
    """A JSON Schema for one document dataclass.

    Derived from `as_dict`'s shape where it differs from the field list: the
    document is what gets written, and it carries `document`, `version` and
    computed values like `fingerprint` that are not fields.
    """
    fields = {f.name: f for f in dataclasses.fields(cls)}
    hints = typing.get_type_hints(cls)

    properties: dict[str, Any] = {}
    required: list[str] = []

    name = getattr(cls, "__name__", "document")
    tag = next((k for k, v in documents.DOCUMENTS.items() if v is cls), None)
    if tag is not None:
        properties["document"] = {"const": tag}
        properties["version"] = {"type": "integer"}
        required += ["document", "version"]

    for field_name, field in fields.items():
        properties[field_name] = _json_type(hints.get(field_name, Any))
        if (field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING):  # type: ignore[misc]
            required.append(field_name)

    # Computed, not stored: present in the written document but not a field.
    for extra in ("fingerprint", "manifest_fingerprint", "speakers"):
        if hasattr(cls, extra) and extra not in properties:
            properties[extra] = {"type": ["string", "array"]}

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": name,
        "type": "object",
        # Open on purpose: a document may gain a field before every reader
        # knows it, and refusing to parse it would make adding one a
        # coordinated release rather than an append.
        "additionalProperties": True,
        "required": sorted(required),
        "properties": properties,
    }


def generate(out_dir: Optional[Path] = None) -> dict[str, Path]:
    """Write one schema per document. Returns name -> path."""
    out_dir = Path(out_dir or SCHEMA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for tag, cls in sorted(documents.DOCUMENTS.items()):
        path = out_dir / f"{tag}.schema.json"
        path.write_text(json.dumps(schema_for(cls), indent=2) + "\n",
                        encoding="utf-8")
        written[tag] = path
    return written


def check(out_dir: Optional[Path] = None) -> list[str]:
    """Which checked-in schemas no longer match the dataclasses.

    What CI runs. Empty means the three descriptions still agree.
    """
    out_dir = Path(out_dir or SCHEMA_DIR)
    stale: list[str] = []
    for tag, cls in sorted(documents.DOCUMENTS.items()):
        path = out_dir / f"{tag}.schema.json"
        if not path.exists():
            stale.append(f"{tag}: {path} does not exist")
            continue
        want = json.dumps(schema_for(cls), indent=2) + "\n"
        if path.read_text(encoding="utf-8") != want:
            stale.append(f"{tag}: {path} is out of date")
    return stale


def validate(document: dict[str, Any], tag: str, deep: bool = False) -> list[str]:
    """Problems with a document, as messages. Empty means it conforms.

    Structural only unless ``deep``: presence, the document tag, and the type
    of each top-level field. That catches a document written by the wrong
    component or an older version, which is what a handoff contract is for --
    and it costs nothing on a manifest holding fifty thousand frame records.
    """
    cls = documents.DOCUMENTS.get(tag)
    if cls is None:
        return [f"unknown document type {tag!r}"]
    schema = schema_for(cls)
    problems: list[str] = []

    found = document.get("document")
    if found is not None and found != tag:
        problems.append(f"document says {found!r}, expected {tag!r}")
    for name in schema["required"]:
        if name not in document:
            problems.append(f"missing required field {name!r}")

    if deep:
        try:
            import jsonschema
        except ImportError:
            problems.append("deep validation needs `jsonschema` installed")
        else:
            for error in jsonschema.Draft202012Validator(schema).iter_errors(
                    document):
                problems.append(f"{'.'.join(str(p) for p in error.path)}: "
                                f"{error.message}")
    return problems


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Generate JSON Schema from the document dataclasses.")
    ap.add_argument("--check", action="store_true",
                    help="fail if the checked-in schemas are stale")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.check:
        stale = check(args.out)
        for line in stale:
            print(f"stale: {line}")
        print(f"{len(documents.DOCUMENTS) - len(stale)}"
              f"/{len(documents.DOCUMENTS)} schemas current")
        return 1 if stale else 0

    written = generate(args.out)
    for tag, path in written.items():
        print(f"  {tag:16} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
