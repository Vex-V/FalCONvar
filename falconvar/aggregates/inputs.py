"""What an aggregate reads: a selection over what video_rag extracted.

Three operators, and nothing else:

    ,        separate inputs -- each is its own answer
    +        sources joined into one input
    [a,b]    fields of one answer joined into one input

A source is where the text comes from:

    transcript                        what was said
    *                                 every answer
    activity                          one question, wherever it was asked
    clip:activity                     one pairing
    clip:*                            everything one sampler answered

and brackets say which part of each answer:

    clip:activity                     the prose alone
    clip:hazards[severity,hazards]    only those fields
    clip:activity[summary,actors]     the prose, and a field
    *[summary,*]                      the prose and every field: what search embeds
    yolo[people.clothing]             keys inside a list's entries

So `clip:hazards[severity,hazards]` is one answer over both fields and
`clip:hazards[severity],clip:hazards[hazards]` is two. `sev=clip:hazards[severity]`
labels an input; its answer is stored as `<aggregate>~sev`. Two or more inputs
are always labelled -- from their fields when no label is given.

**A bare name is a question, never a sampler.** `text` is both, the OCR sampler
and the question that reads the screen, and "the text on screen" is what a
person means by it. A sampler's whole output is `text:*`.

**`+` joins sources, so `clip:text+scene` is not the sampler spec here.** It
reads `clip:text` and the question `scene` wherever it was asked. A sampler
spec's `+` lists questions for one pass over the frames; an input's joins what
is read.

Structured fields are rendered by the function that renders them for the
search index, so a field reads to an aggregate exactly as it reads to a search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

from ..shared.contracts.documents import fingerprint_of
from ..shared.errors import FalconvarError

if TYPE_CHECKING:
    from .base import Context

#: What a text aggregate reads unless told otherwise: what was said, then every
#: account of the picture. Every account rather than a best-ranked one -- a run
#: that picked stub `clip` text over 428 words of real narration produced a
#: summary correctly reporting that it had been given nothing.
DEFAULT = "transcript+*"

#: The prose half of an answer, as a field name. It is what the describe
#: schema calls it, so no shape can have a field of that name.
PROSE = "summary"

_NAME = r"[a-z][a-z0-9_-]{0,31}"
_HEAD = re.compile(rf"^(?:transcript|\*|{_NAME}(?::(?:{_NAME}|\*))?)$")
_FIELD = re.compile(r"^(?:\*|[a-z][a-z0-9_]{0,31}(?:\.[a-z][a-z0-9_]{0,31})?)$")
_SOURCE = re.compile(r"^([^\[\]]+?)(?:\[([^\[\]]*)\])?$")
_LABEL = re.compile(r"^([a-z0-9][a-z0-9_-]{0,31})=(.+)$")


class InputError(FalconvarError, ValueError):
    """A selection that does not parse, or names what does not exist."""


@dataclass(frozen=True)
class Source:
    """One place text comes from, and which part of it."""

    head: str
    fields: tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.head + (f"[{','.join(self.fields)}]" if self.fields else "")

    def matches(self, sampler_id: str, block: dict[str, Any]) -> bool:
        """Whether one stored answer is what this source names."""
        if self.head == "*":
            return True
        sampler = block.get("sampler") or sampler_id.split(":")[0]
        question = block.get("question") or sampler_id
        if ":" in self.head:
            wanted_sampler, wanted_question = self.head.split(":", 1)
            return sampler == wanted_sampler and wanted_question in ("*", question)
        return question == self.head


@dataclass(frozen=True)
class Input:
    """Sources joined into one input, which makes one answer."""

    sources: tuple[Source, ...]
    label: Optional[str] = None

    @property
    def selector(self) -> str:
        return "+".join(str(s) for s in self.sources)

    def __str__(self) -> str:
        return f"{self.label}={self.selector}" if self.label else self.selector


def _split(text: str, separator: str) -> list[str]:
    """Split on a separator outside brackets."""
    parts: list[str] = []
    depth, current = 0, []
    for ch in text:
        depth += (ch == "[") - (ch == "]")
        if depth < 0 or depth > 1:
            raise InputError(f"unbalanced brackets in {text!r}")
        if ch == separator and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if depth:
        raise InputError(f"unbalanced brackets in {text!r}")
    parts.append("".join(current).strip())
    return parts


def parse(selection: str) -> list[Input]:
    """A selection string -> its inputs. Raises `InputError` on syntax alone;
    whether the names exist is `check`'s question."""
    text = (selection or "").strip()
    if not text:
        raise InputError("an empty selection reads nothing")
    out: list[Input] = []
    for piece in _split(text, ","):
        if not piece:
            raise InputError(f"an empty input in {selection!r}")
        label = None
        labelled = _LABEL.match(piece)
        if labelled:
            label, piece = labelled.group(1), labelled.group(2).strip()
        sources = []
        for part in _split(piece, "+"):
            found = _SOURCE.match(part)
            if not found:
                raise InputError(f"cannot read {part!r}: a source is a name, "
                                 "then optionally [fields]")
            head = found.group(1).strip()
            if not _HEAD.match(head):
                raise InputError(f"{head!r} is not a source: transcript, *, a "
                                 "question, sampler:question or sampler:*")
            fields: tuple[str, ...] = ()
            if found.group(2) is not None:
                fields = tuple(f.strip() for f in found.group(2).split(","))
                bad = [f for f in fields if not _FIELD.match(f)]
                if bad or not fields:
                    raise InputError(f"{part!r}: a field is a name, `field.key` "
                                     f"or `*`; got {', '.join(map(repr, bad)) or 'none'}")
                if head == "transcript":
                    raise InputError("the transcript has no fields; read it as `transcript`")
            sources.append(Source(head, fields))
        out.append(Input(tuple(sources), label))
    given = [one.label for one in out if one.label]
    if len(given) != len(set(given)):
        raise InputError(f"a label is used twice in {selection!r}")
    return out


def check(inputs: Sequence[Input], vocabulary: dict[str, Any]) -> list[str]:
    """Everything a selection names that does not exist, as messages.

    Against the vocabulary rather than a video: a typo is a 422 before a run is
    queued. Whether *this* video was asked a question is a run-time answer, and
    an input that finds nothing is skipped with the reason.
    """
    questions: dict[str, dict[str, Any]] = vocabulary["questions"]
    samplers = set(vocabulary["samplers"])
    problems: list[str] = []
    for one in inputs:
        for source in one.sources:
            head = source.head
            if head in ("transcript", "*"):
                continue                # `*` fields differ by answer; read, not checked
            question = head
            if ":" in head:
                sampler, question = head.split(":", 1)
                if sampler not in samplers:
                    problems.append(f"unknown sampler {sampler!r} in {source}; "
                                    f"known: {', '.join(sorted(samplers))}")
                if question == "*":
                    continue
            if question not in questions:
                hint = (f" -- `{head}:*` reads everything that sampler answered"
                        if head in samplers else "")
                problems.append(f"unknown question {question!r} in {source}{hint}")
                continue
            shape = questions[question]
            for name in source.fields:
                if name in (PROSE, "*"):
                    continue
                top, _, key = name.partition(".")
                if top not in shape:
                    problems.append(f"{source}: {question} has no field {top!r} "
                                    f"-- it has {', '.join([PROSE, *shape])}")
                elif key and key not in (shape[top] or ()):
                    problems.append(f"{source}: {top} entries have "
                                    f"{', '.join(shape[top] or ()) or 'no keys'}, "
                                    f"not {key!r}")
    return problems


# ------------------------------------------------------------------- reading

@dataclass
class Row:
    """What one input says about one chunk: labelled parts, in source order."""

    chunk_id: int
    start: float
    end: float
    parts: list[tuple[str, str]]

    @property
    def text(self) -> str:
        return "  ".join(f"{label}: {said}" for label, said in self.parts)

    @property
    def line(self) -> str:
        """The chunk id travels with the text, so a model can cite it."""
        return f"[{self.chunk_id}] {self.start:.0f}-{self.end:.0f}s  {self.text}"


@dataclass
class Read:
    """One input, read against one video."""

    input: Input
    rows: list[Row]
    #: The answer ids that contributed, and `transcript`.
    answers: list[str]

    @property
    def chars(self) -> int:
        return sum(len(row.line) for row in self.rows)

    @property
    def empty(self) -> bool:
        return not self.rows

    @property
    def why_empty(self) -> str:
        return f"this video has nothing for `{self.input.selector}`"

    def fingerprint(self) -> str:
        """A hash of the text actually read -- never of what was available.

        Hashing every source's text rebuilt a summary that reads only the
        transcript whenever a description changed.
        """
        return fingerprint_of({"rows": [[r.chunk_id, [list(p) for p in r.parts]]
                                        for r in self.rows]})


def render(block: dict[str, Any], fields: Sequence[str],
           render_answer: Callable[[str, dict[str, Any]], str]) -> str:
    """One stored answer, reduced to the selected fields, as text."""
    prose = (block.get("description") or "").strip()
    if not fields:
        return prose
    structured = block.get("structured") or {}
    chosen: dict[str, Any] = {}
    nested: dict[str, list[str]] = {}
    for name in fields:
        if name == "*":
            chosen.update(structured)
        elif "." in name:
            top, key = name.split(".", 1)
            nested.setdefault(top, []).append(key)
        elif name != PROSE and name in structured:
            chosen[name] = structured[name]
    for top, keys in nested.items():
        if top in chosen:                # the whole field was selected as well
            continue
        entries = [{k: entry[k] for k in keys if k in entry}
                   for entry in (structured.get(top) or []) if isinstance(entry, dict)]
        entries = [e for e in entries if any(str(v or "").strip() for v in e.values())]
        if entries:
            chosen[top] = entries
    return render_answer(prose if PROSE in fields else "", chosen)


def read(context: "Context", one: Input) -> Read:
    """Every chunk's text for one input. Chunks with nothing are left out.

    A field renders through the index's own renderer, so the text a summary is
    built from is the text that was embedded -- one string, not two that look
    alike.
    """
    from ..shared.contracts.units import render as render_unit

    chunks = {c["chunk_id"]: c for c in
              (context.descriptions.chunks if context.descriptions else [])}
    rows: list[Row] = []
    answers: set[str] = set()
    for chunk_id in context.chunk_ids():
        parts: list[tuple[str, str]] = []
        for source in one.sources:
            if source.head == "transcript":
                said = (context.transcript.text_of(chunk_id) or "").strip() \
                    if context.transcript is not None else ""
                if said:
                    parts.append(("transcript", said))
                    answers.add("transcript")
                continue
            for sampler_id, block in ((chunks.get(chunk_id) or {}).get("samplers") or {}).items():
                if not source.matches(sampler_id, block):
                    continue
                said = render(block, source.fields, render_unit)
                if said:
                    parts.append((sampler_id, said))
                    answers.add(sampler_id)
        if parts:
            start, end = context.span_of(chunk_id)
            rows.append(Row(chunk_id, start, end, parts))
    return Read(one, rows, sorted(answers))


# ----------------------------------------------------------------- answer ids

def labels(inputs: Sequence[Input]) -> list[Optional[str]]:
    """The label each input's answer is stored under.

    A lone input keeps the aggregate's own id unless it was labelled, so
    `--input summary=transcript` replaces `summary` rather than adding beside
    it. Two or more are always labelled, from their fields when not by hand.
    """
    if len(inputs) == 1:
        return [inputs[0].label]
    derived: list[str] = []
    for one in inputs:
        fields = [f for s in one.sources for f in s.fields if f not in (PROSE, "*")]
        derived.append(one.label or _slug("-".join(fields) if fields else one.selector))
    for index, one in enumerate(inputs):
        if not one.label and derived.count(derived[index]) > 1:
            derived[index] = _slug(one.selector)
    if len(set(derived)) != len(derived):
        raise InputError("two inputs would be stored under one label; label "
                         "them, as `name=selector`")
    return list(derived)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "all"


def answer_id(definition_id: str, label: Optional[str]) -> str:
    return f"{definition_id}~{label}" if label else definition_id


def definition_of(answer: str) -> str:
    """`entities:people~clothing` -> `entities:people`."""
    return answer.split("~", 1)[0]


def filename(answer: str) -> str:
    """Windows forbids `:` in a filename. No definition name or label contains
    a `.`, so the mapping is reversible."""
    return answer.replace(":", ".") + ".json"


def answer_of_file(stem: str) -> str:
    return stem.replace(".", ":")


__all__ = ["DEFAULT", "PROSE", "Input", "InputError", "Read", "Row", "Source",
           "answer_id", "answer_of_file", "check", "definition_of", "filename",
           "labels", "parse", "read", "render"]
