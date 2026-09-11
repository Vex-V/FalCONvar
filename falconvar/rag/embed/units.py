"""Documents into embeddable units.

One unit is one `(video_id, chunk_id, sampler_id)`. Both modalities produce
them: a description from the picture, a transcript chunk from the soundtrack
with `sampler_id = "transcript"`.

What gets embedded is the summary *and* the structured fields -- every summary
repeats the same setting, the fields do not.

`render` sorts keys at every level. `jsonb` preserves array order but not object
key order, so a document read back from Postgres hands its keys back in a
different order from the file; joining values in iteration order made the same
person into different text, a different hash and a different vector depending
on where it was read from.

Three separators, one per level: `. ` between fields, ` | ` between entities,
`; ` between one entity's attributes. Fields are named, because a bare `", "`
also occurs inside values and made field boundaries invisible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ...shared.documents import Descriptions, Transcript, fingerprint_of


def render(summary: str, structured: dict[str, Any]) -> str:
    """One string per unit: the summary, then every structured field, named."""
    parts: list[str] = []
    if summary and summary.strip():
        parts.append(summary.strip())
    for key in sorted(structured):
        value = structured[key]
        rendered = _render_value(value)
        if rendered:
            parts.append(f"{key}: {rendered}")
    return ". ".join(parts)


def _render_value(value: Any) -> str:
    if value is None or value == "" or value == []:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        # Sorted here too. Determinism had been handled one level deep and not
        # two, which is exactly how the Postgres/file divergence survived.
        return "; ".join(f"{k} {_render_value(value[k])}" for k in sorted(value)
                         if _render_value(value[k]))
    if isinstance(value, list):
        return " | ".join(r for r in (_render_value(v) for v in value) if r)
    return str(value)


@dataclass
class Unit:
    """One embeddable thing, keyed by a hash of its own text."""

    video_id: str
    chunk_id: int
    sampler_id: str
    content: str
    structured: dict[str, Any] = field(default_factory=dict)
    vector: Optional[list[float]] = None
    #: The two halves of `sampler_id`, carried rather than parsed. Filtering by
    #: question is the query a person actually makes -- "the text on screen",
    #: not "what the CLIP sampler said" -- and it is not expressible as a
    #: suffix match, because a bare id like `clip` means question == strategy.
    sampler: str = ""
    question: str = ""

    @property
    def text_hash(self) -> str:
        """Identity is the text. Re-embedding is then "what changed", not
        "what is here", which is what makes a re-run cost nothing."""
        return fingerprint_of({"content": self.content})

    @property
    def key(self) -> str:
        return f"{self.video_id}:{self.chunk_id}:{self.sampler_id}"

    def as_dict(self) -> dict[str, Any]:
        return {"video_id": self.video_id, "chunk_id": self.chunk_id,
                "sampler_id": self.sampler_id, "content": self.content,
                "structured": self.structured, "text_hash": self.text_hash,
                "sampler": self.sampler, "question": self.question,
                "vector": self.vector}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Unit":
        return cls(d["video_id"], d["chunk_id"], d["sampler_id"], d["content"],
                   d.get("structured", {}), d.get("vector"),
                   d.get("sampler", ""), d.get("question", ""))


def from_descriptions(document: Descriptions) -> list[Unit]:
    """One unit per (chunk, sampler) that has an answer."""
    units: list[Unit] = []
    for chunk in document.chunks:
        for sampler_id, block in chunk.get("samplers", {}).items():
            structured = block.get("structured") or {}
            content = render(block.get("description", ""), structured)
            if not content:
                continue
            units.append(Unit(document.video_id, chunk["chunk_id"], sampler_id,
                              content, structured,
                              sampler=block.get("sampler")
                              or sampler_id.split(":")[0],
                              question=block.get("question") or sampler_id))
    return units


def from_transcript(document: Transcript) -> list[Unit]:
    """One unit per chunk that has speech, keyed `transcript`.

    Silent chunks are skipped -- they stay in the transcript document because
    `chunk_id` is shared with the video side, but a vector of the empty string
    answers every query equally badly.
    """
    units: list[Unit] = []
    for chunk in document.chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        # `speakers` only, never `turns`: `turns[].text` *is* the transcript,
        # so rendering it appends the whole chunk a second time interleaved
        # with timestamps read as numbers.
        structured = {"speakers": (chunk.get("structured") or {}).get("speakers", [])}
        units.append(Unit(document.video_id, chunk["chunk_id"], "transcript",
                          render(text, structured), structured,
                          sampler="transcript", question="transcript"))
    return units


def from_summary(video_id: str, payload: dict[str, Any]) -> Optional[Unit]:
    """The whole video as one unit, from the `summary` aggregate.

    **Kept apart from the chunk units, in its own table.** `embeddings` answers
    *which twenty seconds*; a summary answers *which video*, and a video is not
    a moment you can play. Mixing them would return a whole-video "moment"
    beside real ones in every search, and it would need a sentinel `chunk_id`
    to sit in a table keyed by one.

    Only the final summary, never the intermediate layers. A leaf summary
    covers a real span and is worth keeping as a record, but indexing the
    layers would return the same moment two or three times over under
    different wordings -- the count bias the moment aggregation guards against,
    one level up.

    `chunk_id = -1` marks it as not-a-chunk for anything that reads a Unit
    generically; nothing keyed by chunk ever sees it, because this goes to its
    own table.
    """
    summary = (payload.get("summary") or "").strip()
    if not summary:
        return None
    # The same three-level render as a chunk unit, so the text a video is
    # found by is built exactly like the text a moment is found by.
    structured = {key: payload[key] for key in ("topics", "setting", "notable")
                  if payload.get(key)}
    content = render(summary, structured)
    return Unit(video_id, -1, "summary", content, structured,
                sampler="summary", question="summary")


__all__ = ["Unit", "render", "from_descriptions", "from_summary",
           "from_transcript"]
