"""What gets embedded, and how it is identified.

One unit is one description -- one ``(video_id, chunk_id, sampler)`` -- which
is the unit a describer call produced. Not the chunk: merging a chunk's
descriptions into one blob averages away the reason there is more than one.
The clip description of test1 chunk 2 is about the room and the yolo
description is about the people, and they share almost no vocabulary; a single
vector for both answers each question half as well.

Every unit carries a **hash of its own text**. Embeddings are derived data --
the descriptions are the record -- so the question an index has to be able to
answer is not "do I have a vector for this pair" but "do I have a vector for
*this text*". Re-describing a chunk with a better model changes the text, the
hash changes with it, and the stale vector is detectable rather than silently
retrieved.

The point id is a UUID derived from the same three fields, so re-indexing the
same pair overwrites rather than duplicating, in Qdrant as in Postgres.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Optional

#: Fixed namespace so ids are stable across machines and runs.
NAMESPACE = uuid.UUID("2f1a6a1e-5f3c-4a4e-9a0d-6d1c2b3a4f50")


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def embedder_key(config: dict[str, Any]) -> str:
    """A short, stable name for one embedder configuration.

    Part of the index key, because two embedders' vectors are not comparable
    and must never be mixed in one ranking. Also the Qdrant collection name,
    so switching embedder makes a new collection rather than corrupting one.
    """
    return f"{config['name']}:{config.get('model', '')}:{config['dimensions']}"


def collection_name(key: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in key)
    return f"descriptions__{safe}"


def render(structured: Optional[dict[str, Any]]) -> str:
    """The structured fields as text, one line per field.

    A specialist's entry is a bound record -- appearance, clothing, role and
    action for one person -- so its fields are kept together in one clause
    rather than scattered across parallel lists. Three separators, one per
    level of nesting, so the structure survives into the text: ``. `` between
    fields, ``|`` between entities, ``;`` between one entity's attributes.

    **Keys are sorted, and that is a correctness requirement rather than
    tidiness.** ``jsonb`` does not preserve object key order -- it stores keys
    by (length, bytewise) -- so a description read back from Postgres hands
    back ``{role, action, clothing, appearance}`` where the file had
    ``{appearance, clothing, role, action}``. Rendering in iteration order made
    the text, and therefore ``text_hash`` and the vector, depend on which
    *copy* of a description it came from: indexing the same data from the file
    and from ``--video-id`` produced different embeddings, and each run
    re-embedded what the other had just written, reporting "description
    changed" forever. Sorting makes the text a function of content alone.

    **Each attribute is named**, because the separator between two fields used
    to be ``", "`` -- which also occurs inside them, as in "dark green top,
    dark pants, black sneakers". Where one field ended and the next began was
    invisible. Measured on test1 over 29 disjoint query pairs, naming them
    changes retrieval by less than the noise floor (every paired bootstrap CI
    spans zero); it is kept for the structure it makes explicit, at about 6%
    more characters, and because a field that is to become a filterable enum
    should be a named term rather than a bare word among clothing.
    """
    lines = []
    for key, value in sorted((structured or {}).items()):
        if isinstance(value, str) and value.strip():
            lines.append(f"{key}: {value}")
        elif isinstance(value, list) and value:
            items = []
            for item in value:
                if isinstance(item, dict):
                    items.append("; ".join(f"{k} {item[k]}"
                                           for k in sorted(item) if item[k]))
                else:
                    items.append(str(item))
            lines.append(f"{key}: " + " | ".join(items))
    return ". ".join(lines)


@dataclass
class Unit:
    video_id: str
    chunk_id: int
    sampler: str
    text: str
    structured: Optional[dict[str, Any]] = None
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    frame_indexes: Optional[list[int]] = None
    manifest_fingerprint: Optional[str] = None

    @property
    def embed_text(self) -> str:
        """Summary and structured fields together -- what the vector is built from.

        Measured on test1, 22 disjoint query pairs, dense MRR: summary alone
        0.528 literal / 0.522 paraphrase; structured alone 0.636 / 0.586; both
        0.705 / 0.608. Summary alone loses because every summary repeats the
        same setting, while the fields are almost all distinctive -- mean
        pairwise similarity 0.854 against 0.802.
        """
        rendered = render(self.structured)
        return "\n\n".join([self.text, rendered]) if rendered else self.text

    @property
    def hash(self) -> str:
        # Over what is actually embedded, so a change to either half is caught.
        return text_hash(self.embed_text)

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(NAMESPACE,
                              f"{self.video_id}:{self.chunk_id}:{self.sampler}"))

    def payload(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "chunk_id": self.chunk_id,
            "sampler": self.sampler,
            "content": self.text,
            "structured": self.structured or {},
            "text_hash": self.hash,
            "manifest_fingerprint": self.manifest_fingerprint,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "frame_indexes": self.frame_indexes or [],
        }


def from_document(document: dict[str, Any]) -> list[Unit]:
    """Units from a description document, skipping pairs with no description."""
    units: list[Unit] = []
    for chunk in document.get("chunks", []):
        for sampler, block in chunk.get("samplers", {}).items():
            text = block.get("description")
            if not text:
                continue
            units.append(Unit(
                video_id=document["video_id"],
                chunk_id=chunk["chunk_id"],
                sampler=sampler,
                text=text,
                structured=block.get("structured") or {},
                start_ts=chunk.get("start_ts"),
                end_ts=chunk.get("end_ts"),
                frame_indexes=block.get("frame_indexes"),
                manifest_fingerprint=document.get("manifest_fingerprint"),
            ))
    return units


def from_transcript(document: dict[str, Any]) -> list[Unit]:
    """Units from a transcript document. One per chunk that holds speech.

    The whole test of whether `embed` was cut in the right place. A transcript
    chunk is text with a time span and some bound structure, which is exactly
    what a description is, so it needs no new index, no new table and no new
    code past this function -- it lands in `chunk_embeddings` beside the
    describers' output with `sampler = "transcript"`, and `--sampler
    transcript` narrows a search to what was said the same way `--sampler yolo`
    narrows it to who was seen.

    Silent chunks are skipped. They are kept in the transcript document because
    `chunk_id` is shared with the video side and dropping them would renumber
    everything after -- but a vector for the empty string is not a thing worth
    storing, and it would answer every query equally badly.

    `manifest_fingerprint` carries the **timeline** fingerprint here. The
    column's job is to name the upstream artifact this was derived from, and
    for audio that is the grid rather than a manifest.

    **Only `speakers` is carried into `structured`, not `turns`.** For a
    description the structured half is worth embedding because it holds content
    the summary does not -- measured at 0.705 against 0.528 MRR for summary
    alone. For a transcript it is the opposite: `turns[].text` *is* the text, so
    rendering it appends the entire chunk a second time, interleaved with
    timestamps read as numbers. The turns are not lost -- they are the record,
    and they live in `audio_chunks.structured` where they can be queried.
    `speakers` stays because it is the one genuinely useful filter here: every
    moment a particular voice was talking.
    """
    units: list[Unit] = []
    fingerprint = document.get("timeline_fingerprint")
    for chunk in document.get("chunks", []):
        if not (chunk.get("text") or "").strip():
            continue
        units.append(Unit(
            video_id=document["video_id"],
            chunk_id=chunk["chunk_id"],
            sampler="transcript",
            text=chunk["text"],
            structured={"speakers": (chunk.get("structured") or {}).get("speakers") or []},
            start_ts=chunk.get("start_ts"),
            end_ts=chunk.get("end_ts"),
            frame_indexes=None,
            manifest_fingerprint=fingerprint,
        ))
    return units


def from_rows(video_id: str, rows: Iterable[dict[str, Any]],
              bounds: Optional[dict[int, tuple[float, float]]] = None) -> list[Unit]:
    """Units straight from `descriptions` rows.

    ``bounds`` maps chunk_id to (start_ts, end_ts) from the manifest, because a
    description row does not know where its chunk sits -- the same split that
    makes `recovery.supabase_description` fetch two things.
    """
    units: list[Unit] = []
    for row in rows:
        if not row.get("description"):
            continue
        start, end = (bounds or {}).get(row["chunk_id"], (None, None))
        units.append(Unit(
            video_id=video_id,
            chunk_id=row["chunk_id"],
            sampler=row["sampler"],
            text=row["description"],
            structured=row.get("structured") or {},
            start_ts=start,
            end_ts=end,
            frame_indexes=row.get("frame_indexes"),
            manifest_fingerprint=row.get("manifest_fingerprint"),
        ))
    return units
