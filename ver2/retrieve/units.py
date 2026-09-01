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
    action for one person -- so joining its values keeps who-did-what together
    rather than scattering it across parallel lists.
    """
    lines = []
    for key, value in sorted((structured or {}).items()):
        if isinstance(value, str) and value.strip():
            lines.append(f"{key}: {value}")
        elif isinstance(value, list) and value:
            items = []
            for item in value:
                if isinstance(item, dict):
                    items.append(", ".join(str(v) for v in item.values() if v))
                else:
                    items.append(str(item))
            lines.append(f"{key}: " + "; ".join(items))
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
