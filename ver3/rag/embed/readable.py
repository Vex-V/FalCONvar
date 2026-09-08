"""`embedded.json` -- what text went into the index, for a human to read.

Not a backend. Nothing searches this, which is exactly why it can be simple:
there is no ranking here to drift from the two that do rank, and no vectors to
keep in step with a store.

**It answers one question**: what did this chunk actually contribute? That is
what gets asked when a ranking looks wrong, and it is not answerable from the
index -- Qdrant's store is opaque and Postgres holds the text beside a vector
nobody can read at a glance. It is also the question the "summary *and*
structured fields" measurement was made from: 0.705 against 0.528 MRR, decided
by reading what each rendering actually produced.

**No vectors, on purpose.** 1536 floats per unit is the part of the answer
nobody can read, and it is already in whichever index the run wrote to.

**One file per video, not per embedder.** `units.render()` produces the same
text whichever model will embed it, so a file per embedder would be several
identical copies of the readable half.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ...shared import paths, sinks
from ...shared.documents import Embedded
from .units import Unit


def build(video_id: str, units: Sequence[Unit],
          timeline_fingerprint: str = "") -> Embedded:
    """The dump, in chunk order so it reads top to bottom like the video."""
    ordered = sorted(units, key=lambda u: (u.chunk_id, u.sampler_id))
    return Embedded(
        video_id=video_id,
        timeline_fingerprint=timeline_fingerprint,
        units=[{"chunk_id": u.chunk_id,
                "sampler_id": u.sampler_id,
                "text_hash": u.text_hash,
                "characters": len(u.content),
                "content": u.content,
                "structured": u.structured} for u in ordered],
    )


def write(video_id: str, units: Sequence[Unit],
          timeline_fingerprint: str = "") -> str:
    """Rewritten whole every time, so it never holds a stale unit.

    The reason `prune` exists for the real indexes and not here: an upserting
    store keeps whatever it was never told to remove, while a file that is
    replaced wholesale cannot.
    """
    document = build(video_id, units, timeline_fingerprint)
    return str(sinks.write_json(paths.artifact(video_id, "embedded"),
                                document.as_dict()))


def load(video_id: str) -> Embedded:
    return Embedded.from_dict(
        sinks.read_json(paths.artifact(video_id, "embedded")))


__all__ = ["build", "load", "write"]
