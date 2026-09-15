"""`embedded.json` -- what text went into the index, for a human to read.

Not a backend. Nothing searches it, which is why it needs no ranking code to
drift from the two that do.

No vectors: 1536 floats per unit is the part nobody can read, and it is already
in the index. One file per video rather than per embedder, because `render`
produces the same text whichever model will embed it.

Rewritten whole each time, so it cannot hold a stale unit.
"""

from __future__ import annotations

from typing import Sequence

from ...shared import paths
from ...shared.storage import sinks
from ...shared.contracts.documents import Embedded
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
                "sampler": u.sampler, "question": u.question,
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
