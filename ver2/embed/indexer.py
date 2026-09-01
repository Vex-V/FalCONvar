"""Descriptions in, vectors out.

Embeds only what is not already indexed *with its current text*. The check is
a hash comparison, not a presence check: a pair that was described by the stub
and then re-described by a real model is present under the same key, and an
index that only asked "do I have this pair" would keep serving the stub's
sentence forever.

Embedding is cheap enough that this is a convenience rather than a necessity --
all of test1 is a fraction of a cent, or a few seconds locally -- but the same
comparison is what makes an index detectably stale rather than quietly wrong,
and that matters at any price.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .embedders import Embedder
from .index.base import VectorIndex
from .units import Unit, embedder_key


@dataclass
class IndexResult:
    embedder: str
    total: int
    embedded: int
    skipped: int
    stale: int
    elapsed_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder,
            "units": self.total,
            "embedded": self.embedded,
            "skipped": self.skipped,
            "restated": self.stale,
            "elapsed_s": round(self.elapsed_s, 3),
        }


def index_units(
    units: Sequence[Unit],
    embedder: Embedder,
    index: VectorIndex,
    video_id: Optional[str] = None,
    force: bool = False,
    batch_size: int = 64,
) -> IndexResult:
    """Embed and store every unit whose text is not already indexed."""
    started = time.perf_counter()
    config = embedder.config()
    key = embedder_key(config)
    index.ensure(config)

    known: dict[tuple[int, str], str] = {}
    if not force and units:
        known = index.existing(video_id or units[0].video_id, config)

    pending: list[Unit] = []
    stale = 0
    for unit in units:
        held = known.get((unit.chunk_id, unit.sampler))
        if held == unit.hash:
            continue
        if held is not None:
            # Present but produced from different text -- the description was
            # rewritten since. Re-embedding replaces it in place.
            stale += 1
        pending.append(unit)

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        vectors = embedder.embed_documents([u.embed_text for u in batch])
        index.upsert(batch, vectors, config)

    return IndexResult(
        embedder=key,
        total=len(units),
        embedded=len(pending),
        skipped=len(units) - len(pending),
        stale=stale,
        elapsed_s=time.perf_counter() - started,
    )
