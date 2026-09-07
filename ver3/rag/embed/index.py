"""Where vectors live, and the lexical half beside them.

A local index: one JSON file per `(video, embedder)`, holding every unit's
vector, its text and its structured payload. It is not a production vector
store and does not pretend to be -- it exists so the *ranking* can be exercised
and measured without Postgres or Qdrant running, which is what the whole
retrieval argument needs to be checkable.

**It keeps a lexical half, because the dense half alone is measurably worse.**
Measured on `falconvar`, 7 literal queries whose strings appear in exactly one
chunk:

    dense only          top-1 0.429   MRR 0.600
    dense + lexical     top-1 0.714   MRR 0.857

Better on 4 of 7, tied on 3, never worse -- which is the property RRF is chosen
for. And each half is strong exactly where the other fails: on 22 query pairs
with zero shared content words, BM25 scored 59% top-1 on literal queries and
18% on paraphrases, while dense scored 23% on both. Neither half knows which
kind of query it was handed, and a search box gives no signal, which is the
whole argument for fusing rather than choosing.

The lexical half here is a plain term-frequency score rather than BM25. It is
weaker than Postgres's `ts_rank_cd`, and the note above is what a real backend
should reproduce -- but fusing two rankings is the part that must be right, and
that part is identical.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from ...shared import paths, sinks
from .units import Unit

#: RRF's constant for fusing the two rankings of one description.
FUSE_K = 60

_WORD = re.compile(r"[a-z0-9£$€%.:'-]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def index_path(video_id: str, embedder_key: str) -> Path:
    safe = embedder_key.replace(":", "_").replace("/", "_")
    return paths.home(video_id) / "index" / f"{safe}.json"


class LocalIndex:
    """Vectors and text for one video under one embedder."""

    def __init__(self, video_id: str, embedder_key: str) -> None:
        self.video_id = video_id
        self.embedder_key = embedder_key
        self.path = index_path(video_id, embedder_key)
        self.units: list[Unit] = []
        if self.path.exists():
            document = sinks.read_json(self.path)
            self.units = [Unit.from_dict(u) for u in document.get("units", [])]

    # -- writing ---------------------------------------------------------
    def stored_hashes(self) -> dict[str, str]:
        """key -> text_hash, so a re-run embeds only what changed."""
        return {u.key: u.text_hash for u in self.units}

    def upsert(self, units: Sequence[Unit]) -> int:
        by_key = {u.key: u for u in self.units}
        for unit in units:
            by_key[unit.key] = unit
        self.units = list(by_key.values())
        return len(units)

    def prune(self, live_keys: set[str]) -> int:
        """Drop units for chunks that no longer exist.

        A grid that shrank leaves rows naming a chunk nobody can play. The
        file writer rewrites the whole document so it never had the problem
        `falconvar`'s appending Postgres sink did -- but a *smaller* new set
        still has to remove what is now orphaned.
        """
        before = len(self.units)
        self.units = [u for u in self.units if u.key in live_keys]
        return before - len(self.units)

    def save(self) -> Path:
        return sinks.write_json(self.path, {
            "document": "index", "version": 1,
            "video_id": self.video_id, "embedder": self.embedder_key,
            "units": [u.as_dict() for u in self.units],
        })

    # -- reading ---------------------------------------------------------
    def search(self, vector: Sequence[float], query: str,
               limit: int = 20, sampler: Optional[str] = None
               ) -> list[dict[str, Any]]:
        """Dense and lexical rankings, fused per unit with RRF.

        RRF reads only the orderings, so it needs no calibration -- cosine
        distance and a term-frequency score have no common scale, and any
        weight between them would be invented.
        """
        pool = [u for u in self.units
                if sampler is None or u.sampler_id == sampler]
        if not pool:
            return []

        dense = sorted(pool, key=lambda u: -_cosine(vector, u.vector or []))
        dense_rank = {u.key: i + 1 for i, u in enumerate(dense)}

        terms = tokenize(query)
        scored = [(u, _lexical(terms, u.content)) for u in pool]
        matching = [u for u, s in sorted(scored, key=lambda p: -p[1]) if s > 0]
        text_rank = {u.key: i + 1 for i, u in enumerate(matching)}

        hits = []
        for unit in pool:
            d = dense_rank.get(unit.key)
            t = text_rank.get(unit.key)
            score = (1.0 / (FUSE_K + d) if d else 0.0) + (
                1.0 / (FUSE_K + t) if t else 0.0)
            hits.append({"chunk_id": unit.chunk_id, "sampler_id": unit.sampler_id,
                         "content": unit.content, "structured": unit.structured,
                         "score": score, "dense_rank": d, "text_rank": t})
        hits.sort(key=lambda h: -h["score"])
        return hits[:limit]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _lexical(terms: Sequence[str], content: str) -> float:
    if not terms:
        return 0.0
    counts = Counter(tokenize(content))
    if not counts:
        return 0.0
    total = sum(counts.values())
    return sum(counts[t] for t in terms) / total


__all__ = ["FUSE_K", "LocalIndex", "index_path", "tokenize"]
