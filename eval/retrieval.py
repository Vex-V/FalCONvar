"""Measure the shipped retrieval path, with and without its lexical half.

The two arms use the *same* vectors in the same table; only `p_query_text` is
withheld, which makes `search_descriptions` skip its `by_text` CTE and fuse
nothing. So the difference is the lexical half and nothing else -- cleaner than
comparing pgvector to qdrant, where the index, the client and the ranking code
all differ as well.

Literal and paraphrase forms are reported separately because they are what the
two halves are respectively good at: BM25 collapses when the words change,
dense is unmoved by rephrasing, and neither knows which it was handed.

Ground truth is the chunk a query was written from. Five chunks, so a random
ranking scores 0.200 top-1 and 0.457 MRR.
"""

from __future__ import annotations

import json
import statistics
import random
from pathlib import Path

from ver2 import db
from ver2.embed import embedders as embedders_mod
from ver2.embed import index as index_mod
from ver2.retrieve.search import to_moments


def rank_of(hits, chunk_id):
    for position, moment in enumerate(to_moments(hits), start=1):
        if moment.chunk_id == chunk_id:
            return position
    return None


def main():
    db.load_env()
    pairs = json.loads(Path("eval/query_pairs.json").read_text(encoding="utf-8"))
    emb = embedders_mod.build("openai")
    index = index_mod.build(["pgvector"])
    cfg = emb.config()

    forms = ("literal", "paraphrase")
    vectors = {f: emb.embed_documents([p[f] for p in pairs]) for f in forms}
    ranks: dict[tuple[str, bool], list[int]] = {}
    text_hits: dict[str, int] = {}

    for form in forms:
        for hybrid in (False, True):
            got, lex = [], 0
            for pair, vec in zip(pairs, vectors[form]):
                hits = index.search(vec, cfg,
                                    query_text=pair[form] if hybrid else None,
                                    limit=20)
                lex += any(h.text_rank for h in hits)
                got.append(rank_of(hits, pair["chunk_id"]))
            ranks[(form, hybrid)] = got
            if hybrid:
                text_hits[form] = lex

    # The random baseline depends on how many chunks are in play, so it is
    # derived rather than written down -- the corpus went from 5 chunks to 16
    # the first time a second video was indexed, and a hardcoded 0.457 lied.
    keyed = [(q.get("video_id"), q["chunk_id"]) for q in pairs]
    n = len(set(keyed))
    rnd = sum(1 / r for r in range(1, n + 1)) / n
    print(f"{len(pairs)} query pairs, {n} chunks   (random: top-1 {1/n:.3f}, MRR {rnd:.3f})")
    print()
    head = f"{'':<14}{'arm':<22}{'top-1':<9}{'MRR':<9}queries with any text rank"
    print(head); print("-" * len(head))
    for form in forms:
        for hybrid in (False, True):
            r = ranks[(form, hybrid)]
            name = "dense + BM25 (RRF)" if hybrid else "dense only"
            lex = f"{text_hits[form]}/{len(pairs)}" if hybrid else "--"
            print(f"{form if not hybrid else '':<14}{name:<22}"
                  f"{sum(x == 1 for x in r)/len(r):<9.3f}{sum(1/x for x in r)/len(r):<9.3f}{lex}")
        print()

    random.seed(0)
    print("paired bootstrap on the lexical half's contribution (95% CI on delta MRR):")
    for form in forms:
        a = [1/x for x in ranks[(form, False)]]
        b = [1/x for x in ranks[(form, True)]]
        d = [y - x for x, y in zip(a, b)]
        boots = sorted(statistics.mean(random.choices(d, k=len(d))) for _ in range(4000))
        lo, hi = boots[100], boots[3900]
        # An exactly-zero delta gives CI [0, 0], which `lo < 0 < hi` reads as
        # significant. That is the case where the lexical half matched nothing
        # at all, so it is the strongest possible "no effect", not a negative.
        if lo == hi == 0:
            verdict = "no effect -- BM25 matched nothing"
        elif lo <= 0 <= hi:
            verdict = "no difference"
        else:
            verdict = "HELPS" if lo > 0 else "HURTS"
        print(f"  {form:<12}{statistics.mean(d):+.3f}   CI [{lo:+.3f}, {hi:+.3f}]   {verdict}")


if __name__ == "__main__":
    main()
