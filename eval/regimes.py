"""Where each half of the hybrid earns its place, per query.

One call to `search_descriptions` returns both ranks per description, so three
rankings come out of it without three searches:

  dense-only    order by vector_rank
  lexical-only  order by text_rank, hits without one dropped
  hybrid        the order the RPC already fused

Each is folded into chunks by `to_moments`, exactly as the shipped path does,
and scored by where the true chunk lands. Classifying a query by which half
found it is the point: a search box gives no signal about which kind it is,
which is why both halves run on every query.
"""

from __future__ import annotations

import json
from pathlib import Path

from ver2 import db
from ver2.embed import embedders as embedders_mod
from ver2.embed import index as index_mod
from ver2.retrieve.search import to_moments


def rank_by(hits, key, chunk_id):
    """Chunk rank under one ordering; None if that half ranked nothing."""
    usable = [h for h in hits if key(h) is not None]
    if not usable:
        return None
    ordered = sorted(usable, key=key)
    for position, moment in enumerate(to_moments(ordered), start=1):
        if moment.chunk_id == chunk_id:
            return position
    return None


def probe(index, emb, cfg, query, truth):
    hits = index.search(emb.embed_query(query), cfg, query_text=query, limit=20)
    return {
        "dense": rank_by(hits, lambda h: h.vector_rank, truth),
        "lexical": rank_by(hits, lambda h: h.text_rank, truth),
        "hybrid": rank_by(hits, lambda h: -h.score, truth),
        "lex_hits": sum(1 for h in hits if h.text_rank),
    }


def main():
    db.load_env()
    pairs = json.loads(Path("eval/query_pairs.json").read_text(encoding="utf-8"))
    emb = embedders_mod.build("openai")
    index = index_mod.build(["pgvector"])
    cfg = emb.config()

    rows = []
    for p in pairs:
        for form in ("literal", "paraphrase"):
            r = probe(index, emb, cfg, p[form], p["chunk_id"])
            rows.append({"query": p[form], "form": form, "truth": p["chunk_id"], **r})

    def score(r):                      # reciprocal rank, 0 if that half found nothing
        return 0.0 if r is None else 1.0 / r

    dense_only = [r for r in rows if score(r["dense"]) > score(r["lexical"])]
    lex_only = [r for r in rows if score(r["lexical"]) > score(r["dense"])]
    both = [r for r in rows if r["dense"] == 1 and r["lexical"] == 1]
    lifted = [r for r in rows
              if score(r["hybrid"]) > max(score(r["dense"]), score(r["lexical"]))]

    print(f"{len(rows)} queries (29 literal + 29 paraphrase)\n")
    print(f"  dense strictly better than lexical   {len(dense_only)}")
    print(f"  lexical strictly better than dense   {len(lex_only)}")
    print(f"  both put the true chunk at rank 1    {len(both)}")
    print(f"  hybrid beats BOTH halves alone       {len(lifted)}")
    print(f"  lexical contributed nothing at all   {sum(1 for r in rows if r['lex_hits'] == 0)}")

    def show(title, sel, n=5):
        print(f"\n=== {title} ===")
        print(f"  {'query':<46}{'form':<12}{'dense':<7}{'lex':<7}{'hybrid'}")
        for r in sel[:n]:
            print(f"  {r['query'][:44]:<46}{r['form']:<12}"
                  f"{str(r['dense']):<7}{str(r['lexical']):<7}{r['hybrid']}")

    show("DENSE wins -- lexical is blind here",
         sorted(dense_only, key=lambda r: (score(r["lexical"]), -score(r["dense"]))))
    show("LEXICAL wins -- dense is blind here",
         sorted(lex_only, key=lambda r: (score(r["dense"]), -score(r["lexical"]))))
    show("BOTH agree at rank 1", both)
    show("HYBRID beats either half alone", lifted)
    Path("eval/regimes.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n-> eval/regimes.json")


if __name__ == "__main__":
    main()
