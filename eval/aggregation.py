"""How a chunk's score should be built from its descriptions' ranks.

`to_moments` sums `1/(k + rank)` over every description a chunk contributed.
That is RRF, but RRF is defined for fusing several rankings of the *same*
items, where every item appears once per ranking and the count is therefore
constant. Here the count varies -- a chunk described by three samplers
contributes three terms, one described by a single sampler contributes one --
and at k=60 over ~20 candidates `1/(k+rank)` spans only 1.31x, so count
overwhelms rank. Measured live: a Chernobyl chunk ranked 1st lost to a test1
chunk whose best description ranked 13th.

Two things must hold at once, and they pull against each other:

  count-blindness  a chunk must not win for being described more often
  agreement        two independent accounts of one window that both match
                   is stronger evidence than one, and should still show

So the arms below vary k (how fast rank decays) and how the per-chunk terms
combine (sum, max, or max plus a discounted remainder).

Ground truth is the (video, chunk) a query was written from, over a mixed
index of two videos with different sampler counts -- which is the case that
exposes the bias, and the normal case in production.
"""

from __future__ import annotations

import collections
import json
import random
import statistics
from pathlib import Path

from ver2 import db
from ver2.embed import embedders as embedders_mod
from ver2.embed import index as index_mod


def make(kind: str, k: int, w: float = 0.0):
    def score(hits):
        per = collections.defaultdict(list)
        for rank, h in enumerate(hits, start=1):
            per[(h.video_id, h.chunk_id)].append(1.0 / (k + rank))
        out = {}
        for key, terms in per.items():
            terms.sort(reverse=True)
            if kind == "sum":
                out[key] = sum(terms)
            elif kind == "max":
                out[key] = terms[0]
            elif kind == "maxplus":                # bounded by nothing: still sums
                out[key] = terms[0] + w * sum(terms[1:])
            elif kind == "maxsecond":              # bounded: at most two terms
                out[key] = terms[0] + w * (terms[1] if len(terms) > 1 else 0.0)
            else:                                  # bounded: mean of the rest
                out[key] = terms[0] + w * (statistics.mean(terms[1:]) if len(terms) > 1 else 0.0)
        return out
    return score


ARMS = {
    "sum k=60  (shipped)": make("sum", 60),
    "sum k=20":            make("sum", 20),
    "sum k=10":            make("sum", 10),
    "sum k=5":             make("sum", 5),
    "max k=60":            make("max", 60),
    "max+0.5rest k=10":    make("maxplus", 10, 0.5),
    "max+0.25rest k=10":   make("maxplus", 10, 0.25),
    "max+0.5rest k=5":     make("maxplus", 5, 0.5),
    "max+0.5second k=10":  make("maxsecond", 10, 0.5),
    "max+0.5mean k=10":    make("maxmean", 10, 0.5),
    "max+1.0mean k=10":    make("maxmean", 10, 1.0),
}


def rank_of(scores, truth):
    order = sorted(scores, key=scores.get, reverse=True)
    return order.index(truth) + 1 if truth in order else None


def main():
    db.load_env()
    pairs = json.loads(Path("eval/query_pairs.json").read_text(encoding="utf-8"))
    emb = embedders_mod.build("openai")
    index = index_mod.build(["pgvector"])
    cfg = emb.config()

    cached = []
    for form in ("literal", "paraphrase"):
        vecs = emb.embed_documents([p[form] for p in pairs])
        for p, v in zip(pairs, vecs):
            cached.append((p, form, index.search(v, cfg, query_text=p[form], limit=20)))

    ranks = {name: {"literal": [], "paraphrase": []} for name in ARMS}
    agree = {name: [] for name in ARMS}
    for p, form, hits in cached:
        truth = (p["video_id"], p["chunk_id"])
        for name, fn in ARMS.items():
            r = rank_of(fn(hits), truth)
            ranks[name][form].append(r if r else 99)
            if form == "literal":
                # does the winning chunk have more than one matching description?
                s = fn(hits)
                top = max(s, key=s.get)
                agree[name].append(sum(1 for h in hits
                                       if (h.video_id, h.chunk_id) == top))

    print(f"{len(pairs)} query pairs over 2 videos, 16 chunks, mixed sampler counts")
    print("ground truth = the (video, chunk) the query was written from\n")
    head = (f"{'aggregation':<22}{'video ok':<10}{'lit top-1':<11}{'lit MRR':<10}"
            f"{'par top-1':<11}{'par MRR':<10}{'mean descs in winner'}")
    print(head); print("-" * len(head))
    for name in ARMS:
        lit, par = ranks[name]["literal"], ranks[name]["paraphrase"]
        vid_ok = sum(
            1 for (p, form, hits) in cached if form == "literal"
            and max(ARMS[name](hits), key=ARMS[name](hits).get)[0] == p["video_id"]
        ) / len(pairs)
        print(f"{name:<22}{vid_ok:<10.3f}"
              f"{sum(x == 1 for x in lit)/len(lit):<11.3f}{sum(1/x for x in lit)/len(lit):<10.3f}"
              f"{sum(x == 1 for x in par)/len(par):<11.3f}{sum(1/x for x in par)/len(par):<10.3f}"
              f"{statistics.mean(agree[name]):.2f}")

    random.seed(0)
    print("\npaired bootstrap vs shipped (95% CI on delta MRR, literal+paraphrase):")
    base = [1/x for x in ranks["sum k=60  (shipped)"]["literal"] + ranks["sum k=60  (shipped)"]["paraphrase"]]
    for name in ARMS:
        if name.startswith("sum k=60"):
            continue
        arm = [1/x for x in ranks[name]["literal"] + ranks[name]["paraphrase"]]
        d = [a - b for b, a in zip(base, arm)]
        boots = sorted(statistics.mean(random.choices(d, k=len(d))) for _ in range(4000))
        lo, hi = boots[100], boots[3900]
        verdict = "no difference" if lo <= 0 <= hi else ("BETTER" if lo > 0 else "WORSE")
        print(f"  {name:<22}{statistics.mean(d):+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  {verdict}")


if __name__ == "__main__":
    main()
