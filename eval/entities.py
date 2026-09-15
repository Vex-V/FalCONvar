"""Grade entity linking against hand labels.

    python -m eval.entities                         # grid, tuned on test1
    python -m eval.entities --embedder local

Pairwise: every pair of scored mentions is either linked or not, and either the
same subject by the labels or not. A mention labelled `unsure` is left out of
every pair. Precision is the share of links that are right; recall the share
of same-subject pairs that were linked.

A config is chosen per embedder on the first video only -- the best F1 among
those with precision >= 0.9, since a wrong merge corrupts a narrative where a
missed one only shortens it -- and reported on the rest, which it never saw.
The aggregator runs on whatever embedder a deployment has, so the rules that
hold precision >= 0.9 under EVERY embedder are listed too: its default has to
be one of them.

Also scored: v0's linker, fixed thresholds 0.88 / 0.80 and cannot-link per
chunk, as the baseline this replaced.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations, product
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

LABELS = Path(__file__).with_name("entity_labels.json")


def score(mentions, groups, labels) -> dict[str, Any]:
    """Pairwise precision and recall, plus how many links nobody checked.

    `unchecked` counts links touching an unsure mention. It is reported
    because it is exactly where a wrong merge hides: the first real run scored
    precision 1.00 while merging a dark puffy coat and a cream coat into the
    woman in the gray top.
    """
    keys = [m.key for m in mentions]
    truth = {k: name for name, members in labels["groups"].items() for k in members}
    unsure = set(labels["unsure"])
    different = {k: set(v) for k, v in (labels.get("different") or {}).items()}
    unknown = sorted((set(truth) | unsure | set(different)) - set(keys))
    bad_groups = sorted({g for v in different.values() for g in v} - set(labels["groups"]))
    if unknown or bad_groups:
        raise ValueError(f"labels name what does not exist: {unknown + bad_groups}")
    predicted = {keys[i]: n for n, group in enumerate(groups) for i in group}
    tp = fp = fn = unchecked = 0
    for a, b in combinations(keys, 2):
        linked = predicted[a] == predicted[b]
        if a in unsure or b in unsure:
            doubtful, other = (a, b) if a in unsure else (b, a)
            if other not in unsure and truth.get(other) in different.get(doubtful, ()):
                fp += linked                    # known to be someone else
            else:
                unchecked += linked
            continue
        same = a in truth and b in truth and truth[a] == truth[b]
        tp += same and linked
        fp += linked and not same
        fn += same and not linked
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "unchecked": unchecked}


def v0_groups(mentions, vectors, link=0.88, generic=0.80) -> list[list[int]]:
    """v0's linker as it was: fixed thresholds, cannot-link per chunk."""
    v = np.asarray(vectors, dtype=float)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    sim = v @ v.T
    n = len(mentions)
    chunk = [m.chunk_id for m in mentions]
    distinctive = ((sim - np.eye(n)).sum(axis=1) / max(n - 1, 1)) < generic
    group_of, members = list(range(n)), {i: [i] for i in range(n)}
    pairs = [(sim[i, j], i, j) for i in range(n) for j in range(i + 1, n)
             if sim[i, j] >= link and chunk[i] != chunk[j] and distinctive[i] and distinctive[j]]
    for _, i, j in sorted(pairs, key=lambda p: -p[0]):
        a, b = group_of[i], group_of[j]
        if a == b or {chunk[k] for k in members[a]} & {chunk[k] for k in members[b]}:
            continue
        members[a].extend(members[b])
        for k in members[b]:
            group_of[k] = a
        del members[b]
    return [sorted(g) for g in members.values()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    from falconvar.aggregates.driver import context_for
    from falconvar.aggregates.linking import RULES, link, mentions_of
    from falconvar.video_rag.embed import embedders
    from falconvar.shared import env

    ap = argparse.ArgumentParser(description="Grade entity linking against hand labels.")
    ap.add_argument("--embedder", action="append", default=None,
                    help="repeatable; default openai and local")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    env.load()

    labels = json.loads(LABELS.read_text(encoding="utf-8"))["videos"]
    videos = list(labels)
    names = args.embedder or ["openai", "local"]

    mentions, vectors = {}, {}
    for vid in videos:
        context = context_for(vid)
        mentions[vid] = mentions_of(context.descriptions,
                                    lambda q, c=context: c.identity.get(q, {}))
        for name in names:
            built = embedders.build(name)
            vectors[(vid, built.key)] = built.embed([m.signature for m in mentions[vid]])
    keys = sorted({k for _, k in vectors})

    rows = []
    for key, rule, mutual in product(keys, RULES, (True, False)):
        row = {"embedder": key, "config": f"{rule}/{'mutual' if mutual else 'any'}"}
        for vid in videos:
            linked = link(mentions[vid], vectors[(vid, key)], rule, mutual)
            row[vid] = {**score(mentions[vid], linked.groups, labels[vid]),
                        "threshold": linked.threshold}
        rows.append(row)
    for key in keys:
        row = {"embedder": key, "config": "v0 (0.88 / 0.80, per chunk)"}
        for vid in videos:
            row[vid] = {**score(mentions[vid], v0_groups(mentions[vid], vectors[(vid, key)]),
                                labels[vid]), "threshold": 0.88}
        rows.append(row)

    tune, rest = videos[0], videos[1:]
    chosen = {}
    for key in keys:
        own = [r for r in rows if r["embedder"] == key and not r["config"].startswith("v0")]
        eligible = [r for r in own if r[tune]["precision"] >= 0.9]
        chosen[key] = max(eligible or own, key=lambda r: (r[tune]["f1"], r[tune]["precision"]))
    configs = sorted({r["config"] for r in rows if not r["config"].startswith("v0")})
    safe = [c for c in configs
            if all(r[tune]["precision"] >= 0.9 for r in rows if r["config"] == c)]

    if args.json:
        print(json.dumps({"rows": rows, "chosen": chosen, "safe_everywhere": safe,
                          "tuned_on": tune}, indent=2))
        return 0

    print(f"mentions: " + ", ".join(f"{v} {len(mentions[v])}" for v in videos))
    header = "  ".join(f"{v:>31}" for v in videos)
    print(f"\n{'embedder':<38} {'config':<28} {header}")
    print(f"{'':<38} {'':<28} " + "  ".join(f"{'P     R     F1   thr    unchk':>31}" for _ in videos))
    for r in rows:
        cells = "  ".join(
            f"{r[v]['precision']:.2f}  {r[v]['recall']:.2f}  {r[v]['f1']:.2f}  "
            f"{(r[v]['threshold'] or 0):.3f}  {r[v]['unchecked']:>3}".rjust(31) for v in videos)
        mark = f"  <- chosen for this embedder on {tune}" if chosen.get(r["embedder"]) is r else ""
        print(f"{r['embedder']:<38} {r['config']:<28} {cells}{mark}")
    for key, r in chosen.items():
        print(f"\nchosen on {tune} for {key}: {r['config']}")
        for vid in rest:
            s = r[vid]
            print(f"  on {vid} (not tuned on): precision {s['precision']:.2f}  "
                  f"recall {s['recall']:.2f}  f1 {s['f1']:.2f}  ({s['tp']} right links, "
                  f"{s['fp']} wrong, {s['fn']} missed, {s['unchecked']} unchecked)")
    print(f"\nprecision >= 0.9 on {tune} under every embedder: {', '.join(safe) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
