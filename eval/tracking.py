"""Attribute tracking with a merge pass: which of the tracking fixes help.

    python -m eval.tracking

Builds on `eval.attributes` (same mentions, distance and scoring) and varies:

- how a detection is compared with a track: its last 2 appearances, or the
  whole track summarised as a mean / median / max;
- a gender veto against every appearance in the track (`unclear` never vetoes);
- a merge pass afterwards: tracks that never share an answer, do not clash and
  are close enough are joined, most compatible pair first.

Age group is not a veto. The describer's reading drifts for one person (adult /
older_adult, child / adult), and vetoing on it split real tracks and pushed
their mentions into wrong ones: F1 0.93 -> 0.81 on test1.

Gender is not a good veto either. Measured on test1, best of each:

    last 2, merge                      F1 0.94  wrong 4    the best
    last 2, no fixes                   F1 0.93  wrong 7
    last 2, gender veto + merge        F1 0.92  wrong 8
    whole track, gender veto + merge   F1 0.90  wrong 8

A veto only ever refuses a link, yet it adds wrong ones: a refused mention
starts or joins another track, and that track is the wrong person. Whole-track
matching did not beat last 2.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .attributes import DATA, LABELS, UNKNOWN, Corpus, use_data

SUMMARIES = ("last2", "mean", "median", "max")


def vetoes(c: Corpus) -> np.ndarray:
    """Pairs whose genders contradict."""
    g = [m.entry.get("gender", "unclear") for m in c.mentions]
    return np.array([[g[i] not in UNKNOWN and g[j] not in UNKNOWN and g[i] != g[j]
                      for j in range(c.n)] for i in range(c.n)])


def cost_to_track(D, V, members, d, how, veto) -> float:
    if veto and V[members, d].any():
        return np.inf
    if how == "last2":
        return min(D[m, d] for m in members[-2:])
    return float({"mean": np.mean, "median": np.median, "max": np.max}[how](D[members, d]))


def assign(c: Corpus, D, V, threshold, how, veto, gap=2):
    from scipy.optimize import linear_sum_assignment

    tracks = []
    for step, answer in enumerate(c.answers):
        dets = [i for i in range(c.n) if c.answer[i] == answer]
        live = [t for t in tracks if step - t["step"] <= gap]
        if live and dets:
            cost = np.array([[cost_to_track(D, V, t["members"], d, how, veto) for d in dets]
                             for t in live])
            taken = set()
            for r, col in zip(*linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e6))):
                if cost[r, col] < threshold:
                    live[r]["members"].append(dets[col])
                    live[r]["step"] = step
                    taken.add(dets[col])
            dets = [d for d in dets if d not in taken]
        tracks += [{"members": [d], "step": step} for d in dets]
    return [t["members"] for t in tracks]


def merge(c: Corpus, D, V, groups, threshold, how, veto):
    """Most compatible pair first, until nothing qualifies. Two tracks with a
    member in the same answer are different people by construction."""
    groups = [list(grp) for grp in groups]
    merged = 0
    while True:
        best = None
        for a, b in combinations(range(len(groups)), 2):
            A, B = groups[a], groups[b]
            if {c.answer[i] for i in A} & {c.answer[i] for i in B}:
                continue
            if veto and V[np.ix_(A, B)].any():
                continue
            block = D[np.ix_(A, B)]
            cost = float(block.min() if how == "last2" else
                         {"mean": np.mean, "median": np.median, "max": np.max}[how](block))
            if cost < threshold and (best is None or cost < best[0]):
                best = (cost, a, b)
        if best is None:
            return groups, merged
        _, a, b = best
        groups[a] += groups.pop(b)
        merged += 1


def run(c: Corpus, D, V, threshold, how="last2", veto=False, do_merge=True):
    groups = assign(c, D, V, threshold, how, veto)
    merged = 0
    if do_merge:
        groups, merged = merge(c, D, V, groups, threshold, how, veto)
    return groups, merged


def line(c: Corpus, groups, merged) -> str:
    people = sum(1 for grp in groups if len(grp) > 1)
    return c.line(groups, f"tracks>1 {people:>2}  merges {merged:>2}")


def errors(c: Corpus, D, groups) -> None:
    """Every wrong link, and how each person's mentions were split."""
    shown = ("gender", "age_group", "hair_color", "top_color", "top_type", "bottom_color")
    by = {m.key: m for m in c.mentions}
    index = {m.key: i for i, m in enumerate(c.mentions)}
    pred = {c.mentions[i].key: n for n, grp in enumerate(groups) for i in grp}
    labelled = [m.key for m in c.mentions if m.key in c.truth]

    def attrs(k):
        return "/".join(by[k].entry.get(f, "?") for f in shown)

    missed: dict[str, int] = {}
    for a, b in combinations(labelled, 2):
        if pred[a] == pred[b] and c.truth[a] != c.truth[b]:
            print(f"  WRONG d={D[index[a], index[b]]:+.1f} {c.truth[a]} ~ {c.truth[b]}: "
                  f"{a} [{attrs(a)}] ~ {b} [{attrs(b)}]")
        if c.truth[a] == c.truth[b] and pred[a] != pred[b]:
            missed[c.truth[a]] = missed.get(c.truth[a], 0) + 1
    print("  missed pairs by person:", missed)
    for person in missed:
        split: dict[int, list[str]] = {}
        for k in labelled:
            if c.truth[k] == person:
                split.setdefault(pred[k], []).append(k)
        print(f"    {person}: " + " | ".join(", ".join(ks) for ks in split.values()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Attribute tracking: veto and merge variants.")
    ap.add_argument("--data", type=Path, default=DATA, help="data root holding the video")
    ap.add_argument("--video", default="test1")
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--errors", type=float, default=None, metavar="THRESHOLD",
                    help="list wrong links for the best config at this threshold")
    args = ap.parse_args(argv)
    use_data(args.data)

    c = Corpus(args.video, args.labels)
    D, V = c.distances(), vetoes(c)

    print("== before: last 2 appearances, no veto, no merge")
    for t in (3, 4, 5):
        print(f"  thr<{t}  {line(c, *run(c, D, V, t, do_merge=False))}")

    print("\n== each change on its own and together")
    variants = (("last2", False, True, "merge"),
                ("last2", True, False, "gender veto"),
                ("last2", True, True, "merge + gender veto"),
                ("mean", False, False, "whole track"),
                ("mean", True, True, "whole track + veto + merge"))
    for t in (3, 4, 5):
        for how, veto, m, name in variants:
            print(f"  thr<{t}  {name:<28} {line(c, *run(c, D, V, t, how, veto, m))}")

    print("\n== every combination, best 10")
    rows = []
    for t in (2, 3, 4, 5, 6, 7):
        for how in SUMMARIES:
            for veto in (False, True):
                for m in (False, True):
                    groups, merged = run(c, D, V, t, how, veto, m)
                    p, r, f, fp = c.score(groups)
                    rows.append((-f, fp, t, how, veto, m, groups, merged))
    rows.sort(key=lambda row: row[:2])
    for _, _, t, how, veto, m, groups, merged in rows[:10]:
        print(f"  thr<{t} {how:<6} veto {'on ' if veto else 'off'} merge {'on ' if m else 'off'}  "
              f"{line(c, groups, merged)}")

    if args.errors is not None:
        best = min((row for row in rows if row[2] == args.errors), key=lambda row: row[:2])
        _, _, t, how, veto, m, groups, _ = best
        print(f"\n== errors, thr<{t} {how} veto {veto} merge {m}")
        errors(c, D, groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
