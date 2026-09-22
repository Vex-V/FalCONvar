"""Link people by fixed-vocabulary attributes (test.md), against today's linker.

    python -m eval.attributes                  # one embedding call, for the cosine rows
    python -m eval.attributes --no-embed

test.md's proposal: the `people` shape answers in closed vocabularies (gender,
age group, hair, top and bottom colour and type, accessories), two readings of
one person are compared by a weighted distance over those fields, and answers
are walked in time order with a Hungarian assignment of detections to tracks.
`eval.tracking` builds on the distance and the scoring here.

Reads a PROTOTYPE data root, not `data/out`: test1 re-described with the
attribute fields, so the `activity` labels in `entity_labels.json` survived on
the real one. `--data` points elsewhere; it is set before `falconvar` is
imported, because `FALCONVAR_DATA` is read when the paths resolve.

Labels are `people_labels.json`: 15 people, 62 of 87 mentions across `yolo` and
`clip:yolo`, written by the builder after one run. A direction, not a result.
Measured: attributes separate same from different people at AUC 0.970, against
0.862 for word Jaccard over the same entries.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = HERE / "people_labels.json"
DATA = HERE.parent / "data" / "eval" / "people"

WEIGHTS = {"gender": 3.0, "age_group": 2.0, "hair_color": 2.5, "hair_style": 1.0,
           "top_color": 2.0, "top_type": 1.0, "bottom_color": 2.0, "bottom_type": 1.0}
UNKNOWN = {"unclear", "covered", ""}
#: Near values cost half: phrasing drift between two readings of one person.
NEAR = [{"dark_blue", "black"}, {"dark_blue", "blue"}, {"blue", "light_blue"},
        {"grey", "white"}, {"beige", "white"}, {"beige", "brown"}, {"red", "pink"},
        {"blonde", "grey"}, {"blonde", "brown"},
        {"jacket", "coat"}, {"jacket", "puffer"}, {"coat", "puffer"},
        {"sweater", "hoodie"}, {"sweater", "jacket"}, {"shirt", "t_shirt"},
        {"trousers", "jeans"}, {"trousers", "leggings"},
        {"teen", "young_adult"}, {"young_adult", "adult"}, {"adult", "older_adult"},
        {"short", "shaved"}, {"ponytail", "bun"}, {"long", "ponytail"}]
#: Things a person picks up and puts down, so sharing one is weaker evidence.
CARRIED = {"shopping_bag", "basket", "box", "stroller"}
WORD = re.compile(r"[a-z]+")
STOP = {"a", "an", "the", "and", "with", "of", "on", "in", "none", "nothing", "no"}


def use_data(root: Path) -> None:
    """Point `falconvar` at a data root. Must run before it is imported."""
    os.environ["FALCONVAR_DATA"] = str(root)


class Corpus:
    """One video's people mentions, its labels, and the answers in time order."""

    def __init__(self, video: str = "test1", labels: Path = LABELS):
        from falconvar.aggregates.definitions import Selection
        from falconvar.aggregates.driver import context_for
        from falconvar.aggregates.inputs import Source
        from falconvar.aggregates.entities.linking import mentions_of

        self.labels = json.loads(labels.read_text(encoding="utf-8"))
        self.truth = {m: person for person, ms in self.labels.items() for m in ms}
        self.mentions = mentions_of(context_for(video),
                                    Selection((Source("*"),), "people", ("appearance", "clothing")))
        missing = sorted(set(self.truth) - {m.key for m in self.mentions})
        if missing:
            raise SystemExit(f"labelled mentions not in the data: {missing}")
        self.n = len(self.mentions)
        self.answer = [m.answer for m in self.mentions]
        #: chunk order, and within a chunk `yolo` before `clip:yolo`.
        self.answers = sorted(set(self.answer), key=lambda a: (a[0], a[1] != "yolo", a[1]))

    def distances(self, **kw) -> np.ndarray:
        D = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                D[i, j] = D[j, i] = distance(self.mentions[i].entry, self.mentions[j].entry, **kw)
        return D

    def score(self, groups) -> tuple[float, float, float, int]:
        """Pairwise precision, recall, F1 and wrong links over labelled mentions."""
        pred = {self.mentions[i].key: n for n, grp in enumerate(groups) for i in grp}
        keys = [m.key for m in self.mentions if m.key in self.truth]
        tp = fp = fn = 0
        for a, b in combinations(keys, 2):
            same, linked = self.truth[a] == self.truth[b], pred[a] == pred[b]
            tp += same and linked
            fp += linked and not same
            fn += same and not linked
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 1.0
        return p, r, (2 * p * r / (p + r) if p + r else 0.0), fp

    def auc(self, sim: np.ndarray) -> float:
        """How often a same-person pair outscores a provably different one."""
        idx = {m.key: n for n, m in enumerate(self.mentions)}
        same = np.array([sim[idx[a], idx[b]] for g in self.labels.values()
                         for a, b in combinations(g, 2)])
        diff = np.array([sim[i, j] for i in range(self.n) for j in range(i + 1, self.n)
                         if self.answer[i] == self.answer[j]])
        return float((same[:, None] > diff[None, :]).mean()
                     + 0.5 * (same[:, None] == diff[None, :]).mean())

    def line(self, groups, extra: str = "") -> str:
        p, r, f, fp = self.score(groups)
        return f"P {p:.2f}  R {r:.2f}  F1 {f:.2f}  wrong {fp:>3}  {extra}"


def field_cost(a: str, b: str) -> float:
    if a in UNKNOWN or b in UNKNOWN or a == b:
        return 0.0
    return 0.5 if {a, b} in NEAR else 1.0


def distance(x: dict, y: dict, carried: bool = True, bonus: bool = True) -> float:
    d = sum(w * field_cost(x.get(f, "unclear"), y.get(f, "unclear")) for f, w in WEIGHTS.items())
    if bonus:
        shared = set(x.get("accessories") or []) & set(y.get("accessories") or [])
        if not carried:
            shared -= CARRIED
        d -= 2.0 * len(shared)
        wx = set(WORD.findall((x.get("distinguishing") or "").lower())) - STOP
        wy = set(WORD.findall((y.get("distinguishing") or "").lower())) - STOP
        if wx and wy and len(wx & wy) / len(wx | wy) >= 0.3:
            d -= 2.0
    return d


def hungarian(corpus: Corpus, D: np.ndarray, threshold: float, gap: int, recent: int = 2):
    """test.md's assignment: answers in time order, detections matched one-to-one
    to tracks seen within `gap` answers, against each track's last `recent`."""
    from scipy.optimize import linear_sum_assignment

    tracks = []
    for step, answer in enumerate(corpus.answers):
        dets = [i for i in range(corpus.n) if corpus.answer[i] == answer]
        live = [t for t in tracks if step - t["step"] <= gap]
        if live and dets:
            cost = np.array([[min(D[i, d] for i in t["members"][-recent:]) for d in dets]
                             for t in live])
            taken = set()
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] < threshold:
                    live[r]["members"].append(dets[c])
                    live[r]["step"] = step
                    taken.add(dets[c])
            dets = [d for d in dets if d not in taken]
        tracks += [{"members": [d], "step": step} for d in dets]
    return [t["members"] for t in tracks]


def rules(corpus: Corpus, sim: np.ndarray, rule: str):
    """Today's linker over any similarity: threshold read off within-answer
    pairs, mutual best matches, greedy merge under cannot-link."""
    n, answer = corpus.n, corpus.answer
    diff = [sim[i, j] for i in range(n) for j in range(i + 1, n) if answer[i] == answer[j]]
    thr = float(max(diff) if rule == "max" else np.quantile(diff, {"q95": .95, "q90": .9}[rule]))
    by: dict = {}
    for i, a in enumerate(answer):
        by.setdefault(a, []).append(i)

    def best(i, a):
        return max(by[a], key=lambda k: sim[i, k])

    pairs = sorted(((sim[i, j], i, j) for i in range(n) for j in range(i + 1, n)
                    if answer[i] != answer[j] and sim[i, j] > thr
                    and best(i, answer[j]) == j and best(j, answer[i]) == i),
                   key=lambda p: (-p[0], p[1], p[2]))
    group, members = list(range(n)), {i: [i] for i in range(n)}
    for _, i, j in pairs:
        a, b = group[i], group[j]
        if a == b or {answer[k] for k in members[a]} & {answer[k] for k in members[b]}:
            continue
        members[a] += members[b]
        for k in members[b]:
            group[k] = a
        del members[b]
    return list(members.values()), thr


def text_similarities(corpus: Corpus) -> dict[str, np.ndarray]:
    """Word Jaccard and colour+garment Dice over the signature text."""
    texts = [m.signature for m in corpus.mentions]
    n = corpus.n
    jac, cg = np.eye(n), np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a = set(WORD.findall(texts[i].lower())) - STOP
            b = set(WORD.findall(texts[j].lower())) - STOP
            jac[i, j] = jac[j, i] = len(a & b) / len(a | b)
            A = colour_garment(corpus.mentions[i].entry)
            B = colour_garment(corpus.mentions[j].entry)
            cg[i, j] = cg[j, i] = 2 * len(A & B) / (len(A) + len(B)) if A or B else 0
    return {"word jaccard (text)": jac, "colour+garment dice (text)": cg}


COLOURS = {"black", "white", "grey", "gray", "dark", "light", "navy", "blue", "red", "maroon",
           "burgundy", "pink", "green", "olive", "khaki", "tan", "beige", "cream", "brown",
           "yellow", "orange", "purple"}
GARMENTS = {"jacket", "coat", "top", "shirt", "sweater", "hoodie", "hooded", "cardigan", "puffer",
            "vest", "trousers", "pants", "jeans", "leggings", "skirt", "dress", "shorts", "shoes",
            "trainers", "sneakers", "boots", "cap", "beanie", "hat", "scarf", "bag", "backpack"}


def colour_garment(entry: dict) -> set[str]:
    toks = WORD.findall(f"{entry.get('clothing', '')} {entry.get('appearance', '')}".lower())
    out: set[str] = set()
    for i, t in enumerate(toks):
        if t in GARMENTS:
            out |= {f"{c} {t}" for c in toks[max(0, i - 3):i] if c in COLOURS} or {t}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Attribute linking (test.md) against today's linker.")
    ap.add_argument("--data", type=Path, default=DATA, help="data root holding the video")
    ap.add_argument("--video", default="test1")
    ap.add_argument("--labels", type=Path, default=LABELS)
    ap.add_argument("--no-embed", action="store_true", help="skip the cosine rows")
    args = ap.parse_args(argv)
    use_data(args.data)
    from falconvar.shared import env
    env.load()

    c = Corpus(args.video, args.labels)
    print(f"{c.n} mentions in {len(c.answers)} answers; labelled {len(c.truth)} in "
          f"{len(c.labels)} people, "
          f"{sum(len(g) * (len(g) - 1) // 2 for g in c.labels.values())} same-person pairs")
    print("unclear per field:",
          {f: sum(1 for m in c.mentions if m.entry.get(f) in UNKNOWN) for f in WEIGHTS})

    sims = {}
    if not args.no_embed:
        from falconvar.video_rag.embed import embedders
        v = np.asarray(embedders.build("openai").embed([m.signature for m in c.mentions]))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        sims["cosine openai (text)"] = v @ v.T
    sims.update(text_similarities(c))
    D = {"test.md": c.distances(), "no carried-item bonus": c.distances(carried=False),
         "no bonuses": c.distances(bonus=False)}
    for name, d in D.items():
        sims[f"attributes, {name}"] = -d

    print("\n== separation (AUC: a same-person pair beats a provably different one)")
    for name, sim in sims.items():
        print(f"  {name:<44} AUC {c.auc(sim):.3f}")

    print("\n== today's rules (calibrated threshold, mutual, greedy)")
    for name, sim in sims.items():
        for rule in ("max", "q95", "q90"):
            groups, thr = rules(c, sim, rule)
            print(f"  {name + ' ' + rule:<44} {c.line(groups, f'thr {thr:.3f}')}")

    print("\n== test.md: Hungarian assignment over answers in time order, best 12")
    results = [(c.score(groups)[2], f"{name} thr<{t} gap {g}", groups)
               for name, d in D.items() for t in (0.5, 1, 1.5, 2, 3, 4, 5, 6) for g in (1, 2, 4, 8)
               for groups in [hungarian(c, d, t, g)]]
    for _, name, groups in sorted(results, key=lambda r: -r[0])[:12]:
        print(f"  {name:<44} {c.line(groups)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
