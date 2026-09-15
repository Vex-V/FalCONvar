"""Which mentions across chunks are the same person or thing.

Decided by embedding what identifies each mention and merging under rules --
never by asking a model. Given vague descriptions a model links too eagerly:
measured on test1, gpt-5.4-mini linked 31 of 39 observations, broke the
same-answer rule twice after being told it, and merged an older woman with an
older man. The model writes each entity's narrative afterwards, which is what
it is good at.

**The rules do the work, because similarity alone cannot.** Provably different
people (two entries in one answer) score as high as the same person seen twice,
so no fixed threshold separates them:

    cannot-link   two entries in one answer are different: the question asks
                  for one entry per distinct person. Per ANSWER, not per chunk
                  -- two questions about one chunk may describe the same person
    calibrated    the threshold is read off those provably different pairs, in
                  this video, with this embedder. A fixed number is a fact about
                  one embedder: v0's 0.88 linked 47% of true pairs on
                  text-embedding-3-small and 63% on bge
    mutual        a pair links only if each is the other's best match in the
                  other's answer, so a vague description cannot attach to
                  whatever it drifted closest to
    merge         greedy, most similar first, and a merge that would put two
                  entries of one answer in one entity is refused

Only fields a shape declares as identity are read (`clothing`, `appearance`),
never what someone is doing: "woman entering from right" matched "woman
entering/exiting" on behaviour, not on who she was.

v0's genericness filter -- a description resembling everything stays unlinked
-- was tried as an outlier test on mean similarity and changed no result on
either labelled video, under either embedder, so it is not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

#: How the threshold is read off the provably-different pairs: their maximum,
#: or a quantile of them.
RULES = ("max", "q95", "q90")


@dataclass
class Mention:
    """One entry in one answer's identity field."""

    chunk_id: int
    sampler_id: str
    field: str
    index: int
    signature: str
    entry: dict[str, Any]

    @property
    def key(self) -> str:
        return f"c{self.chunk_id}/{self.sampler_id}/{self.field}/{self.index}"

    @property
    def answer(self) -> tuple[int, str, str]:
        """The list this entry came from. Two entries of one list differ."""
        return (self.chunk_id, self.sampler_id, self.field)


@dataclass
class Linked:
    groups: list[list[int]]
    threshold: Optional[float]
    calibration_pairs: int
    candidate_pairs: int


def mentions_of(descriptions: Any,
                identity_of: Callable[[str], dict[str, list[str]]]) -> list[Mention]:
    """Every identity-bearing entry, in chunk order.

    `identity_of(question)` is `{field: [keys]}`. A question whose shape
    declares none contributes nothing -- there is no guessing which keys
    identify someone.
    """
    out: list[Mention] = []
    for chunk in descriptions.chunks:
        for sampler_id, block in (chunk.get("samplers") or {}).items():
            identity = identity_of(block.get("question") or sampler_id)
            structured = block.get("structured") or {}
            for field_name, keys in identity.items():
                for index, item in enumerate(structured.get(field_name) or []):
                    if not isinstance(item, dict):
                        continue
                    signature = "; ".join(str(item[k]).strip() for k in keys
                                          if str(item.get(k) or "").strip())
                    if signature:
                        out.append(Mention(chunk["chunk_id"], sampler_id,
                                           field_name, index, signature, item))
    return out


def link(mentions: Sequence[Mention], vectors: Sequence[Sequence[float]],
         rule: str = "max", mutual: bool = True) -> Linked:
    """Group mentions of the same subject. Indices into `mentions`."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {', '.join(RULES)}")
    count = len(mentions)
    if count == 0:
        return Linked([], None, 0, 0)

    matrix = np.asarray(vectors, dtype=float)
    matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    sim = matrix @ matrix.T
    answers = [m.answer for m in mentions]

    # Calibration: pooled over the whole video and every field, because a
    # field with one entry per answer has no different-pairs of its own.
    different = [sim[i, j] for i in range(count) for j in range(i + 1, count)
                 if answers[i] == answers[j]]
    if not different:
        # Nothing here is provably different, so nothing says how similar is
        # similar enough. Linking anyway would be a guess.
        return Linked([[i] for i in range(count)], None, 0, 0)
    threshold = float(max(different) if rule == "max"
                      else np.quantile(different, {"q95": 0.95, "q90": 0.90}[rule]))

    by_field: dict[str, list[int]] = {}
    for i, mention in enumerate(mentions):
        by_field.setdefault(mention.field, []).append(i)
    in_answer: dict[tuple[int, str, str], list[int]] = {}
    for i, answer in enumerate(answers):
        in_answer.setdefault(answer, []).append(i)

    def best(i: int, answer: tuple[int, str, str]) -> int:
        return max(in_answer[answer], key=lambda k: sim[i, k])

    pairs = []
    for indices in by_field.values():
        for position, i in enumerate(indices):
            for j in indices[position + 1:]:
                if answers[i] == answers[j] or sim[i, j] <= threshold:
                    continue
                if mutual and (best(i, answers[j]) != j or best(j, answers[i]) != i):
                    continue
                pairs.append((float(sim[i, j]), i, j))

    group_of = list(range(count))
    members = {i: [i] for i in range(count)}
    for _, i, j in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        a, b = group_of[i], group_of[j]
        if a == b:
            continue
        if {answers[k] for k in members[a]} & {answers[k] for k in members[b]}:
            continue
        members[a].extend(members[b])
        for k in members[b]:
            group_of[k] = a
        del members[b]

    groups = sorted((sorted(g) for g in members.values()), key=lambda g: g[0])
    return Linked(groups, threshold, len(different), len(pairs))


__all__ = ["Linked", "Mention", "RULES", "link", "mentions_of"]
