"""Which mentions across chunks are the same person or thing.

Decided by embedding what identifies each mention and merging under rules --
never by asking a model. Given vague descriptions a model links too eagerly:
measured on test1, gpt-5.4-mini linked 31 of 39 observations, broke the
same-answer rule twice after being told it, and merged an older woman with an
older man. The model writes each entity's account afterwards, which is what it
is good at.

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

Only the keys a link profile names as identity are read (`clothing`,
`appearance`), never what someone is doing: "woman entering from right" matched
"woman entering/exiting" on behaviour, not on who she was.

**Whole values have no calibration set.** A profile linking a text field, the
prose or the transcript yields one mention per answer, so nothing is provably
different from anything else -- and the profile has to give a `threshold`.

v0's genericness filter -- a description resembling everything stays unlinked
-- was tried as an outlier test on mean similarity and changed no result on
either labelled video, under either embedder, so it is not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

from ...shared.contracts.documents import fingerprint_of

if TYPE_CHECKING:
    from ..base import Context

#: How the threshold is read off the provably-different pairs: their maximum,
#: or a quantile of them.
RULES = ("max", "q95", "q90")


@dataclass
class Mention:
    """One entry in one answer's field -- or, for a whole value, the answer."""

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


@dataclass
class Mentions:
    """One profile's selection, read against one video."""

    selection: Any
    items: list[Mention]

    @property
    def empty(self) -> bool:
        return not self.items

    @property
    def why_empty(self) -> str:
        what = (f"`{self.selection.field}` entries carrying "
                f"{', '.join(self.selection.keys)}" if self.selection.keys
                else f"a value for `{self.selection.field}`")
        heads = "+".join(s.head for s in self.selection.sources)
        return f"no answer read by `{heads}` has {what}"

    @property
    def chars(self) -> int:
        return sum(len(m.signature) for m in self.items)

    def fingerprint(self) -> str:
        """Everything a mention carries, not just its signature: the account
        is written from the other keys too."""
        return fingerprint_of({"mentions": [[m.key, m.signature, m.entry]
                                            for m in self.items]})


def mentions_of(context: "Context", selection: Any) -> list[Mention]:
    """Every mention a profile's selection finds, in chunk order.

    Entries when the selection names keys: each object in the list field of
    every matching answer, signed by those keys. A question whose answer has no
    such list contributes nothing -- there is no guessing which keys identify
    someone. Otherwise one mention per matching answer, carrying the field's
    whole value, the prose (`summary`) or the chunk's transcript.
    """
    from ...shared.contracts.units import render
    from ..inputs import PROSE

    field, keys = selection.field, tuple(selection.keys)
    out: list[Mention] = []
    if field == "transcript":
        for chunk_id in context.chunk_ids():
            said = (context.transcript.text_of(chunk_id) or "").strip() \
                if context.transcript is not None else ""
            if said:
                out.append(Mention(chunk_id, "transcript", field, 0, said,
                                   {"transcript": said}))
        return out

    for chunk in (context.descriptions.chunks if context.descriptions else []):
        for sampler_id, block in (chunk.get("samplers") or {}).items():
            if not any(s.matches(sampler_id, block) for s in selection.sources):
                continue
            structured = block.get("structured") or {}
            if not keys:
                value = (block.get("description") if field == PROSE
                         else structured.get(field))
                said = (value.strip() if isinstance(value, str) else
                        render("", {field: value}).removeprefix(f"{field}: "))
                if said:
                    out.append(Mention(chunk["chunk_id"], sampler_id, field, 0,
                                       said, {field: said}))
                continue
            for index, item in enumerate(structured.get(field) or []):
                if not isinstance(item, dict):
                    continue
                signature = "; ".join(str(item[k]).strip() for k in keys
                                      if str(item.get(k) or "").strip())
                if signature:
                    out.append(Mention(chunk["chunk_id"], sampler_id, field,
                                       index, signature, item))
    return out


def link(mentions: Sequence[Mention], vectors: Sequence[Sequence[float]],
         rule: str = "max", mutual: bool = True,
         threshold: Optional[float] = None) -> Linked:
    """Group mentions of the same subject. Indices into `mentions`.

    `threshold` fixes the bar instead of reading it off the video, for whole
    values where nothing is provably different.
    """
    import numpy as np

    if rule not in RULES:
        raise ValueError(f"rule must be one of {', '.join(RULES)}")
    count = len(mentions)
    if count == 0:
        return Linked([], threshold, 0, 0)

    matrix = np.asarray(vectors, dtype=float)
    matrix = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    sim = matrix @ matrix.T
    answers = [m.answer for m in mentions]

    # Calibration: pooled over the whole video and every field, because a
    # field with one entry per answer has no different-pairs of its own.
    different = [sim[i, j] for i in range(count) for j in range(i + 1, count)
                 if answers[i] == answers[j]]
    if threshold is None:
        if not different:
            # Nothing here is provably different, so nothing says how similar
            # is similar enough. Linking anyway would be a guess.
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
    return Linked(groups, float(threshold), len(different), len(pairs))


__all__ = ["Linked", "Mention", "Mentions", "RULES", "link", "mentions_of"]
