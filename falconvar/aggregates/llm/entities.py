"""The same person or thing across chunks, and what each did.

Identity is decided by `aggregate.linking` -- embeddings under rules, no model.
The model is asked only afterwards, once per linked entity, to write what that
entity did across the video; those calls run concurrently.

Also answered here, because they fall out of the linking for free: how long
each entity was in shot, and which entities were in shot together.
"""

from __future__ import annotations

import asyncio
from itertools import combinations
from typing import Any, Optional

from ...shared.contracts.documents import fingerprint_of
from ...shared.models.llm import Model
from ..base import Context
from ..linking import link, mentions_of
from ..rendering import resolve_span

_NARRATIVE_SCHEMA = {
    "name": "entity_narrative",
    "schema": {
        "type": "object", "additionalProperties": False,
        "required": ["description", "narrative", "role"],
        "properties": {
            "description": {"type": "string",
                            "description": "How to recognise this one, in one line."},
            "narrative": {"type": "string",
                          "description": "What they did across the video, in order."},
            "role": {"type": "string",
                     "description": "Their apparent role, e.g. customer, staff."},
        },
    },
}

_NARRATIVE_PROMPT = """\
Below are separate observations of what appears to be the same subject, from \
consecutive segments of one video, in time order.

Write a single account of what they did across the video, a one-line \
description of how to recognise them, and their apparent role.

Only use what is below. If the observations conflict, prefer what appears most \
often and do not invent a reason for the difference. Never give a name.

{observations}"""


class EntitiesAggregator:
    name = "entities"
    tier = "llm"
    about = "the same person or thing across chunks, and what each did"
    depends_on: tuple[str, ...] = ()
    #: Takes an embedder as well as an llm; the driver passes both.
    embeds = True

    def __init__(self, llm: Optional[str] = None, embedder: Optional[str] = None,
                 rule: str = "max", mutual: bool = True,
                 min_appearances: int = 2, max_narratives: int = 12) -> None:
        # The embedder is resolved exactly as video_rag's `embed` resolves one,
        # and reached through its driver -- the other tier, never a component.
        # Imported here so `--tier free` never loads a client.
        from ...video_rag import driver as video_rag
        self.llm = Model(llm, role="llm")
        self.embedder = video_rag.embedder(embedder)
        self.rule, self.mutual = rule, mutual
        self.min_appearances = min_appearances
        self.max_narratives = max_narratives

    @property
    def model_key(self) -> str:
        """Everything that decides the answer besides the text: who writes the
        narratives, which space identity is measured in, and the rules."""
        return (f"{self.llm.key}|{self.embedder.key}|{self.rule}/"
                f"{'mutual' if self.mutual else 'any'}")

    def inputs_of(self, context: Context) -> Any:
        """What identifies a subject is declared by the shapes, and changing
        that declaration changes the answer without changing any text."""
        return context.identity

    def run(self, context: Context) -> dict[str, Any]:
        return asyncio.run(self._run(context))

    async def _run(self, context: Context) -> dict[str, Any]:
        mentions = ([] if context.descriptions is None else
                    mentions_of(context.descriptions,
                                lambda q: context.identity.get(q, {})))
        linking = {"embedder": self.embedder.key, "rule": self.rule,
                   "mutual": self.mutual}
        if len(mentions) < 2:
            return {"entities": [], "count": 0, "linked": 0,
                    "mentions": len(mentions), "together": [], "linking": linking,
                    "note": ("no question asked here declares identity fields"
                             if not mentions else "one mention; nothing to link")}

        vectors = self.embedder.embed([m.signature for m in mentions])
        linked = link(mentions, vectors, self.rule, self.mutual)

        entities = []
        for group in linked.groups:
            ordered = sorted(group, key=lambda i: (mentions[i].chunk_id, mentions[i].sampler_id))
            chunk_ids = sorted({mentions[i].chunk_id for i in ordered})
            start, end = resolve_span(context, chunk_ids)
            entities.append({
                "field": mentions[ordered[0]].field,
                "label": max((mentions[i].signature for i in ordered), key=len),
                "appearances": len(chunk_ids),
                "chunk_ids": chunk_ids,
                "start_ts": round(start, 3), "end_ts": round(end, 3),
                "observed_s": round(sum(context.span_of(c)[1] - context.span_of(c)[0]
                                        for c in chunk_ids), 3),
                "mentions": [{"key": mentions[i].key, "chunk_id": mentions[i].chunk_id,
                              "sampler_id": mentions[i].sampler_id, **mentions[i].entry}
                             for i in ordered],
            })
        entities.sort(key=lambda e: (-e["appearances"], e["start_ts"]))
        for number, entity in enumerate(entities):
            entity["entity_id"] = f"e{number:03d}"

        narrate = [e for e in entities
                   if e["appearances"] >= self.min_appearances][:self.max_narratives]
        stories = await asyncio.gather(*(self._narrate(context, e) for e in narrate))
        for entity, story in zip(narrate, stories):
            entity.update(story)

        present = [e for e in entities if e["appearances"] >= self.min_appearances]
        together = sorted((
            {"a": a["entity_id"], "b": b["entity_id"], "chunk_ids": shared,
             "count": len(shared)}
            for a, b in combinations(present, 2)
            if (shared := sorted(set(a["chunk_ids"]) & set(b["chunk_ids"])))),
            key=lambda p: (-p["count"], p["a"], p["b"]))

        return {
            "entities": entities,
            "count": len(entities),
            "linked": len(present),
            "mentions": len(mentions),
            "narrated": len(narrate),
            "together": together[:50],
            "linking": {**linking, "threshold": linked.threshold,
                        "calibration_pairs": linked.calibration_pairs,
                        "candidate_pairs": linked.candidate_pairs,
                        "identity": fingerprint_of(context.identity)},
        }

    async def _narrate(self, context: Context, entity: dict[str, Any]) -> dict[str, Any]:
        lines = []
        for mention in entity["mentions"]:
            start, end = context.span_of(mention["chunk_id"])
            said = " | ".join(f"{k}: {v}" for k, v in sorted(mention.items())
                              if k not in ("key", "chunk_id", "sampler_id"))
            lines.append(f"[{start:.0f}-{end:.0f}s] {said}")
        return await self.llm.complete(
            _NARRATIVE_PROMPT.format(observations="\n".join(lines)), _NARRATIVE_SCHEMA)


__all__ = ["EntitiesAggregator"]
