"""Named entities across everything that was written about the video.

Runs locally on the GPU, so it costs nothing per video and can be re-run freely
with a different label set -- which matters, because the useful labels are
domain-specific and nobody gets them right first time. GLiNER takes its labels
as text at inference, so changing them is a parameter rather than a retrained
model.

**Reads the pictures as well as the speech.** A brand on a package, a name on a
sign and a place mentioned aloud are all entities, and the `text` sampler's
verbatim signage is often the densest source of them in surveillance footage
where nobody says anything. Restricting this to the transcript would throw away
the half of the corpus that has no transcript at all.

Pronouns and bare determiners are dropped. GLiNER will happily label "he" as a
person, which is true and useless: an entity index exists to be looked up, and
nobody looks up "he".
"""

from __future__ import annotations

import collections
from typing import Any, Optional

from .base import Context

MODEL = "urchade/gliner_multi-v2.1"

#: Deliberately broad. Narrow labels miss; overlapping ones are cheap because
#: the model scores each independently and the caller can filter after.
DEFAULT_LABELS = (
    "person", "organization", "location", "product", "brand",
    "date", "time", "money", "vehicle", "job title", "event",
)

PREFERENCE = ("transcript", "text", "overview", "clip", "uniform", "yolo", "objects")

STOP = frozenset("""
i you he she it we they me him her us them my your his its our their this that
these those someone somebody anyone everyone one who whom whose there here
""".split())

_model = None


def _load(name: str):
    """One process-wide model. Loading it twice is a second copy in VRAM."""
    global _model
    if _model is None:
        try:
            from gliner import GLiNER
        except ImportError as exc:                      # pragma: no cover
            raise RuntimeError(
                "gliner is not installed: pip install gliner") from exc
        import torch

        _model = GLiNER.from_pretrained(name)
        if torch.cuda.is_available():
            _model = _model.to("cuda")
        _model.eval()
    return _model


class NERAggregator:
    """Every named entity mentioned or shown, with where it appeared."""

    id = "ner"
    tier = "local"
    depends_on = ()

    def __init__(self, model: str = MODEL, labels: tuple[str, ...] = DEFAULT_LABELS,
                 threshold: float = 0.5) -> None:
        self.model_name = model
        self.labels = list(labels)
        self.threshold = threshold

    def aggregate(self, ctx: Context) -> Optional[dict[str, Any]]:
        from .llm import pick_sources

        sources = pick_sources(ctx, PREFERENCE)
        passages = [(c["chunk_id"], ctx.text_of(c, sources)) for c in ctx.chunks]
        passages = [(cid, text) for cid, text in passages if text.strip()]
        if not passages:
            return None

        model = _load(self.model_name)
        found: dict[tuple[str, str], dict] = {}
        for chunk_id, text in passages:
            # GLiNER's window is finite and a chunk's combined text can be
            # long; a truncated passage silently loses its tail's entities.
            for piece in _windows(text, 1800):
                for entity in model.predict_entities(piece, self.labels,
                                                     threshold=self.threshold):
                    name = entity["text"].strip()
                    if len(name) < 2 or name.lower() in STOP:
                        continue
                    key = (name.lower(), entity["label"])
                    record = found.setdefault(key, {
                        "text": name, "label": entity["label"],
                        "mentions": 0, "score": 0.0, "chunk_ids": []})
                    record["mentions"] += 1
                    record["score"] = max(record["score"], round(float(entity["score"]), 3))
                    if chunk_id not in record["chunk_ids"]:
                        record["chunk_ids"].append(chunk_id)

        if not found:
            return None
        entities = sorted(found.values(),
                          key=lambda e: (-e["mentions"], -e["score"], e["text"]))
        by_label: dict[str, list] = collections.defaultdict(list)
        for entity in entities:
            by_label[entity["label"]].append(entity["text"])
        return {"entities": entities, "count": len(entities),
                "by_label": {k: v[:25] for k, v in sorted(by_label.items())},
                "labels": self.labels, "based_on": sources}

    def config(self) -> dict[str, Any]:
        return {"id": self.id, "tier": self.tier,
                "params": {"model": self.model_name, "labels": self.labels,
                           "threshold": self.threshold}}


def _windows(text: str, size: int) -> list[str]:
    """Split on whitespace near `size`, so a window never bisects a word."""
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        out.append(text[start:end])
        start = end
    return out
