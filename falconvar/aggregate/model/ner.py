"""Named entities, and which chunks each appears in.

GLiNER is zero-shot, so the label set IS the configuration -- the same lesson
as the open-vocabulary detector, where a mismatched vocabulary found 2.4
detections per frame and a matched one 5.1."""

from __future__ import annotations

from typing import Any, Optional
from collections import Counter

from ..base import Context
from ..rendering import chunk_rows
from . import DEFAULT_LABELS, DEFAULT_NER_MODEL, ModelUnavailable

class NERAggregator:
    """Named entities across the video, with the chunks they appear in."""

    name = "ner"
    tier = "local"
    about = "named entities, and which chunks each appears in"
    depends_on: tuple[str, ...] = ()

    def __init__(self, model: Optional[str] = None,
                 labels: tuple[str, ...] = DEFAULT_LABELS,
                 threshold: float = 0.5) -> None:
        self.model_name = model or DEFAULT_NER_MODEL
        self.labels = list(labels)
        self.threshold = threshold

    def _model(self) -> Any:
        try:
            from gliner import GLiNER
        except ImportError as exc:                       # pragma: no cover
            raise ModelUnavailable(
                "gliner is not installed: pip install gliner") from exc
        try:
            return GLiNER.from_pretrained(self.model_name)
        except Exception as exc:                         # noqa: BLE001
            raise ModelUnavailable(
                f"could not load {self.model_name!r}: {exc}") from None

    def run(self, context: Context) -> dict[str, Any]:
        rows = chunk_rows(context)
        if not rows:
            return {"entities": [], "count": 0, "chunks_read": 0}
        model = self._model()

        # text -> {label, chunks}. Grouped by surface form, because "which
        # chunks does this name appear in" is the question an aggregate can
        # answer that retrieval cannot.
        found: dict[tuple[str, str], set[int]] = {}
        for chunk_id, line in rows:
            for entity in model.predict_entities(line, self.labels,
                                                 threshold=self.threshold):
                key = (entity["text"].strip(), entity["label"])
                if key[0]:
                    found.setdefault(key, set()).add(chunk_id)

        entities = [
            {"text": text, "label": label,
             "chunk_ids": sorted(chunks), "mentions": len(chunks)}
            for (text, label), chunks in found.items()
        ]
        entities.sort(key=lambda e: (-e["mentions"], e["text"].lower()))
        return {
            "entities": entities,
            "count": len(entities),
            "by_label": dict(Counter(e["label"] for e in entities)),
            "chunks_read": len(rows),
            "model": self.model_name,
            "labels": self.labels,
        }


__all__ = ["NERAggregator"]
