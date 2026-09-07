"""The local tier: GPU models over text already produced.

Between `free` (arithmetic) and `llm` (paid calls). These load a model and cost
electricity, but no money and no network once the weights are cached -- which
is the distinction the tier ladder is drawing.

Both read the *rendered* chunk text rather than the raw documents, so they see
what a reader would: the same lines `summary` and `chapters` are given.

Imported lazily. A `--tier free` run must not pull in torch.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from .base import Context
from .rendering import chunk_rows

#: What to look for. GLiNER is zero-shot, so the label set *is* the
#: configuration -- the same lesson as the open-vocabulary detector, where a
#: mismatched vocabulary found 2.4 detections per frame and a matched one 5.1.
DEFAULT_LABELS = ("person", "organisation", "location", "product",
                  "event", "date")

DEFAULT_NER_MODEL = "urchade/gliner_small-v2.1"
DEFAULT_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

#: Sentiment models are trained on single sentences and truncate hard. Scoring
#: a whole 20-second chunk in one pass reports the sentiment of its first
#: clause and calls it the chunk's.
MAX_CHARS = 480


class ModelUnavailable(RuntimeError):
    """No package, no weights, or a load that retrying will not fix."""


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


class SentimentAggregator:
    """How the tone moves across the video, chunk by chunk."""

    name = "sentiment"
    tier = "local"
    about = "tone per chunk, and where it turns"
    depends_on: tuple[str, ...] = ()

    def __init__(self, model: Optional[str] = None) -> None:
        self.model_name = model or DEFAULT_SENTIMENT_MODEL

    def _pipeline(self) -> Any:
        try:
            from transformers import pipeline
        except ImportError as exc:                       # pragma: no cover
            raise ModelUnavailable(
                "transformers is not installed") from exc
        try:
            return pipeline("sentiment-analysis", model=self.model_name,
                            truncation=True, max_length=512)
        except Exception as exc:                         # noqa: BLE001
            raise ModelUnavailable(
                f"could not load {self.model_name!r}: {exc}") from None

    def run(self, context: Context) -> dict[str, Any]:
        rows = chunk_rows(context)
        if not rows:
            return {"per_chunk": [], "chunks_read": 0}
        classify = self._pipeline()

        per_chunk = []
        for chunk_id, line in rows:
            result = classify(line[:MAX_CHARS])[0]
            start, end = context.span_of(chunk_id)
            # Signed, so a mean over the video is meaningful. Two chunks at
            # 0.9 positive and 0.9 negative should average to nothing, not to
            # 0.9 confident-about-something.
            signed = (result["score"] if result["label"].upper().startswith("POS")
                      else -result["score"])
            per_chunk.append({"chunk_id": chunk_id,
                              "start_ts": round(start, 3),
                              "end_ts": round(end, 3),
                              "label": result["label"].lower(),
                              "score": round(float(result["score"]), 4),
                              "signed": round(float(signed), 4)})

        signs = [c["signed"] for c in per_chunk]
        turns = [b["chunk_id"] for a, b in zip(per_chunk, per_chunk[1:])
                 if (a["signed"] < 0) != (b["signed"] < 0)]
        return {
            "per_chunk": per_chunk,
            "mean": round(sum(signs) / len(signs), 4) if signs else 0.0,
            "positive_chunks": sum(1 for s in signs if s > 0),
            "negative_chunks": sum(1 for s in signs if s < 0),
            # Where the tone flips. The interesting moments in a narrative are
            # usually next to one of these.
            "turning_points": turns,
            "chunks_read": len(rows),
            "model": self.model_name,
        }


__all__ = ["DEFAULT_LABELS", "ModelUnavailable", "NERAggregator",
           "SentimentAggregator"]
