"""Tone per chunk, and where it turns.

Signed, so a mean over the video is meaningful: two chunks at 0.9 positive and
0.9 negative should average to nothing, not to 0.9 confident-about-something.

A chunk is scored over every piece of its text, weighted by length -- never by
its first 480 characters."""

from __future__ import annotations

from typing import Any, Optional

from ...shared.contracts.documents import fingerprint_of
from ..base import Context, ModelUnavailable
from ..inputs import Input, Read, read
from ..rendering import pieces, plain

DEFAULT_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

#: The longest piece a sentiment model is handed. Trained on single sentences.
MAX_CHARS = 480


class SentimentAggregator:
    """How the tone moves across the video, chunk by chunk."""

    name = "sentiment"
    tier = "local"
    about = "tone per chunk, and where it turns"
    depends_on: tuple[str, ...] = ()
    takes_inputs = True

    def __init__(self, model: Optional[str] = None) -> None:
        self.model_name = model or DEFAULT_SENTIMENT_MODEL

    @property
    def version(self) -> str:
        return fingerprint_of({"model": self.model_name, "chars": MAX_CHARS})

    def _pipeline(self) -> Any:
        from transformers import pipeline
        try:
            return pipeline("sentiment-analysis", model=self.model_name,
                            truncation=True, max_length=512)
        except Exception as exc:                         # noqa: BLE001
            raise ModelUnavailable(
                f"could not load {self.model_name!r}: {exc}") from None

    def read(self, context: Context, one: Input) -> Read:
        return read(context, one)

    def run(self, context: Context, read: Read) -> dict[str, Any]:
        classify = self._pipeline()
        per_chunk = []
        for row in read.rows:
            parts = pieces(plain(row), MAX_CHARS)
            results = classify(parts)
            weights = [len(p) for p in parts]
            signed = sum(w * (r["score"] if r["label"].upper().startswith("POS")
                              else -r["score"])
                         for w, r in zip(weights, results)) / sum(weights)
            per_chunk.append({"chunk_id": row.chunk_id,
                              "start_ts": round(row.start, 3),
                              "end_ts": round(row.end, 3),
                              "label": "positive" if signed >= 0 else "negative",
                              "score": round(abs(float(signed)), 4),
                              "signed": round(float(signed), 4),
                              "pieces": len(parts)})

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
            "chunks_read": len(read.rows),
            "model": self.model_name,
        }


__all__ = ["SentimentAggregator"]
