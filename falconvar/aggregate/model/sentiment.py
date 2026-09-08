"""Tone per chunk, and where it turns.

Signed, so a mean over the video is meaningful: two chunks at 0.9 positive and
0.9 negative should average to nothing, not to 0.9 confident-about-something."""

from __future__ import annotations

from typing import Any, Optional

from ..base import Context
from ..rendering import chunk_rows
from . import DEFAULT_SENTIMENT_MODEL, MAX_CHARS, ModelUnavailable

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


__all__ = ["SentimentAggregator"]
