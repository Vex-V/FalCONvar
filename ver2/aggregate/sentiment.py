"""The tone of the language people used, per speaker and over time.

**Speech only, and that is a correctness decision rather than a limitation.** A
vision model's description of a shop floor has no affect to measure. Scoring
"a small retail shop is shown from a high corner camera" produces a number that
looks meaningful and is not, and once it is in a column somebody will average
it. So this returns None on a video with no transcript rather than scoring the
descriptions.

**Reported as language observed, never as emotion felt.** The model reads
words, ASR output is noisy, and a flat delivery of alarming content scores
however the words score. "Negative language" is a claim that survives scrutiny;
"the speaker was unhappy" is not, and the field names say so.
"""

from __future__ import annotations

import collections
from typing import Any, Optional

from .base import Context

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_pipeline = None


def _load(name: str):
    global _pipeline
    if _pipeline is None:
        import torch
        from transformers import pipeline

        _pipeline = pipeline("sentiment-analysis", model=name,
                             device=0 if torch.cuda.is_available() else -1,
                             truncation=True)
    return _pipeline


class SentimentAggregator:
    """Sentiment of transcribed speech, by turn, speaker and chunk."""

    id = "sentiment"
    tier = "local"
    depends_on = ("transcript",)

    def __init__(self, model: str = MODEL, min_words: int = 4) -> None:
        self.model_name = model
        # Two words carry no measurable tone; scoring them adds noise with the
        # same confidence as a real reading.
        self.min_words = min_words

    def aggregate(self, ctx: Context) -> Optional[dict[str, Any]]:
        turns = [{"chunk_id": chunk["chunk_id"], "speaker": turn.get("speaker"),
                  "start": turn["start"], "end": turn["end"],
                  "text": (turn.get("text") or "").strip()}
                 for chunk in ctx.chunks for turn in chunk["turns"]]
        turns = [t for t in turns if len(t["text"].split()) >= self.min_words]
        if not turns:
            return None

        model = _load(self.model_name)
        scores = model([t["text"][:1000] for t in turns])

        scored = []
        for turn, score in zip(turns, scores):
            label = str(score["label"]).lower()
            scored.append({**turn, "label": label,
                           "confidence": round(float(score["score"]), 3)})

        def share(rows: list[dict]) -> dict[str, Any]:
            counts = collections.Counter(r["label"] for r in rows)
            total = len(rows) or 1
            return {"turns": len(rows),
                    "counts": dict(counts),
                    "shares": {k: round(v / total, 3) for k, v in counts.items()},
                    "dominant": counts.most_common(1)[0][0] if counts else None}

        by_speaker = collections.defaultdict(list)
        by_chunk = collections.defaultdict(list)
        for row in scored:
            by_speaker[row["speaker"] or "unattributed"].append(row)
            by_chunk[row["chunk_id"]].append(row)

        return {
            "overall": share(scored),
            "by_speaker": {k: share(v) for k, v in sorted(by_speaker.items())},
            "by_chunk": {str(k): share(v) for k, v in sorted(by_chunk.items())},
            "turns": [{"chunk_id": r["chunk_id"], "speaker": r["speaker"],
                       "start_ts": round(r["start"], 2), "end_ts": round(r["end"], 2),
                       "label": r["label"], "confidence": r["confidence"],
                       "text": r["text"][:160]} for r in scored],
            "measures": "language observed in transcribed speech, not emotion felt",
            "scored_turns": len(scored),
            "skipped_short": len([1 for c in ctx.chunks for t in c["turns"]]) - len(scored),
        }

    def config(self) -> dict[str, Any]:
        return {"id": self.id, "tier": self.tier,
                "params": {"model": self.model_name, "min_words": self.min_words}}
