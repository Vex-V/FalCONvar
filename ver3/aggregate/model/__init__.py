"""The local tier: GPU models over text already produced.

Between `statistics` (arithmetic) and `llm` (paid calls). These load a model
and cost electricity, but no money and no network once the weights are cached
-- which is the distinction the tier ladder draws.

Both read the *rendered* chunk text rather than the raw documents, so they see
what a reader would: the same lines the llm tier is given.

Nothing here is imported until an aggregator is asked for by name. A `--tier
free` run must not pull in torch.
"""

from __future__ import annotations

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


__all__ = ["DEFAULT_LABELS", "DEFAULT_NER_MODEL", "DEFAULT_SENTIMENT_MODEL",
           "MAX_CHARS", "ModelUnavailable"]
