"""The local tier: GPU models over text already produced.

Between `statistics` (arithmetic) and `llm` (paid calls). These load a model
and cost electricity, but no money and no network once the weights are cached
-- which is the distinction the tier ladder draws.

Both read an input, as the llm tier does, and answer once per input.

**Long text is cut into pieces, never truncated.** Each model has a length it
reads, and past it the rest is silently ignored: a sentiment model scoring a
whole 20-second chunk reported the tone of its first clause and called it the
chunk's. So a chunk's text is cut at sentence ends into pieces that fit, and
the answer is taken over the pieces.

Nothing here is imported until an aggregator is asked for by name. A `--tier
free` run must not pull in torch.
"""

from __future__ import annotations

import re

#: What to look for. GLiNER is zero-shot, so the label set *is* the
#: configuration -- the same lesson as the open-vocabulary detector, where a
#: mismatched vocabulary found 2.4 detections per frame and a matched one 5.1.
DEFAULT_LABELS = ("person", "organisation", "location", "product",
                  "event", "date")

DEFAULT_NER_MODEL = "urchade/gliner_small-v2.1"
DEFAULT_SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

#: The longest piece a sentiment model is handed. Trained on single sentences.
MAX_CHARS = 480

#: The longest piece GLiNER is handed; it reads 384 tokens.
NER_CHARS = 1200


class ModelUnavailable(RuntimeError):
    """No package, no weights, or a load that retrying will not fix."""


def pieces(text: str, limit: int) -> list[str]:
    """Text cut at sentence ends -- then at spaces -- into pieces of at most
    `limit` characters. Nothing is dropped."""
    out: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if current:
                out.append(current)
                current = ""
            out.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= limit:
            current = f"{current} {sentence}"
        else:
            out.append(current)
            current = sentence
    if current:
        out.append(current)
    return [p for p in out if p]


def plain(row) -> str:
    """A row's text without labels or times: what a model should read."""
    return " ".join(said for _, said in row.parts)


__all__ = ["DEFAULT_LABELS", "DEFAULT_NER_MODEL", "DEFAULT_SENTIMENT_MODEL",
           "MAX_CHARS", "NER_CHARS", "ModelUnavailable", "pieces", "plain"]
