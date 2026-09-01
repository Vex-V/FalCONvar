"""Embedding on the machine that already has the GPU.

Built on ``transformers`` directly rather than ``sentence-transformers``,
because the whole of what that library adds here is mean pooling and an L2
normalise -- about fifteen lines -- and this project has been bitten before by
an install that silently downgraded numpy and swapped the active cv2. A
dependency that buys fifteen lines is not worth that risk.

**Context length is a property to check, not to assume.** Summaries currently
run 236-441 tokens, so nothing is truncated even at 512 -- but that is one
video, and appending rendered structured fields to the embedded text would push
the median to ~870 and put 8 of 10 over the line. A 512-token model does not
fail on those, it truncates them, and it truncates the end: in a `clip`
description that is the "what changed between frames" part, in a `yolo`
description the later movement. The index would then quietly answer questions
about the first half of every window.

So the models below are all long-context, and `_check_lengths` refuses rather
than truncating. The popular 512-token families (bge-*-v1.5, e5-*-v2, mxbai)
are excluded on that basis alone.

Prefixes matter as much as the weights. Asymmetric models were trained with
one string in front of passages and another in front of questions, and getting
that wrong costs accuracy without changing anything observable.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


class TextTooLong(Exception):
    """A document would be silently truncated by this model's context window."""

#: Long-context models that fit comfortably in 8 GiB and suit English prose.
#: `document`/`query` are the prefixes the model was trained with; empty means
#: the model is symmetric and wants none.
KNOWN: dict[str, dict[str, Any]] = {
    "nomic-ai/nomic-embed-text-v1.5": {
        "dimensions": 768, "max_tokens": 8192,
        "document": "search_document: ", "query": "search_query: ",
        "note": "137M, tiny and fast; Matryoshka, so dims can be truncated",
    },
    "BAAI/bge-m3": {
        "dimensions": 1024, "max_tokens": 8192,
        "document": "", "query": "",
        "note": "568M, multilingual; also emits sparse weights for hybrid",
    },
    "Alibaba-NLP/gte-large-en-v1.5": {
        "dimensions": 1024, "max_tokens": 8192,
        "document": "", "query": "",
        "note": "434M, English, strong on long passages",
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "dimensions": 1024, "max_tokens": 32768,
        "document": "", "query": "",
        "note": "600M, instruction-aware, very long context",
    },
}


class LocalEmbedder:
    """A transformer run locally, mean-pooled and normalised. An ``Embedder``."""

    name = "local"

    def __init__(
        self,
        model: str = "nomic-ai/nomic-embed-text-v1.5",
        device: Optional[str] = None,
        batch_size: int = 8,
        max_tokens: Optional[int] = None,
        allow_truncation: bool = False,
        dimensions: Optional[int] = None,
        document_prefix: Optional[str] = None,
        query_prefix: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        known = KNOWN.get(model, {})
        self.model_name = model
        self.batch_size = batch_size
        self.allow_truncation = allow_truncation
        self.document_prefix = (document_prefix if document_prefix is not None
                                else known.get("document", ""))
        self.query_prefix = (query_prefix if query_prefix is not None
                             else known.get("query", ""))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        # Ask the tokenizer rather than assuming 512. A hardcoded default is
        # how an unregistered model ends up quietly truncating: the vector
        # still has the right shape, the search still returns results, and the
        # tail of every description -- which is where a scene sampler puts what
        # changed -- is simply not in the index. Some tokenizers report a
        # sentinel instead of a real limit, so that is treated as unknown.
        reported = getattr(self._tokenizer, "model_max_length", None)
        if not isinstance(reported, int) or reported > 1_000_000:
            reported = None
        self.max_tokens = max_tokens or known.get("max_tokens") or reported or 512
        self._model = AutoModel.from_pretrained(model, trust_remote_code=True)
        self._model.to(self.device).eval()
        # The model is the authority on its own width; KNOWN is a hint that can
        # go stale, and a wrong dimension is not discovered until pgvector
        # rejects an insert halfway through a run.
        self.dimensions = dimensions or int(self._model.config.hidden_size)

    def _check_lengths(self, texts: Sequence[str], prefix: str) -> None:
        """Refuse rather than truncate, unless explicitly told otherwise."""
        if self.allow_truncation:
            return
        counts = [len(self._tokenizer(prefix + t, add_special_tokens=True)["input_ids"])
                  for t in texts]
        over = [(i, n) for i, n in enumerate(counts) if n > self.max_tokens]
        if over:
            worst = max(n for _, n in over)
            raise TextTooLong(
                f"{len(over)} of {len(texts)} texts exceed {self.model_name}'s "
                f"{self.max_tokens}-token window (longest {worst}). They would be "
                "truncated from the end, which is where a description says what "
                "changed. "
                "Use a long-context model -- see KNOWN in this module, all 8192+ "
                "-- or pass allow_truncation=True if losing the tail is acceptable."
            )

    def _encode(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        torch = self._torch
        self._check_lengths(texts, prefix)
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [prefix + t for t in texts[start:start + self.batch_size]]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_tokens, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self._model(**encoded).last_hidden_state
            # Mean over real tokens only: padding would otherwise drag every
            # short passage's vector toward the same place.
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.extend(pooled.cpu().to(torch.float32).numpy().astype(np.float32).tolist())
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(list(texts), self.document_prefix)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self.query_prefix)[0]

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model_name,
            "dimensions": self.dimensions,
            "params": {
                "max_tokens": self.max_tokens,
                "allow_truncation": self.allow_truncation,
                "device": self.device,
                "document_prefix": self.document_prefix,
                "query_prefix": self.query_prefix,
            },
        }
