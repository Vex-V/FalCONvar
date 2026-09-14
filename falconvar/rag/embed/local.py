"""Vectors from a Hugging Face model loaded into this process.

No key, no server, and nothing leaves the machine after the download. Weights
land in `weights/embedders/`, beside the detector checkpoints, rather than
wherever a library decides.

`sentence-transformers` is used when it is installed, because it reads a
model's whole module list. Without it `transformers` does the pooling the
model declares in `1_Pooling/config.json` -- CLS, mean or last token -- and
refuses a model whose pipeline has a `Dense` layer after pooling. Skipping that
layer would still produce vectors of a plausible width, and then install
`sentence-transformers` later and the same key would hold vectors from two
different functions.

Loaded once per (model, device) per process: `/search` is answered in the
server process, and reloading a model per query costs seconds.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

from ...shared import env, paths
from .embedders import EmbedderUnavailable, key_for, prefixes_for

CACHE = paths.WEIGHTS / "embedders"

_ENGINES: dict[tuple[str, str], "_Engine"] = {}
_LOCK = threading.Lock()


def _repo_json(model: str, filename: str) -> Optional[Any]:
    """A JSON file from the model's repo or folder, or None if it has none."""
    folder = Path(model)
    if folder.is_dir():
        path = folder / filename
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model, filename, cache_dir=str(CACHE))
    except Exception:                                      # noqa: BLE001
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


class _Engine:
    def __init__(self, model: str, device: str) -> None:
        self.model, self.device = model, device
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise EmbedderUnavailable("the local embedder needs torch") from exc
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            SentenceTransformer = None
        try:
            if SentenceTransformer is not None:
                self._st = SentenceTransformer(model, device=device, cache_folder=str(CACHE))
                self.dims = int(self._st.get_sentence_embedding_dimension())
                self.backend, self.pooling = "sentence-transformers", "declared"
                return
            self._st = None
            self._load_transformers()
        except EmbedderUnavailable:
            raise
        except Exception as exc:                           # noqa: BLE001
            raise EmbedderUnavailable(f"could not load {model!r}: {exc}") from None

    def _load_transformers(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        modules = _repo_json(self.model, "modules.json") or []
        kinds = [m.get("type", "") for m in modules]
        if any("Dense" in k for k in kinds):
            raise EmbedderUnavailable(
                f"{self.model} has a Dense layer after pooling, which only "
                "sentence-transformers applies: pip install sentence-transformers")
        folder = next((m.get("path") for m in modules if "Pooling" in m.get("type", "")),
                      "1_Pooling")
        pooling = _repo_json(self.model, f"{folder}/config.json") or {}
        self.pooling = ("cls" if pooling.get("pooling_mode_cls_token")
                        else "last" if pooling.get("pooling_mode_lasttoken") else "mean")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, cache_dir=str(CACHE))
        self.net = AutoModel.from_pretrained(self.model, cache_dir=str(CACHE)
                                             ).to(self.device).eval()
        self.dims = int(self.net.config.hidden_size)
        self.backend = "transformers"

    def encode(self, texts: Sequence[str], batch: int, max_length: int) -> list[list[float]]:
        if self._st is not None:
            return self._st.encode(list(texts), batch_size=batch,
                                   normalize_embeddings=True,
                                   convert_to_numpy=True).tolist()
        import torch

        limit = min(max_length, int(getattr(self.tokenizer, "model_max_length", max_length)))
        out: list[list[float]] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch):
                encoded = self.tokenizer(list(texts[start:start + batch]), padding=True,
                                         truncation=True, max_length=limit,
                                         return_tensors="pt").to(self.device)
                hidden = self.net(**encoded).last_hidden_state
                mask = encoded["attention_mask"]
                if self.pooling == "cls":
                    pooled = hidden[:, 0]
                elif self.pooling == "last":
                    if getattr(self.tokenizer, "padding_side", "right") == "left":
                        pooled = hidden[:, -1]
                    else:
                        last = mask.sum(dim=1) - 1
                        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
                else:
                    weights = mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
                out.extend(pooled.cpu().tolist())
        return out


def _engine(model: str, device: str) -> _Engine:
    with _LOCK:
        if (model, device) not in _ENGINES:
            _ENGINES[(model, device)] = _Engine(model, device)
        return _ENGINES[(model, device)]


class LocalEmbedder:
    name = "local"

    def __init__(self, model: Optional[str] = None, dims: Optional[int] = None,
                 device: Optional[str] = None, batch: int = 32,
                 max_length: int = 512,
                 query_prefix: Optional[str] = None,
                 document_prefix: Optional[str] = None) -> None:
        env.load()                       # HF_TOKEN, for a gated model
        from ...shared import providers
        self.model = model or providers.get("local").embed_model or ""
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device, self.batch, self.max_length = device, batch, max_length
        auto_query, auto_document = prefixes_for(self.model)
        self.query_prefix = auto_query if query_prefix is None else query_prefix
        self.document_prefix = auto_document if document_prefix is None else document_prefix
        self._engine = _engine(self.model, device)
        self.dims = self._engine.dims
        if dims and int(dims) != self.dims:
            raise EmbedderUnavailable(
                f"{self.model} makes {self.dims}-dimension vectors, not {dims}")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._engine.encode([self.document_prefix + t for t in texts],
                                   self.batch, self.max_length)

    def embed_query(self, text: str) -> list[float]:
        return self._engine.encode([self.query_prefix + text], 1, self.max_length)[0]

    @property
    def key(self) -> str:
        return key_for(self.name, self.model, self.dims,
                       self.query_prefix, self.document_prefix)

    def config(self) -> dict[str, Any]:
        return {"embedder": self.name, "model": self.model, "dims": self.dims,
                "backend": self._engine.backend, "pooling": self._engine.pooling,
                "device": self.device, "max_length": self.max_length,
                "query_prefix": self.query_prefix,
                "document_prefix": self.document_prefix}


__all__ = ["CACHE", "LocalEmbedder"]
