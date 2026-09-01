"""Frame embedders, kept separate from the samplers that use them.

A sampler decides *whether* a frame is worth keeping; an embedder decides
*what a frame looks like* as a vector. Splitting them means the sampling
policy can be tested without loading model weights, and the model can be
swapped without touching the policy.

All embedders return L2-normalised vectors, so cosine similarity is a dot
product.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np


class FrameEmbedder(ABC):
    """Turns frames into unit-length vectors."""

    dim: int = 0
    name: str = "base"

    @abstractmethod
    def embed(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Embed a batch. Returns (N, dim), L2-normalised."""

    def embed_one(self, image: np.ndarray) -> np.ndarray:
        return self.embed([image])[0]

    def config(self) -> dict:
        return {"name": self.name, "dim": self.dim}


def _l2(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


class CLIPEmbedder(FrameEmbedder):
    """OpenAI CLIP image tower.

    Frames arrive from OpenCV as BGR; CLIP expects RGB, and getting that
    backwards quietly degrades every similarity in the pipeline rather than
    failing, so the conversion happens here and only here.

    Frames are downscaled with cv2 before the processor sees them. The
    processor would resize anyway, but doing it on a 1920x1080 PIL image is
    most of the per-frame cost.
    """

    name = "clip"

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        pre_resize: int = 256,
        batch_size: int = 16,
    ) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pre_resize = pre_resize
        self.batch_size = batch_size

        self._model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = int(self._model.config.projection_dim)

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        import cv2

        h, w = image.shape[:2]
        if self.pre_resize and min(h, w) > self.pre_resize:
            scale = self.pre_resize / min(h, w)
            image = cv2.resize(
                image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
            )
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def embed(self, images: Sequence[np.ndarray]) -> np.ndarray:
        torch = self._torch
        out: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = [self._prepare(im) for im in images[start : start + self.batch_size]]
            with torch.no_grad():
                pixels = self._processor(images=batch, return_tensors="pt")["pixel_values"]
                features = self._model.get_image_features(pixel_values=pixels.to(self.device))
            # transformers >=5 returns a model output here rather than a tensor.
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            out.append(features.float().cpu().numpy())
        return _l2(np.concatenate(out, axis=0))

    def config(self) -> dict:
        return {
            "name": self.name,
            "model": self.model_name,
            "dim": self.dim,
            "device": self.device,
        }


