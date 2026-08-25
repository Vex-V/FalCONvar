"""Object detectors, kept separate from the samplers that use them.

Same split as the embedders: a detector answers "where are the people in this
frame", a sampler decides what to do about it. Keeping them apart means the
sampling policy is testable against fixed boxes, without model weights.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected object in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str = "person"

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def crop(self, image: np.ndarray, pad: float = 0.08) -> Optional[np.ndarray]:
        """Cut this detection out of the frame, with a little context around it.

        Padding matters: a box clipped tight to a body loses whatever the
        person is interacting with, which is usually the thing worth
        describing.
        """
        h, w = image.shape[:2]
        px, py = self.width * pad, self.height * pad
        x1 = int(max(0, round(self.x1 - px)))
        y1 = int(max(0, round(self.y1 - py)))
        x2 = int(min(w, round(self.x2 + px)))
        y2 = int(min(h, round(self.y2 + py)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return image[y1:y2, x1:x2]


class ObjectDetector(ABC):
    """Finds regions of interest in a frame."""

    name: str = "base"

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]: ...

    def config(self) -> dict:
        return {"name": self.name}


class YoloPersonDetector(ObjectDetector):
    """Ultralytics YOLO restricted to the person class.

    ``min_height`` drops detections too small to crop usefully. A person
    forty pixels tall carries no describable detail once cropped and
    resized, and including them adds noise to the change signal without
    adding information.
    """

    name = "yolo"

    def __init__(
        self,
        model: str = "yolo11n.pt",
        confidence: float = 0.35,
        device: Optional[str] = None,
        min_height: int = 64,
        imgsz: int = 640,
    ) -> None:
        os.environ.setdefault("YOLO_VERBOSE", "False")
        from ultralytics import YOLO

        import torch

        self.model_name = model
        self.confidence = confidence
        self.min_height = min_height
        self.imgsz = imgsz
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = YOLO(model)
        self._model.to(self.device)

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self._model.predict(
            image,
            classes=[0],  # COCO class 0 is person
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        out = []
        for (x1, y1, x2, y2), c in zip(xyxy, conf):
            det = Detection(float(x1), float(y1), float(x2), float(y2), float(c))
            if det.height >= self.min_height:
                out.append(det)
        return out

    def config(self) -> dict:
        return {
            "name": self.name,
            "model": self.model_name,
            "confidence": self.confidence,
            "min_height": self.min_height,
            "imgsz": self.imgsz,
            "device": self.device,
        }


class OpenVocabDetector(ObjectDetector):
    """YOLO-World: classes given as text rather than fixed at training time.

    COCO's eighty classes are the wrong vocabulary for most real footage.
    Measured on supermarket checkout video, plain YOLO found seven non-person
    classes, dominated by ``bench`` and ``suitcase`` at ~0.30 confidence —
    both of which were the same static empty counter. Shopping bags, carts,
    groceries and boxes have no COCO class at all. Naming the classes
    directly took that from 1.8 junk detections per frame to 12.9 real ones,
    at the same cost, because ``set_classes`` embeds the vocabulary once.

    Confidence runs much lower than a closed-vocabulary detector — 0.15 to
    0.40 is normal for a correct box — so thresholds tuned against COCO
    numbers do not transfer.
    """

    name = "openvocab"

    DEFAULT_VOCAB = (
        "shopping bag",
        "plastic bag",
        "shopping cart",
        "shopping basket",
        "cardboard box",
        "bottle",
        "grocery item",
        "cash register",
    )

    def __init__(
        self,
        vocabulary: Optional[Sequence[str]] = None,
        model: str = "yolov8s-worldv2.pt",
        confidence: float = 0.20,
        device: Optional[str] = None,
        min_height: int = 0,
        imgsz: int = 640,
    ) -> None:
        os.environ.setdefault("YOLO_VERBOSE", "False")
        import torch
        from ultralytics import YOLOWorld

        self.vocabulary = list(vocabulary or self.DEFAULT_VOCAB)
        self.model_name = model
        self.confidence = confidence
        self.min_height = min_height
        self.imgsz = imgsz
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = YOLOWorld(model)
        self._model.set_classes(self.vocabulary)
        self._model.to(self.device)

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self._model.predict(
            image,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        out = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            label = self.vocabulary[k] if k < len(self.vocabulary) else str(k)
            det = Detection(float(x1), float(y1), float(x2), float(y2), float(c), label)
            if det.height >= self.min_height:
                out.append(det)
        return out

    def config(self) -> dict:
        return {
            "name": self.name,
            "model": self.model_name,
            "vocabulary": self.vocabulary,
            "confidence": self.confidence,
            "min_height": self.min_height,
            "device": self.device,
        }


class TextRegionDetector(ObjectDetector):
    """EasyOCR's detector only — where is there text, not what does it say.

    Recognition is skipped deliberately. The VLM downstream reads text far
    better than an OCR engine will, so all that is wanted here is the
    regions: enough to notice that the text on screen has changed and a
    frame is worth sending.

    Every region found is weighted equally. Suppressing an area — a burnt-in
    camera clock is the usual reason — is deliberately not handled here: that
    is one setting of a general per-region weighting the caller should own,
    not a binary exclusion baked into the detector.
    """

    name = "text"

    def __init__(
        self,
        languages: Sequence[str] = ("en",),
        gpu: Optional[bool] = None,
        min_height: int = 0,
        low_text: float = 0.4,
        # 0.85 rather than EasyOCR's 0.7 default. On animated infographics the
        # looser setting finds text-like structure in graphics — pipework, gauge
        # faces — and those spurious regions drift frame to frame, which reads
        # as the text having changed. Measured on an animated documentary, one
        # chunk went from 24 samples to 2 with no loss on clean rendered text.
        text_threshold: float = 0.85,
        link_threshold: float = 0.4,
        canvas_size: int = 1280,
        mag_ratio: float = 1.0,
    ) -> None:
        import easyocr
        import torch

        self.languages = list(languages)
        self.gpu = torch.cuda.is_available() if gpu is None else gpu
        self.min_height = min_height
        self.low_text = low_text
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)

    def detect(self, image: np.ndarray) -> list[Detection]:
        horizontal, free = self._reader.detect(
            image,
            low_text=self.low_text,
            text_threshold=self.text_threshold,
            link_threshold=self.link_threshold,
            canvas_size=self.canvas_size,
            mag_ratio=self.mag_ratio,
        )
        out: list[Detection] = []

        def keep(det: Detection) -> None:
            if det.height >= self.min_height and det.area > 0:
                out.append(det)

        # detect() returns a list-per-image; unwrap it.
        for box in (horizontal[0] if horizontal else []):
            x_min, x_max, y_min, y_max = box
            keep(Detection(float(x_min), float(y_min), float(x_max), float(y_max), 1.0, "text"))
        for quad in (free[0] if free else []):
            pts = np.array(quad, dtype=np.float64)
            keep(Detection(
                float(pts[:, 0].min()), float(pts[:, 1].min()),
                float(pts[:, 0].max()), float(pts[:, 1].max()), 1.0, "text",
            ))
        return out

    def config(self) -> dict:
        return {
            "name": self.name,
            "languages": self.languages,
            "gpu": self.gpu,
            "min_height": self.min_height,
            "low_text": self.low_text,
            "text_threshold": self.text_threshold,
        }


class FixedDetector(ObjectDetector):
    """Returns pre-set detections. For testing sampling policy without a model."""

    name = "fixed"

    def __init__(self, per_call: Sequence[Sequence[Detection]]) -> None:
        self.per_call = list(per_call)
        self._i = 0

    def detect(self, image: np.ndarray) -> list[Detection]:
        if self._i >= len(self.per_call):
            return []
        out = list(self.per_call[self._i])
        self._i += 1
        return out
