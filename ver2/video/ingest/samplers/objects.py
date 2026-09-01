"""Sampling on which of a named vocabulary of objects is in shot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from .detection import DetectionChangeSampler

if TYPE_CHECKING:
    from .components.descriptors import RegionDescriptor
    from .components.detectors import ObjectDetector


class ObjectChangeSampler(DetectionChangeSampler):
    """Objects, compared by presence and position -- no embedder involved.

    Objects have no state the way people do. Measured on real footage, the
    same object one second later scores 0.989 CLIP similarity and has moved
    half a pixel, so an appearance embedding carries almost no signal while
    costing a forward pass per detection. What changes is whether a thing is
    there and where.

    Detection uses an open vocabulary because COCO's classes do not contain
    the objects most footage is actually about -- plain YOLO found seven
    non-person classes on checkout video, dominated by ``bench`` and
    ``suitcase`` at ~0.30 confidence, both of which were the same empty
    counter. Matching is class-aware, so a cart never counts as a bag.
    """

    name = "objects"

    def __init__(
        self,
        detector: Optional["ObjectDetector"] = None,
        vocabulary: Optional[Sequence[str]] = None,
        threshold: float = 0.30,
        class_aware: bool = True,
        confidence: float = 0.30,
        metric: str = "proximity",
        min_interval_s: float = 0.0,
        max_per_chunk: Optional[int] = None,
        sampler_id: Optional[str] = None,
    ) -> None:
        from .components.descriptors import BoxGeometryDescriptor

        if detector is None:
            from .components.detectors import OpenVocabDetector

            detector = OpenVocabDetector(vocabulary=vocabulary, confidence=confidence)
        super().__init__(
            detector=detector,
            descriptor=BoxGeometryDescriptor(class_aware=class_aware, metric=metric),
            threshold=threshold,
            min_interval_s=min_interval_s,
            max_per_chunk=max_per_chunk,
            sampler_id=sampler_id,
        )


