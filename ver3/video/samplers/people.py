"""Sampling on who is in shot: how many, where, and how that changes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from .detection import DetectionChangeSampler

if TYPE_CHECKING:
    from .components.descriptors import RegionDescriptor
    from .components.detectors import ObjectDetector


class PersonChangeSampler(DetectionChangeSampler):
    """People, compared by appearance.

    YOLO locates each person, CLIP embeds each person *crop*, and the crop
    embeddings are compared. Cropping before embedding is the point: a full
    frame is resized to 224x224 and centre-cropped before CLIP sees it, so a
    person occupies roughly one patch and their detail is gone. A crop is
    *upsampled* to 224 instead, so posture, orientation and held objects
    survive into the vector -- which is how a person standing perfectly still
    who starts reading their phone registers as a change.

    No identity tracking. At 1 fps a walking person moves several times their
    own box width between frames, so IoU-based trackers cannot associate them.
    Comparing sets sidesteps identity and still catches what box comparison
    cannot: one person leaving as another arrives elsewhere.

    The score is a minimum over people, so it falls as headcount rises purely
    as an order statistic -- median 0.949 for one to three people against
    0.856 for eight or more. Crowded frames therefore sample more;
    ``min_interval_s`` is the practical control. Threshold 0.83 was calibrated
    against a person-crop median of 0.907.
    """

    name = "yolo"

    def __init__(
        self,
        detector: Optional["ObjectDetector"] = None,
        embedder: Optional["FrameEmbedder"] = None,
        threshold: float = 0.83,
        crop_pad: float = 0.08,
        min_interval_s: float = 0.0,
        max_per_chunk: Optional[int] = None,
        sampler_id: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> None:
        from .components.descriptors import CropEmbeddingDescriptor

        if detector is None:
            from .components.detectors import YoloPersonDetector

            detector = YoloPersonDetector()
        super().__init__(
            detector=detector,
            descriptor=CropEmbeddingDescriptor(embedder=embedder, crop_pad=crop_pad),
            threshold=threshold,
            min_interval_s=min_interval_s,
            max_per_chunk=max_per_chunk,
            sampler_id=sampler_id,
            prompt=prompt,
        )


