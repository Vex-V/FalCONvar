"""Detect, describe, compare -- the shape every detection sampler shares.

What varies is the descriptor, because what carries the signal varies by
subject: appearance for people, position for objects, layout for text. See
:mod:`descriptors`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from ..source import Frame
from .base import Sampler

if TYPE_CHECKING:
    from .descriptors import RegionDescriptor
    from .detectors import ObjectDetector
    from .embedders import FrameEmbedder


class DetectionChangeSampler(Sampler):
    """Samples when detected regions stop looking like the last kept frame.

    The score is the *weakest best match*, taken in one direction only: for
    each region visible now, how well does it match anything in the reference
    frame; the worst of those is the score. Direction matters more than it
    looks. A detector that momentarily loses something and finds it again
    produces a region that matches its own earlier self, so nothing fires.
    Scoring the reverse direction too would treat every dropout as a
    departure, which on busy footage is most frames. Consequence: regions
    leaving does not trigger a sample; regions arriving or changing does.

    Region *count* is deliberately never a trigger. On checkout footage the
    person count changed on roughly half of all frame pairs at every
    confidence threshold tried, and raising confidence to suppress it simply
    lost real people. Counting is not a change signal on busy footage.
    """

    name = "detection"

    def __init__(
        self,
        detector: Optional["ObjectDetector"] = None,
        descriptor: Optional["RegionDescriptor"] = None,
        threshold: float = 0.83,
        min_interval_s: float = 0.0,
        max_per_chunk: Optional[int] = None,
        sampler_id: Optional[str] = None,
    ) -> None:
        super().__init__(min_interval_s, max_per_chunk, sampler_id)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be a similarity in [0, 1]")
        if detector is None or descriptor is None:
            raise ValueError("detector and descriptor are required")
        self.detector = detector
        self.descriptor = descriptor
        self.threshold = threshold
        self._reference: Optional[object] = None
        self._last_score: Optional[float] = None

    def on_reset(self, chunk_id: int) -> None:
        self._reference = None
        self._last_score = None

    def last_score(self) -> Optional[float]:
        return self._last_score

    def describe(self, frame: Frame):
        if frame.image is None:
            raise ValueError(f"{type(self).__name__} needs pixels; frame.image is None")
        detections = self.detector.detect(frame.image)
        return self.descriptor.describe(frame.image, detections)

    def compare(self, current, reference) -> Optional[float]:
        """The weakest best match, one-directional. See the class docstring."""
        if self.descriptor.count(current) == 0:
            return 1.0                      # nothing in shot; reference untouched
        if self.descriptor.count(reference) == 0:
            return 0.0                      # something appeared where there was nothing
        similarity = self.descriptor.similarity(current, reference)
        return float(similarity.max(axis=1).min())

    def propose(self, frame: Frame, chunk_local_index: int) -> bool:
        current = self.describe(frame)
        count = self.descriptor.count(current)

        if self._reference is None:
            self._last_score = None
            self._reference = current
            return True

        if count == 0:
            # Nothing in shot. The reference is left alone, so whatever
            # returns is still compared against what was here before.
            self._last_score = 1.0
            return False

        if self.descriptor.count(self._reference) == 0:
            # Something has appeared where there was nothing.
            self._last_score = 0.0
            self._reference = current
            return True

        similarity = self.descriptor.similarity(current, self._reference)
        score = float(similarity.max(axis=1).min())
        self._last_score = score

        keep = score < self.threshold
        if keep:
            # Only a kept frame moves the reference, so change accumulates
            # across skipped frames instead of resetting each time.
            self._reference = current
        return keep

    def config(self) -> dict:
        return {
            **self._base_config(),
            "threshold": self.threshold,
            "detector": self.detector.config(),
            "descriptor": self.descriptor.config(),
        }


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
    ) -> None:
        from .descriptors import CropEmbeddingDescriptor

        if detector is None:
            from .detectors import YoloPersonDetector

            detector = YoloPersonDetector()
        super().__init__(
            detector=detector,
            descriptor=CropEmbeddingDescriptor(embedder=embedder, crop_pad=crop_pad),
            threshold=threshold,
            min_interval_s=min_interval_s,
            max_per_chunk=max_per_chunk,
            sampler_id=sampler_id,
        )


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
        from .descriptors import BoxGeometryDescriptor

        if detector is None:
            from .detectors import OpenVocabDetector

            detector = OpenVocabDetector(vocabulary=vocabulary, confidence=confidence)
        super().__init__(
            detector=detector,
            descriptor=BoxGeometryDescriptor(class_aware=class_aware, metric=metric),
            threshold=threshold,
            min_interval_s=min_interval_s,
            max_per_chunk=max_per_chunk,
            sampler_id=sampler_id,
        )


class TextChangeSampler(DetectionChangeSampler):
    """Text, compared by where it is *and* what it looks like.

    EasyOCR supplies the regions; nothing is read. The VLM downstream reads
    text better than an OCR engine would, so the only question here is whether
    the text on screen changed enough to be worth sending.

    Geometry alone cannot answer that -- a slide advances and the text block
    does not move an inch. Neither can per-region comparison: EasyOCR merges
    and splits lines frame to frame, returning two, three and four regions in
    successive seconds of a completely static slide, and a split line cannot
    match the merged version it is compared against.

    ``TextLayoutDescriptor`` masks the frame to wherever text was found and
    describes the result *once*. Split or merged, the ink covers the same
    pixels. Position is captured through the mask, content through the pixels.

    This sampler is for screens and slides. On footage where text is
    incidental it keeps ~30% of frames, which is not a bug -- people occlude
    the text constantly and the whole-frame mask shifts as they pass.
    """

    name = "text"

    def __init__(
        self,
        detector: Optional["ObjectDetector"] = None,
        threshold: float = 0.92,
        grid: int = 128,
        languages: Sequence[str] = ("en",),
        min_interval_s: float = 0.0,
        max_per_chunk: Optional[int] = None,
        sampler_id: Optional[str] = None,
    ) -> None:
        from .descriptors import TextLayoutDescriptor

        if detector is None:
            from .detectors import TextRegionDetector

            detector = TextRegionDetector(languages=languages)
        super().__init__(
            detector=detector,
            descriptor=TextLayoutDescriptor(grid=grid),
            threshold=threshold,
            min_interval_s=min_interval_s,
            max_per_chunk=max_per_chunk,
            sampler_id=sampler_id,
        )
