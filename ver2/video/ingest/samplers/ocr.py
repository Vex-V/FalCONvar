"""Sampling on the writing in shot changing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from .detection import DetectionChangeSampler

if TYPE_CHECKING:
    from .components.descriptors import RegionDescriptor
    from .components.detectors import ObjectDetector


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
        from .components.descriptors import TextLayoutDescriptor

        if detector is None:
            from .components.detectors import TextRegionDetector

            detector = TextRegionDetector(languages=languages)
        super().__init__(
            detector=detector,
            descriptor=TextLayoutDescriptor(grid=grid),
            threshold=threshold,
            min_interval_s=min_interval_s,
            max_per_chunk=max_per_chunk,
            sampler_id=sampler_id,
        )
