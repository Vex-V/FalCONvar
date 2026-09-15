"""Perception: turning pixels into something comparable.

The pieces a sampler composes, and the line between this package and its
parent is the one that matters most here. **A sampler is policy** -- given a
comparison, keep this frame or drop it, subject to rate limits and chunk
boundaries. **A component is perception** -- find the regions worth looking at,
or turn a region or a frame into a vector that can be compared at all.

The dependency runs one way and always has: nothing in here imports a sampler.
That is what makes them interchangeable, and it is why the same
DetectionChangeSampler serves people, objects and text by being handed a
different detector and descriptor.

It is also where every model weight lives. Nothing above this package loads
YOLO, CLIP or EasyOCR; everything in it does, which is why the imports are
lazy -- the uniform sampler has to stay usable on a machine with none of them
installed.
"""

from .descriptors import (BoxGeometryDescriptor, CropEmbeddingDescriptor,
                          RegionDescriptor, TextLayoutDescriptor)
from .detectors import (Detection, ObjectDetector, OpenVocabDetector,
                        TextRegionDetector, YoloPersonDetector)
from .embedders import CLIPEmbedder, FrameEmbedder

__all__ = [
    "BoxGeometryDescriptor",
    "CLIPEmbedder",
    "CropEmbeddingDescriptor",
    "Detection",
    "FrameEmbedder",
    "ObjectDetector",
    "OpenVocabDetector",
    "RegionDescriptor",
    "TextLayoutDescriptor",
    "TextRegionDetector",
    "YoloPersonDetector",
]
