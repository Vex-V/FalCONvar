"""Records passed between ingestion stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import numpy as np


@dataclass
class Frame:
    """One frame moving through the pipeline.

    ``media_ts`` is the position on the media clock in seconds and is the only
    clock a downstream *decision* may use. ``pts`` is the same position in the
    container's own integer timebase, and is the only thing that can *address*
    the frame later: seconds are a lossy rendering of it, and at timebases as
    fine as 1/1200000 a rounded float can land on the wrong frame.

    ``index`` is a plain read counter. It cannot be corrupted by bad container
    metadata, which reconstructing the index from a timestamp can, but it is
    only meaningful for a source read from the beginning.

    ``gap_before`` and ``discontinuity`` are always 0/False for a file. They
    are where a live source will differ, so they exist now rather than being
    retrofitted through every stage later.

    ``image`` is BGR (OpenCV's order, not RGB) and is *borrowed*: it is
    released as soon as the frame is known not to be needed, so anything that
    outlives that must copy. It is the only field that ever changes.
    """

    index: int
    media_ts: float
    pts: Optional[int] = None
    image: Optional[np.ndarray] = field(default=None, repr=False)
    is_keyframe: bool = False
    gap_before: int = 0
    discontinuity: bool = False

    @property
    def has_image(self) -> bool:
        return self.image is not None

    @property
    def nbytes(self) -> int:
        return self.image.nbytes if self.image is not None else 0

    def release(self) -> None:
        """Drop the pixels. A method, so the one dangerous operation stays greppable."""
        self.image = None


@dataclass
class SourceInfo:
    """What was established about a source before any frame was processed."""

    uri: str
    fps: float                      # guessed_rate: the container's best estimate
    fps_trusted: bool
    time_base: Optional[Fraction]   # pts * time_base == seconds, exactly
    width: int
    height: int
    frame_count: Optional[int]
    timeline: str                   # "pts" | "derived"
    rotation: float                 # degrees to apply on the way out
    notes: list[str] = field(default_factory=list)

    @property
    def has_pts(self) -> bool:
        return self.timeline == "pts"

    def as_dict(self) -> dict:
        return {
            "uri": self.uri,
            "fps": self.fps,
            "fps_trusted": self.fps_trusted,
            # Serialised as "1/15360" so the exact rational survives JSON,
            # which has no rational type and would round it to a float.
            "time_base": str(self.time_base) if self.time_base else None,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "timeline": self.timeline,
            "rotation": self.rotation,
            "notes": self.notes,
        }
