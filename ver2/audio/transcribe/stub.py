"""A transcriber that needs no model.

Deterministic, so two runs over the same track are byte-identical -- which is
what makes the segmentation, the sinks and the resume path testable without a
GPU or a download. It fabricates one word per second of speech so the
downstream code sees the shape it has to handle, never plausible text.
"""

from __future__ import annotations

from typing import Any

from ..source import Track
from .base import Segment, Transcript, Word


class StubTranscriber:
    """Fixed text on a fixed grid. A ``Transcriber``."""

    name = "stub"

    def __init__(self, segment_s: float = 5.0) -> None:
        self.segment_s = segment_s

    def transcribe(self, track: Track) -> Transcript:
        segments: list[Segment] = []
        if not track.silent:
            t = 0.0
            index = 0
            while t < track.duration_s:
                end = min(t + self.segment_s, track.duration_s)
                words = [Word(start=min(t + i, end), end=min(t + i + 1, end),
                              text=f"[stub{index}.{i}]", probability=1.0)
                         for i in range(int(end - t))]
                segments.append(Segment(start=t, end=end,
                                        text=" ".join(w.text for w in words),
                                        words=words))
                t, index = end, index + 1
        return Transcript(language="zxx", language_probability=1.0,
                          duration_s=track.duration_s, segments=segments,
                          model=self.config())

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "params": {"segment_s": self.segment_s}}
