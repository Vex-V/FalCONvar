"""What a transcriber is, and what it returns.

The same three-part shape as `Describer` and `Embedder`: a protocol, a
``config()`` recorded beside the output so a transcript says what produced it,
and a stub that needs no model so the stage can be exercised without one.

**A word carries its own timestamp, and that is the load-bearing detail.** A
transcript segmented into sentences is one segmentation among many, and the
one Whisper happened to choose. With per-word times the transcript can be
re-cut to any boundary afterwards -- a uniform grid, a scene cut propagated
from the video pass, a speaker change -- without re-running inference and
without losing a word. That is what lets the chunk boundaries be decided by
whichever modality the run says matters, rather than by whichever ran first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from ..source import Track


@dataclass
class Word:
    """One word, with the span it occupies. ``probability`` is the model's own
    confidence, kept because a low-confidence run is worth being able to see
    rather than having to infer from a bad answer downstream."""

    start: float
    end: float
    text: str
    probability: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "text": self.text,
                "probability": (round(self.probability, 4)
                                if self.probability is not None else None)}


@dataclass
class Segment:
    """A run of speech as the model chose to divide it.

    ``speaker`` is filled in later by ``align``, not by the transcriber:
    Whisper does not know who is talking and pyannote does not know what was
    said, and keeping the two passes separate is what lets either be swapped.
    """

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    speaker: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "text": self.text, "speaker": self.speaker,
                "words": [w.as_dict() for w in self.words]}


@dataclass
class Transcript:
    """Everything one pass over the audio produced.

    ``language_probability`` is reported rather than hidden because it is the
    cheapest signal that a track holds no speech: measured on silent CCTV,
    Whisper returned zero segments and named the language `cy` at p=0.41 --
    a confident answer would have been the surprising outcome.
    """

    language: str
    language_probability: float
    duration_s: float
    segments: list[Segment] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    def as_dict(self) -> dict[str, Any]:
        return {"language": self.language,
                "language_probability": round(self.language_probability, 4),
                "duration_s": round(self.duration_s, 3),
                "model": self.model,
                "segments": [s.as_dict() for s in self.segments]}


class Transcriber(Protocol):
    def transcribe(self, track: Track) -> Transcript:
        """The whole track at once. Never a window of it -- see `audio/__init__`."""
        ...

    def config(self) -> dict[str, Any]:
        """Recorded alongside the output, so a transcript says what made it."""
        ...
