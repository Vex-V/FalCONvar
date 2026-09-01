"""What a diarizer is: who spoke, and when. Never what they said.

Kept apart from transcription because the two answer different questions from
the same waveform, and because their labels only become one thing after an
alignment step that belongs to neither. A diarizer that also transcribed would
make swapping either model a rewrite of both.

**Speaker labels are per-file and mean nothing outside it.** They come from
clustering embeddings over the whole recording, so `SPEAKER_00` in one video
is unrelated to `SPEAKER_00` in another -- and would be unrelated between two
windows of the same file, which is the concrete reason diarization cannot be
run chunk by chunk. ``Diarization.embeddings`` is the way out when identity
across files is wanted: a 256-dimensional vector per speaker that can be
compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from ..source import Track


@dataclass
class Turn:
    """One continuous stretch attributed to one speaker."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "speaker": self.speaker}


@dataclass
class Diarization:
    """Every turn found, and the speakers they belong to.

    An empty result is a real answer, not a failure: measured on silent CCTV,
    pyannote returned zero speakers and zero turns in 0.1 s, which is the
    correct reading of a live microphone in an empty shop.
    """

    turns: list[Turn] = field(default_factory=list)
    embeddings: Optional[Any] = None          # (n_speakers, dim) ndarray
    model: dict[str, Any] = field(default_factory=dict)

    @property
    def speakers(self) -> list[str]:
        seen: list[str] = []
        for turn in self.turns:
            if turn.speaker not in seen:
                seen.append(turn.speaker)
        return seen

    @property
    def speech_s(self) -> float:
        return sum(t.duration for t in self.turns)

    def speaker_at(self, ts: float) -> Optional[str]:
        """Who is speaking at ``ts``, or None. Used by `align`."""
        for turn in self.turns:
            if turn.start <= ts <= turn.end:
                return turn.speaker
        return None

    def as_dict(self) -> dict[str, Any]:
        return {"speakers": self.speakers, "speech_s": round(self.speech_s, 3),
                "model": self.model, "turns": [t.as_dict() for t in self.turns]}


class Diarizer(Protocol):
    def diarize(self, track: Track) -> Diarization:
        """The whole track at once -- labels are only consistent within one pass."""
        ...

    def config(self) -> dict[str, Any]:
        ...
