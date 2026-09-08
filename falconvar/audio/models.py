"""The two model families, their protocols, and a lazy registry each.

`Word`, `Segment`, `Transcript` for what was said; `Turn`, `Diarization` for
who spoke. Plus a stub transcriber and a no-op diarizer that load nothing.

Nothing is imported until a registry is asked for it by name: between them the
real backends pull in torch, and a stub run should pay for neither.

A per-word timestamp is what lets a finished transcript be re-cut to any grid
at no cost, which is the property the whole ordering argument rests on.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from .source import Track


class ModelUnavailable(Exception):
    """No model, no runtime, or a load that will not succeed by retrying."""


# ---------------------------------------------------------------- transcribe

@dataclass
class Word:
    """One word, with the span the model placed it in.

    A per-word timestamp is the thing that makes the whole ordering argument
    work: it is what lets a finished transcript be re-cut to any grid, any
    number of times, at no cost and with no word lost.
    """

    start: float
    end: float
    text: str
    speaker: Optional[str] = None

    @property
    def midpoint(self) -> float:
        """Where this word *belongs*.

        A word straddling a boundary belongs to whichever side holds more of
        it, and the two models estimate edges independently -- word spans
        routinely cross a turn boundary by tens of milliseconds. The same rule
        attributes a word to a speaker and to a chunk.
        """
        return (self.start + self.end) / 2.0

    def as_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "text": self.text, "speaker": self.speaker}


@dataclass
class Segment:
    """One utterance as the transcriber grouped it."""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    speaker: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "text": self.text, "speaker": self.speaker}


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    language: Optional[str] = None
    language_probability: Optional[float] = None

    @property
    def words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())


class Transcriber(Protocol):
    def transcribe(self, track: Track) -> Transcript: ...
    def config(self) -> dict[str, Any]: ...


# ------------------------------------------------------------------ diarize

@dataclass
class Turn:
    speaker: str
    start: float
    end: float

    def as_dict(self) -> dict[str, Any]:
        return {"speaker": self.speaker, "start": round(self.start, 3),
                "end": round(self.end, 3)}


@dataclass
class Diarization:
    turns: list[Turn] = field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        return sorted({t.speaker for t in self.turns})

    @property
    def speech_s(self) -> float:
        return sum(t.end - t.start for t in self.turns)


class Diarizer(Protocol):
    def diarize(self, track: Track) -> Diarization: ...
    def config(self) -> dict[str, Any]: ...


# ----------------------------------------------------------------- registry

class StubTranscriber:
    """Returns fixed text without loading anything.

    Exists so the pipeline can be exercised end to end with no model, and it is
    deliberately obvious in the output. `falconvar` learned why that matters:
    a form that offered the registry in alphabetical order defaulted to the
    stub, and an audio-only run completed in 10.6 s reporting 42 segments and
    205 words, having written a transcript of `[stub0.0][stub0.1]`. Nothing was
    wrong enough to report. Defaults are named explicitly here for that reason.
    """

    name = "stub"

    def __init__(self, every_s: float = 5.0) -> None:
        self.every_s = every_s

    def transcribe(self, track: Track) -> Transcript:
        segments: list[Segment] = []
        t = 0.0
        while t < track.duration_s:
            end = min(t + self.every_s, track.duration_s)
            text = f"[stub {t:.1f}]"
            segments.append(Segment(t, end, text,
                                    [Word(t, end, text)]))
            t += self.every_s
        return Transcript(segments, language="stub", language_probability=1.0)

    def config(self) -> dict[str, Any]:
        return {"transcriber": self.name, "every_s": self.every_s}


class NoDiarizer:
    """No speaker information, said plainly.

    Every word keeps `speaker=None`, which is truthful. Labelling everything
    `SPEAKER_00` is not, and is worse than saying nothing because it reads as a
    finding.
    """

    name = "none"

    def diarize(self, track: Track) -> Diarization:
        return Diarization([])

    def config(self) -> dict[str, Any]:
        return {"diarizer": self.name}


#: name -> "module:Class", resolved on first use so nothing heavy is imported
#: for a run that does not ask for it.
TRANSCRIBERS: dict[str, Any] = {"stub": StubTranscriber,
                                "whisper": "backends.whisper:WhisperTranscriber"}
DIARIZERS: dict[str, Any] = {"none": NoDiarizer,
                             "pyannote": "backends.pyannote:PyannoteDiarizer"}


def _resolve(registry: dict[str, Any], name: str, kind: str):
    if name not in registry:
        raise KeyError(f"unknown {kind} {name!r}; known: {', '.join(registry)}")
    entry = registry[name]
    if isinstance(entry, str):
        module_name, class_name = entry.split(":")
        module = importlib.import_module(f".{module_name}", __package__)
        entry = getattr(module, class_name)
        registry[name] = entry
    return entry


def transcriber(name: str, **kwargs) -> Transcriber:
    return _resolve(TRANSCRIBERS, name, "transcriber")(**kwargs)


def diarizer(name: str, **kwargs) -> Diarizer:
    return _resolve(DIARIZERS, name, "diarizer")(**kwargs)


def available() -> dict[str, list[str]]:
    return {"transcribers": sorted(TRANSCRIBERS), "diarizers": sorted(DIARIZERS)}


__all__ = ["ModelUnavailable", "Word", "Segment", "Transcript", "Transcriber",
           "Turn", "Diarization", "Diarizer", "StubTranscriber", "NoDiarizer",
           "TRANSCRIBERS", "DIARIZERS", "transcriber", "diarizer", "available"]
