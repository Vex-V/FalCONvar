"""Turning a finished transcript into chunk boundaries.

Two policies live here, and both answer the same question -- where should a
chunk end? -- from the soundtrack rather than the picture. They are the audio
half of the choice `ver2/timeline.py` describes; `uniform` needs neither pass
and `scene` belongs to the video side.

    vad       cut in the middle of a silence. Speech that runs together stays
              together, and a pause long enough to notice becomes a boundary.
    speaker   cut where the voice changes. The strongest boundary a recording
              offers when more than one person is in it, and worthless when
              only one is -- which is why it falls back rather than failing.

Both produce interior cut times and hand them to `timeline.from_cuts`, then to
`timeline.enforce`, which applies the minimum and maximum lengths. That order
matters: a raw voice-activity cut list on conversational audio has a boundary
every second or two, and without the guards the video pass would be shredded
into chunks too short to describe.

A producer never decides the guards' values and never writes a timeline. It
returns cuts. What they become is the run's decision, made once, in one place.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ...timeline import Timeline, enforce, from_cuts
from ..diarize.base import Diarization
from ..transcribe.base import Transcript
from .cut import to_chunks

#: Below this, a gap between speech is a breath rather than a boundary.
#: Whisper's own segments on narration sit a median 4.8 s apart with sub-second
#: gaps between sentences, so a threshold under about half a second would cut
#: mid-paragraph on every clause.
DEFAULT_SILENCE_S = 0.65


def speech_spans(transcript: Transcript,
                 diarization: Optional[Diarization] = None) -> list[tuple[float, float]]:
    """Where there is speech, from whichever pass knows.

    Diarization is preferred when present: it is a purpose-built voice
    activity model and its turns are what a speaker policy will cut on anyway.
    Whisper's segments are the fallback, so `vad` works without paying for a
    second model or a gated download.
    """
    if diarization is not None and diarization.turns:
        return [(t.start, t.end) for t in diarization.turns]
    return [(s.start, s.end) for s in transcript.segments]


def vad_cuts(transcript: Transcript, diarization: Optional[Diarization] = None,
             silence_s: float = DEFAULT_SILENCE_S) -> list[float]:
    """Cut in the middle of every silence longer than ``silence_s``.

    The middle rather than either edge: a boundary at the end of speech clips
    a trailing word whose timestamp the model placed slightly late, and one at
    the start of the next clips its first. The middle of a gap is the only
    point in it that belongs to neither utterance.
    """
    spans = sorted(speech_spans(transcript, diarization))
    cuts = []
    for (_, end), (start, _) in zip(spans, spans[1:]):
        if start - end >= silence_s:
            cuts.append((end + start) / 2.0)
    return cuts


def speaker_cuts(diarization: Diarization) -> list[float]:
    """Cut wherever the speaker changes.

    Between the two turns, not on either, for the same reason as above -- and
    consecutive turns by the same speaker are not a boundary, which is what
    stops a pause inside one person's answer from becoming a chunk edge.
    """
    turns = sorted(diarization.turns, key=lambda t: t.start)
    cuts = []
    for previous, current in zip(turns, turns[1:]):
        if current.speaker != previous.speaker:
            cuts.append((previous.end + current.start) / 2.0
                        if current.start > previous.end else current.start)
    return cuts


def build(policy: str, duration_s: float, transcript: Transcript,
          diarization: Optional[Diarization] = None,
          min_s: float = 5.0, max_s: Optional[float] = 30.0,
          silence_s: float = DEFAULT_SILENCE_S) -> Timeline:
    """One audio-derived timeline, guards applied.

    A policy that finds nothing to cut on returns a single chunk covering the
    file, which `enforce`'s maximum then divides evenly. That is the honest
    outcome for a monologue asked to be split on speaker changes: there are
    none, and inventing some would be worse than saying so through the grid.
    """
    if policy == "vad":
        cuts = vad_cuts(transcript, diarization, silence_s)
        params: dict[str, Any] = {"silence_s": silence_s}
    elif policy == "speaker":
        if diarization is None:
            raise ValueError("the speaker policy needs a diarizer; "
                             "--diarizer none cannot produce speaker boundaries")
        cuts = speaker_cuts(diarization)
        params = {"speakers": len(diarization.speakers)}
    else:
        raise KeyError(f"unknown audio policy {policy!r}; known: vad, speaker")

    params.update({"min_s": min_s, "max_s": max_s})
    raw = from_cuts(cuts, duration_s, policy, params, derived_from="audio")
    return Timeline(enforce(raw.spans, min_s, max_s), policy, params, "audio")


AVAILABLE: dict[str, Callable] = {"vad": vad_cuts, "speaker": speaker_cuts}


def available() -> list[str]:
    return sorted(AVAILABLE)


__all__ = ["DEFAULT_SILENCE_S", "available", "build", "speaker_cuts",
           "speech_spans", "to_chunks", "vad_cuts"]
