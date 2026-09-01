"""The audio pass: a file in, a transcript out.

One function, and its shape is the argument for why audio is a separate
component. There is no loop over chunks here because there is nothing to loop
over -- decode, transcribe, diarize and align each run once over the whole
file, and only then is a grid applied.

The grid is a parameter, not a decision made here. When the run's policy is
audio-derived this returns before a timeline exists and the caller builds one
from `segment.build`; when the policy is `uniform` or `scene` the caller
already has one. Either way the cut is the same call, because a finished
transcript does not care where its boundaries came from.

Nothing here writes anything. The CLI assembles sinks; this reads audio, calls
two models, and keeps count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from . import align, source
from .diarize.base import Diarization, Diarizer
from .segment.cut import to_chunks
from .transcribe.base import Transcript, Transcriber
from ..timeline import Timeline


@dataclass
class Result:
    track: source.Track
    transcript: Transcript
    diarization: Diarization
    chunks: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def silent(self) -> bool:
        return not self.transcript.segments


def listen(
    uri: str,
    transcriber: Transcriber,
    diarizer: Diarizer,
    timeline: Optional[Timeline] = None,
) -> Result:
    """Decode, transcribe, diarize, attribute -- and cut, if a grid is given.

    ``timeline`` is optional because of the ordering the policies impose. An
    audio-derived grid cannot exist until this has run, so the caller invokes
    this without one, builds the timeline from the result, and calls
    ``cut`` afterwards. A video-derived or arithmetic grid is known already and
    can be passed straight in.
    """
    started = time.perf_counter()

    t0 = time.perf_counter()
    track = source.load(uri)
    decode_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    transcript = transcriber.transcribe(track)
    transcribe_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    diarization = diarizer.diarize(track)
    diarize_s = time.perf_counter() - t0

    align.attribute(transcript, diarization)

    stats = {
        "duration_s": round(track.duration_s, 3),
        "silent": track.silent,
        "rms": round(track.rms, 6),
        "segments": len(transcript.segments),
        "words": len(transcript.words),
        "speakers": len(diarization.speakers),
        "turns": len(diarization.turns),
        "speech_s": round(diarization.speech_s, 3),
        "decode_s": round(decode_s, 3),
        "transcribe_s": round(transcribe_s, 3),
        "diarize_s": round(diarize_s, 3),
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    result = Result(track, transcript, diarization, stats=stats)
    if timeline is not None:
        cut(result, timeline)
    return result


def cut(result: Result, timeline: Timeline) -> Result:
    """Apply a grid to an already-finished transcript. Cheap and repeatable.

    Separate from ``listen`` because it is the only part that depends on the
    boundaries, and because it can be called again with a different grid
    without touching a model -- which is the property the whole ordering
    argument rests on.
    """
    result.chunks = to_chunks(result.transcript, timeline, result.diarization)
    result.stats["chunks"] = len(result.chunks)
    result.stats["chunks_with_speech"] = sum(1 for c in result.chunks
                                             if c["word_count"])
    return result
