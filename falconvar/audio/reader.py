"""`listen()` -- decode, transcribe, diarize, attribute.

Four steps over the whole file, in order. No chunking.

A word goes to the speaker talking at its midpoint, and a segment to whoever
holds most of it by duration: the two models estimate edges independently, so
word spans routinely straddle a turn boundary and a sentence can begin during
the previous speaker's tail.

A track below `SILENCE_RMS` gets no model at all and says so.
"""

from __future__ import annotations

import time
from typing import Optional

from ..shared.documents import RawTranscript
from . import source
from .models import Diarization, Diarizer, Transcriber, Transcript


def speaker_at(diarization: Diarization, ts: float) -> Optional[str]:
    """Who was speaking at ``ts``. None when nobody was, or nobody knows."""
    for turn in diarization.turns:
        if turn.start <= ts < turn.end:
            return turn.speaker
    return None


def attribute(transcript: Transcript, diarization: Diarization) -> None:
    """Give every word a speaker, and every segment the one who holds most of it.

    In place, because the alternative is a parallel structure keyed by index
    that has to be kept in step with the transcript by hand.
    """
    if not diarization.turns:
        return
    for segment in transcript.segments:
        held: dict[str, float] = {}
        for word in segment.words:
            word.speaker = speaker_at(diarization, word.midpoint)
            if word.speaker:
                held[word.speaker] = held.get(word.speaker, 0.0) + (
                    word.end - word.start)
        # By duration, not by count: a sentence beginning in the previous
        # speaker's tail should not be attributed to them for holding three
        # short words of it.
        segment.speaker = max(held, key=held.get) if held else None


def listen(uri: str, transcriber: Transcriber, diarizer: Diarizer,
           video_id: str = "") -> RawTranscript:
    """The whole audio pass, as the document it produces."""
    started = time.perf_counter()

    t0 = time.perf_counter()
    track = source.load(uri)
    decode_s = time.perf_counter() - t0

    if track.silent:
        # No model is loaded at all. "No speech" is the finding.
        return RawTranscript(
            video_id=video_id,
            model={**transcriber.config(), **diarizer.config(),
                   "ran": False, "why": "track is silent"},
            track=track.as_dict(),
            stats={"decode_s": round(decode_s, 3), "segments": 0, "words": 0,
                   "speakers": 0, "turns": 0, "speech_s": 0.0,
                   "elapsed_s": round(time.perf_counter() - started, 3)},
        )

    t0 = time.perf_counter()
    transcript = transcriber.transcribe(track)
    transcribe_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    diarization = diarizer.diarize(track)
    diarize_s = time.perf_counter() - t0

    attribute(transcript, diarization)

    return RawTranscript(
        video_id=video_id,
        model={**transcriber.config(), **diarizer.config(), "ran": True,
               "language": transcript.language,
               "language_probability": transcript.language_probability},
        track=track.as_dict(),
        segments=[s.as_dict() for s in transcript.segments],
        words=[w.as_dict() for w in transcript.words],
        turns=[t.as_dict() for t in diarization.turns],
        stats={
            "segments": len(transcript.segments),
            "words": len(transcript.words),
            "speakers": len(diarization.speakers),
            "turns": len(diarization.turns),
            "speech_s": round(diarization.speech_s, 3),
            "attributed": sum(1 for w in transcript.words if w.speaker),
            "decode_s": round(decode_s, 3),
            "transcribe_s": round(transcribe_s, 3),
            "diarize_s": round(diarize_s, 3),
            "elapsed_s": round(time.perf_counter() - started, 3),
        },
    )


__all__ = ["speaker_at", "attribute", "listen"]
