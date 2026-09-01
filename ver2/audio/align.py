"""Two passes over one waveform, joined on time.

Whisper says what was said and when each word was said. pyannote says who was
speaking and when. Neither knows the other's answer, and that separation is
deliberate -- either model can be swapped without touching the other. This is
the one place the two are brought together, and it is arithmetic rather than
inference.

**A word is attributed by its midpoint, not by its span.** Word spans routinely
straddle a speaker boundary by a few tens of milliseconds, because the two
models estimate their edges independently and neither is exact; asking which
turn *contains* a word would leave those unattributed for no good reason.
Asking which turn contains its midpoint always yields an answer when a turn is
anywhere near, and picks the speaker with more of the word when it straddles.

**A segment's speaker is the one who spoke most of its words**, not the one at
its start. Whisper's segments are sentence-shaped and a sentence can begin
during the previous speaker's tail.

**Nothing is invented when diarization found nothing.** With the `none`
diarizer, or on a track with no speech, every word keeps ``speaker=None`` and
segments are returned unchanged. A transcript with no speaker information is a
truthful artifact; one where every word was assigned to a fabricated
`SPEAKER_00` is not.
"""

from __future__ import annotations

import collections
from typing import Optional

from .diarize.base import Diarization
from .transcribe.base import Segment, Transcript


def speaker_for(diarization: Diarization, start: float, end: float) -> Optional[str]:
    """Who is speaking across ``[start, end]``, by midpoint then by overlap.

    The midpoint answers almost every word. The overlap fallback matters for a
    long segment whose midpoint happens to land in a pause between two turns.
    """
    direct = diarization.speaker_at((start + end) / 2.0)
    if direct is not None:
        return direct
    best, best_overlap = None, 0.0
    for turn in diarization.turns:
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap > best_overlap:
            best, best_overlap = turn.speaker, overlap
    return best


def attribute(transcript: Transcript, diarization: Diarization) -> Transcript:
    """Label each segment with the speaker who said most of it. Returns it."""
    if not diarization.turns:
        return transcript
    for segment in transcript.segments:
        spoken: collections.Counter[str] = collections.Counter()
        for word in segment.words:
            who = speaker_for(diarization, word.start, word.end)
            if who is not None:
                # Weighted by duration, not by word count: one long word from
                # the previous speaker should not outvote three short ones.
                spoken[who] += word.end - word.start
        segment.speaker = (max(spoken, key=spoken.get) if spoken
                           else speaker_for(diarization, segment.start, segment.end))
    return transcript


def split_on_speaker(transcript: Transcript, diarization: Diarization) -> list[Segment]:
    """Re-cut segments so no segment spans a speaker change.

    Whisper segments on sentences and pyannote on voices, and the two disagree:
    one Whisper segment can contain the end of an answer and the start of the
    next question. A segment attributed to whoever said most of it is a
    reasonable summary but a poor retrieval unit, because half its words belong
    to someone else. This produces units that are true rather than
    approximately true, which is what the `speaker` chunking policy is built on.
    """
    out: list[Segment] = []
    for segment in transcript.segments:
        if not segment.words:
            out.append(segment)
            continue
        run: list = []
        current: Optional[str] = None
        for word in segment.words:
            who = speaker_for(diarization, word.start, word.end)
            if run and who != current:
                out.append(_segment_of(run, current))
                run = []
            current = who
            run.append(word)
        if run:
            out.append(_segment_of(run, current))
    return out


def _segment_of(words: list, speaker: Optional[str]) -> Segment:
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text="".join(w.text for w in words).strip(),
        words=list(words),
        speaker=speaker,
    )
