"""Applying a grid to a finished transcript.

This is the step the whole design rests on. Transcription and diarization run
over the whole file and know nothing about chunks; the grid arrives afterwards
and may have been derived from the picture, from the soundtrack, or from
arithmetic. Cutting here is exact and cheap because Whisper timestamps every
word -- so a transcript can be re-cut any number of times, to any grid, at no
cost and with no word lost.

**A word belongs to the chunk containing its midpoint.** The same rule
`align` uses to attribute a word to a speaker, and for the same reason: a word
that straddles a boundary belongs to whichever side holds more of it, and
every word must land in exactly one chunk or the text is either duplicated or
dropped.

**Speaker turns survive a coarse grid rather than being flattened by it.** A
chunk carries `turns` -- one bound record per contiguous run of one speaker,
with its own span and its own words -- so a twenty-second window holding three
speakers says who said what, in order. That is the same shape the `yolo`
describer returns for people and the `text` describer for signage, and it
exists for the same reason: parallel lists of speakers and sentences cannot
say which went with which, and cannot be made to afterwards.
"""

from __future__ import annotations

from typing import Any, Optional

from ...timeline import Timeline
from ..diarize.base import Diarization
from ..transcribe.base import Transcript, Word
from ..align import speaker_for


def _turns_of(words: list[tuple[Word, Optional[str]]]) -> list[dict[str, Any]]:
    """Contiguous runs of one speaker, as bound records."""
    turns: list[dict[str, Any]] = []
    for word, speaker in words:
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["end"] = round(word.end, 3)
            turns[-1]["text"] += word.text
        else:
            turns.append({"speaker": speaker, "start": round(word.start, 3),
                          "end": round(word.end, 3), "text": word.text})
    for turn in turns:
        turn["text"] = turn["text"].strip()
    return turns


def to_chunks(transcript: Transcript, timeline: Timeline,
              diarization: Optional[Diarization] = None) -> list[dict[str, Any]]:
    """One record per chunk of the grid, whether or not anything was said in it.

    Silent chunks are kept with empty text on purpose. The grid is shared with
    the video side, so `chunk_id` has to mean the same thing in both documents;
    dropping the quiet ones would renumber every chunk after them and silently
    break that correspondence. A reader distinguishes them by `word_count`.
    """
    buckets: list[list[tuple[Word, Optional[str]]]] = [[] for _ in timeline.spans]
    for segment in transcript.segments:
        for word in segment.words:
            index = timeline.index_at((word.start + word.end) / 2.0)
            if index is None:                 # a word past the end of the grid
                index = len(buckets) - 1
            speaker = (speaker_for(diarization, word.start, word.end)
                       if diarization is not None and diarization.turns
                       else segment.speaker)
            buckets[index].append((word, speaker))

    chunks: list[dict[str, Any]] = []
    for chunk_id, ((start, end), words) in enumerate(zip(timeline.spans, buckets)):
        turns = _turns_of(words)
        speakers: list[str] = []
        for turn in turns:
            if turn["speaker"] and turn["speaker"] not in speakers:
                speakers.append(turn["speaker"])
        chunks.append({
            "chunk_id": chunk_id,
            "start_ts": round(start, 3),
            "end_ts": round(end, 3),
            "text": " ".join(t["text"] for t in turns if t["text"]).strip(),
            "word_count": len(words),
            "structured": {"speakers": speakers, "turns": turns},
        })
    return chunks
