"""Applying a grid to a finished transcript."""

from __future__ import annotations

from typing import Any

from ..shared.contracts.documents import RawTranscript, Timeline


def _midpoint(word: dict[str, Any]) -> float:
    return (float(word["start"]) + float(word["end"])) / 2.0


def _turns_of(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Contiguous runs of one speaker, as bound records.

    One entry per run rather than per word, and each carries its own span and
    text -- the same shape a person-detector returns for people, and for the
    same reason: a flat list of speakers beside a flat list of sentences does
    not say who said what.
    """
    turns: list[dict[str, Any]] = []
    for word in words:
        speaker = word.get("speaker")
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["end"] = round(float(word["end"]), 3)
            turns[-1]["text"] += word["text"]
        else:
            turns.append({"speaker": speaker,
                          "start": round(float(word["start"]), 3),
                          "end": round(float(word["end"]), 3),
                          "text": word["text"]})
    for turn in turns:
        turn["text"] = turn["text"].strip()
    return turns


def to_chunks(raw: RawTranscript, timeline: Timeline) -> list[dict[str, Any]]:
    """Every chunk in the grid, with the words whose midpoints land in it.

    Every chunk, including the silent ones. `chunk_id` is shared with the
    manifest, so dropping the quiet chunks would renumber everything after them
    and break the correspondence with nothing reporting it.
    """
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(len(timeline))]
    placed = lost = 0
    for word in raw.words:
        index = timeline.index_at(_midpoint(word))
        if index is None:
            # Outside the grid entirely. Counted rather than silently dropped:
            # a transcript that loses words should say how many.
            lost += 1
            continue
        buckets[index].append(word)
        placed += 1

    chunks: list[dict[str, Any]] = []
    for chunk_id, words in enumerate(buckets):
        turns = _turns_of(words)
        speakers = sorted({t["speaker"] for t in turns if t["speaker"]})
        chunks.append({
            "chunk_id": chunk_id,
            "text": " ".join(w["text"] for w in words).strip(),
            "word_count": len(words),
            # `speakers` only. `turns[].text` *is* the transcript, so putting
            # the turns in `structured` would append the whole chunk a second
            # time interleaved with timestamps read as numbers -- 337
            # characters where 151 was right.
            "structured": {"speakers": speakers},
            "turns": turns,
        })
    return chunks


def stats_for(raw: RawTranscript, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    placed = sum(c["word_count"] for c in chunks)
    return {
        "chunks": len(chunks),
        "chunks_with_speech": sum(1 for c in chunks if c["word_count"]),
        "words": placed,
        "words_in_transcript": len(raw.words),
        "words_outside_grid": len(raw.words) - placed,
        "speakers": len(raw.speakers),
    }


__all__ = ["to_chunks", "stats_for"]
