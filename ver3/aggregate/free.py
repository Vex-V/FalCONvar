"""The free tier: arithmetic over documents already produced.

No model, no network, no GPU. Everything here is a count, a ratio or a span,
and each answers something similarity answers approximately or not at all --
"how much of this is speech", "who dominated", "which chunk is busiest".
"""

from __future__ import annotations

from typing import Any

from .base import Context


class StatsAggregator:
    name = "stats"
    tier = "free"
    about = "counts and coverage: chunks, samplers, words, frames"
    depends_on: tuple[str, ...] = ()

    def run(self, context: Context) -> dict[str, Any]:
        spans = [e - s for s, e in context.timeline.spans]
        described = {}
        if context.descriptions is not None:
            for chunk in context.descriptions.chunks:
                for sid in chunk.get("samplers", {}):
                    described[sid] = described.get(sid, 0) + 1

        frames = 0
        if context.manifest is not None:
            frames = context.manifest.stats.get("frames_sampled", 0)

        words = 0
        speech_chunks = 0
        if context.transcript is not None:
            words = sum(c.get("word_count", 0) for c in context.transcript.chunks)
            speech_chunks = sum(1 for c in context.transcript.chunks
                                if c.get("word_count"))

        duration = context.timeline.duration_s
        return {
            "duration_s": round(duration, 3),
            "chunks": len(context.timeline),
            "chunk_s": {"min": round(min(spans), 3) if spans else 0.0,
                        "median": round(sorted(spans)[len(spans) // 2], 3) if spans else 0.0,
                        "max": round(max(spans), 3) if spans else 0.0},
            "policy": context.timeline.policy,
            "derived_from": context.timeline.derived_from,
            "frames_sampled": frames,
            "descriptions_by_sampler": described,
            "words": words,
            "chunks_with_speech": speech_chunks,
            "words_per_minute": round(words / (duration / 60), 1) if duration else 0.0,
        }


class SpeakersAggregator:
    name = "speakers"
    tier = "free"
    about = "who spoke, for how long, and how often the voice changed"
    depends_on = ("transcript",)

    def run(self, context: Context) -> dict[str, Any]:
        held: dict[str, float] = {}
        handovers = 0
        previous = None
        turn_count = 0
        for chunk in context.transcript.chunks:
            for turn in chunk.get("turns", []):
                speaker = turn.get("speaker")
                if speaker is None:
                    continue
                turn_count += 1
                held[speaker] = held.get(speaker, 0.0) + (
                    float(turn["end"]) - float(turn["start"]))
                if previous is not None and speaker != previous:
                    handovers += 1
                previous = speaker

        duration = context.timeline.duration_s
        speech = sum(held.values())
        ranked = sorted(held.items(), key=lambda kv: -kv[1])
        return {
            "speakers": len(held),
            "turns": turn_count,
            "handovers": handovers,
            # True for one voice with no handovers. A fact, not a judgement:
            # it is what a single-narrator documentary looks like from here.
            "monologue": len(held) <= 1 and handovers == 0,
            "speech_s": round(speech, 3),
            "speech_ratio": round(speech / duration, 4) if duration else 0.0,
            "by_speaker": [{"speaker": s, "seconds": round(v, 3),
                            "share": round(v / speech, 4) if speech else 0.0}
                           for s, v in ranked],
            "dominant": ranked[0][0] if ranked else None,
        }


class CoverageAggregator:
    name = "coverage"
    tier = "free"
    about = "which chunks have an account, from which modality"
    depends_on: tuple[str, ...] = ()

    def run(self, context: Context) -> dict[str, Any]:
        rows = []
        both = picture_only = sound_only = neither = 0
        for chunk_id in context.chunk_ids():
            said = context.text_of(chunk_id)
            has_sound = bool(said.get("transcript"))
            has_picture = bool(set(said) - {"transcript"})
            if has_picture and has_sound:
                both += 1
            elif has_picture:
                picture_only += 1
            elif has_sound:
                sound_only += 1
            else:
                neither += 1
            start, end = context.span_of(chunk_id)
            rows.append({"chunk_id": chunk_id,
                         "start_ts": round(start, 3), "end_ts": round(end, 3),
                         "sources": sorted(said)})
        return {
            "both": both, "picture_only": picture_only,
            "sound_only": sound_only, "neither": neither,
            # A chunk nothing described is a hole in the index, and it is worth
            # naming rather than inferring from a count.
            "silent_chunks": [r["chunk_id"] for r in rows if not r["sources"]],
            "chunks": rows,
        }


__all__ = ["StatsAggregator", "SpeakersAggregator", "CoverageAggregator"]
