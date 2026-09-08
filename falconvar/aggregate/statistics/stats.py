"""Counts and coverage: how long, how many chunks, how many words."""

from __future__ import annotations

from typing import Any

from ..base import Context

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


__all__ = ["StatsAggregator"]
