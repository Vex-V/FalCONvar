"""Who spoke, how much, and how the conversation was shaped.

Derived entirely from turns already on disk, so it costs nothing and answers
what retrieval handles badly: who dominated, who barely spoke, where the
back-and-forth was densest. Those are questions about *proportion*, and a
ranked list of similar passages cannot express one.

**Speaker labels are per-video and mean nothing across videos.** They come from
clustering embeddings over one recording, so `SPEAKER_00` here and
`SPEAKER_00` in another file are unrelated strings. The turns are reported
under those labels because that is what the diarizer actually established;
`audio_transcripts.segments` carries the 256-d embeddings that *would* let two
files be compared, and doing that is a different job than this one.
"""

from __future__ import annotations

from typing import Any, Optional

from .base import Context


class SpeakerStatsAggregator:
    """Per-speaker time, turns and words, plus the shape of the exchange."""

    id = "speakers"
    tier = "free"
    depends_on = ("transcript",)

    def aggregate(self, ctx: Context) -> Optional[dict[str, Any]]:
        turns = [{**turn, "chunk_id": chunk["chunk_id"]}
                 for chunk in ctx.chunks for turn in chunk["turns"]]
        if not turns:
            return None
        turns.sort(key=lambda t: t["start"])

        per_speaker: dict[str, dict] = {}
        for turn in turns:
            # A turn with no speaker is a real case: with `--diarizer none`, or
            # on a track where diarization found nothing, every word is
            # unattributed. Bucketing those under a name would invent one.
            speaker = turn.get("speaker") or "unattributed"
            entry = per_speaker.setdefault(speaker, {
                "speaker": speaker, "turns": 0, "seconds": 0.0, "words": 0,
                "first_seen": turn["start"], "last_seen": turn["end"],
                "longest_turn": 0.0, "chunks": set(),
            })
            length = max(turn["end"] - turn["start"], 0.0)
            entry["turns"] += 1
            entry["seconds"] += length
            entry["words"] += len((turn.get("text") or "").split())
            entry["first_seen"] = min(entry["first_seen"], turn["start"])
            entry["last_seen"] = max(entry["last_seen"], turn["end"])
            entry["longest_turn"] = max(entry["longest_turn"], length)
            entry["chunks"].add(turn["chunk_id"])

        total = sum(e["seconds"] for e in per_speaker.values()) or 1.0
        speakers = sorted(
            ({**e,
              "seconds": round(e["seconds"], 2),
              "longest_turn": round(e["longest_turn"], 2),
              "first_seen": round(e["first_seen"], 2),
              "last_seen": round(e["last_seen"], 2),
              "share": round(e["seconds"] / total, 3),
              "words_per_second": (round(e["words"] / e["seconds"], 2)
                                   if e["seconds"] else 0.0),
              "chunks": sorted(e["chunks"])}
             for e in per_speaker.values()),
            key=lambda s: -s["seconds"])

        # A handover is the conversation changing hands. On a monologue this is
        # zero, which is the useful thing it says.
        handovers = sum(1 for a, b in zip(turns, turns[1:])
                        if a.get("speaker") != b.get("speaker"))

        return {
            "speakers": speakers,
            "speaker_count": len(speakers),
            "total_turns": len(turns),
            "handovers": handovers,
            "total_speech_seconds": round(total, 2),
            "dominant_speaker": speakers[0]["speaker"] if speakers else None,
            "monologue": handovers == 0 and len(speakers) == 1,
            "timeline": [{"start_ts": round(t["start"], 2),
                          "end_ts": round(t["end"], 2),
                          "speaker": t.get("speaker"),
                          "chunk_id": t["chunk_id"]} for t in turns],
        }

    def config(self) -> dict[str, Any]:
        return {"id": self.id, "tier": self.tier}
