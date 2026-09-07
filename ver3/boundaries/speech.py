"""3 · speech evidence -- cuts from a finished transcript.

The audio half of `boundaries`, and it lives here rather than in `listen/` on
purpose: deriving boundaries from pauses is boundary logic that happens to read
a transcript. Putting it in the audio module would make that module know
chunking exists, and `listen/` should not -- its whole job is to produce a
record of what was said, with no opinion about how it will later be divided.

It reads `transcript.raw.json` as a **file**, never by importing `listen`.
That is what keeps the import graph acyclic while the *run* order flips between
policies: on a `vad` run `listen` precedes this, on a `scene` run it does not,
and neither imports the other in either case.

Two policies, both answering "where should a chunk end?" from the soundtrack:

    vad       cut in the middle of a silence. Speech that runs together stays
              together; a pause long enough to notice becomes a boundary.
    speaker   cut where the voice changes. The strongest boundary a recording
              offers when more than one person is in it, and worthless when
              only one is -- which is why it returns nothing rather than
              failing, and `grid.enforce` divides the file evenly instead.

Neither applies a guard. Producing cuts and deciding what a chunk may be are
different jobs, and `grid.enforce` owns the second so that `min_s` means the
same thing under every policy.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..shared.documents import Cuts, RawTranscript

#: Below this, a gap between speech is a breath rather than a boundary.
#: Whisper's own segments on narration sit a median 4.8 s apart with sub-second
#: gaps between sentences, so a threshold under about half a second would cut
#: mid-paragraph on every clause.
DEFAULT_SILENCE_S = 0.65


def speech_spans(transcript: RawTranscript) -> list[tuple[float, float]]:
    """Where there is speech, from whichever pass knows.

    Diarization is preferred when present: it is a purpose-built voice-activity
    model, and its turns are what a speaker policy will cut on anyway. Whisper's
    segments are the fallback, so `vad` works without a second model or a gated
    download.
    """
    if transcript.turns:
        return sorted((t["start"], t["end"]) for t in transcript.turns)
    return sorted((s["start"], s["end"]) for s in transcript.segments)


def vad_cuts(transcript: RawTranscript,
             silence_s: float = DEFAULT_SILENCE_S) -> list[tuple[float, float]]:
    """Cut in the middle of every silence longer than ``silence_s``.

    The middle rather than either edge: a boundary at the end of speech clips a
    trailing word whose timestamp the model placed slightly late, and one at the
    start of the next clips its first. The middle of a gap belongs to neither
    utterance.

    Returns ``(cut_ts, gap_length)`` so the caller can cache the gap as the
    score -- the same reason `scenes` caches `content_val`. A silence threshold
    is then re-tunable over the cached series instead of by re-running Whisper,
    which is the expensive thing here by a wide margin.
    """
    spans = speech_spans(transcript)
    return [((end + start) / 2.0, start - end)
            for (_, end), (start, _) in zip(spans, spans[1:])
            if start - end >= silence_s]


def speaker_cuts(transcript: RawTranscript) -> list[tuple[float, float]]:
    """Cut wherever the speaker changes.

    Between the two turns, not on either, for the same reason as above. And
    consecutive turns by the same speaker are not a boundary -- which is what
    stops a pause inside one person's answer from becoming a chunk edge.

    The score carried alongside is the gap, so the two policies produce series
    of the same shape.
    """
    turns = sorted(transcript.turns, key=lambda t: t["start"])
    cuts: list[tuple[float, float]] = []
    for previous, current in zip(turns, turns[1:]):
        if current.get("speaker") == previous.get("speaker"):
            continue
        gap = current["start"] - previous["end"]
        at = ((previous["end"] + current["start"]) / 2.0 if gap > 0
              else current["start"])
        cuts.append((at, max(gap, 0.0)))
    return cuts


def detect(transcript: RawTranscript, policy: str,
           silence_s: float = DEFAULT_SILENCE_S) -> Cuts:
    """Cuts from a finished transcript, as a `Cuts` document."""
    if policy == "vad":
        scored = vad_cuts(transcript, silence_s)
        params: dict[str, Any] = {"silence_s": silence_s}
    elif policy == "speaker":
        if not transcript.turns:
            # Not an error. A recording with one voice, or none, offers no
            # speaker boundaries; `grid.enforce` then divides the file evenly,
            # which is the honest outcome rather than an invented one.
            scored = []
            params = {"speakers": 0}
        else:
            scored = speaker_cuts(transcript)
            params = {"speakers": len(transcript.speakers)}
    else:
        raise KeyError(f"unknown speech policy {policy!r}; known: vad, speaker")

    return Cuts(
        video_id=transcript.video_id,
        source="audio",
        detector=policy,
        params=params,
        cuts=[at for at, _ in scored],
        scores={
            "metric": "gap_s",
            "stride": 1,
            "at": [round(at, 3) for at, _ in scored],
            "values": [round(gap, 3) for _, gap in scored],
        },
        stats={"speech_spans": len(speech_spans(transcript)),
               "cuts_found": len(scored),
               "source": "turns" if transcript.turns else "segments"},
    )


def rethreshold(cuts: Cuts, silence_s: float) -> Cuts:
    """A different silence threshold over the cached gaps. No model reloaded.

    Only meaningful for `vad`: a speaker cut is not a thresholded quantity, so
    raising the number would silently drop real speaker changes.
    """
    if cuts.detector != "vad":
        raise ValueError(
            f"{cuts.detector!r} cuts are not thresholded, so there is nothing "
            "to retune; re-run the pass to change them")
    if not cuts.scores:
        raise ValueError("this cuts document carries no gap series")
    scored = list(zip(cuts.scores["at"], cuts.scores["values"]))
    kept = [(at, gap) for at, gap in scored if gap >= silence_s]
    return Cuts(
        video_id=cuts.video_id, source=cuts.source, detector=cuts.detector,
        params={**cuts.params, "silence_s": silence_s},
        cuts=[at for at, _ in kept], scores=cuts.scores,
        stats={**cuts.stats, "cuts_found": len(kept), "rethresholded": True},
    )


__all__ = ["DEFAULT_SILENCE_S", "speech_spans", "vad_cuts", "speaker_cuts",
           "detect", "rethreshold"]
