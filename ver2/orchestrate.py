"""One media file, both streams, one chunk grid -- without a terminal.

This is what `ver2/driver.py` used to hold inline, extracted because it now has
two callers with nothing in common: the CLI, and the HTTP API. The rule the
stage drivers already follow ("driver.py: the CLI, argparse only, no pipeline
logic") was the one this module's own driver broke, and a server importing an
argparse module to reach the work behind it would have made that permanent.

Progress arrives through a callback rather than `print`, because the two
callers want opposite things from it: a terminal wants lines as they happen, a
job runner wants to store the latest state and answer a poll with it. Neither
belongs here, so neither is here.

The policy decides the order, and that is the whole reason this is one function
rather than a sequence the caller composes:

    uniform   arithmetic. Both sides derive the same grid; nothing propagates.
    scene     the video pass finds the cuts while decoding.
    vad       the audio pass finds them, in the gaps between speech.
    speaker   the audio pass finds them, where the voice changes.

For `uniform` and `scene` the timeline is read back off the chunks the video
pass actually produced. Computing it separately would be a second
implementation of the same boundaries that could disagree with the first --
about a corrected final end, about a merged short tail -- with nothing to
report it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ver2 import db
from ver2.audio import diarize as diarize_mod
from ver2.audio import segment as segment_mod
from ver2.audio import source as audio_source
from ver2.audio import transcribe as transcribe_mod
from ver2.audio.output import (MultiTranscriptSink, TranscriptDocument,
                               build_document)
from ver2.audio.reader import cut, listen
from ver2.timeline import Timeline
from ver2.video.ingest.output import FileManifestWriter, FrameStore, MultiSink
from ver2.video.ingest.pipeline import ingest

#: Policies whose boundaries come from the soundtrack, so audio runs first.
AUDIO_FIRST = ("vad", "speaker")
POLICIES = ("uniform", "scene", "vad", "speaker")
SINKS = ("file", "supabase")


@dataclass
class Options:
    """Everything a run can be told. Defaults match the CLI's."""

    media: Path
    video_id: Optional[str] = None
    out_root: Path = Path("out")

    chunking: str = "uniform"
    chunk_duration: float = 20.0
    min_chunk: float = 5.0
    silence: float = segment_mod.DEFAULT_SILENCE_S
    scene_threshold: float = 27.0
    scene_min_duration: float = 5.0

    samplers: Sequence[Any] = ()          # built Sampler objects
    per_second: float = 1.0
    frame_store: bool = False
    store_scope: str = "sampled"

    use_audio: bool = True
    transcriber: str = "whisper"
    audio_model: Optional[str] = None
    language: Optional[str] = None
    diarizer: str = "pyannote"

    sinks: Sequence[str] = ("file",)


@dataclass
class Outcome:
    """What a run produced, for a caller that cannot read the terminal."""

    video_id: str
    out_dir: Path
    timeline: Timeline
    video_chunks: int
    frames_sampled: int
    audio: dict[str, Any] = field(default_factory=dict)
    used_audio: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "out_dir": str(self.out_dir),
            "chunks": len(self.timeline),
            "policy": self.timeline.policy,
            "derived_from": self.timeline.derived_from,
            "timeline_fingerprint": self.timeline.fingerprint(),
            "frames_sampled": self.frames_sampled,
            "audio": self.audio if self.used_audio else None,
        }


def validate(options: Options) -> list[str]:
    """Everything wrong with these options, as messages. Empty means valid.

    Returned rather than raised so a CLI can print them all at once and an API
    can answer 422 with the list, instead of each learning the rules again.
    """
    problems = []
    if options.chunking not in POLICIES:
        problems.append(f"chunking must be one of {', '.join(POLICIES)}")
    if options.chunking == "speaker" and options.diarizer == "none":
        problems.append("chunking 'speaker' needs a diarizer; 'none' cannot "
                        "produce speaker boundaries")
    bad = [s for s in options.sinks if s not in SINKS]
    if bad or not options.sinks:
        problems.append(f"sinks must be some of {', '.join(SINKS)}")
    if not Path(options.media).exists():
        problems.append(f"{options.media} does not exist")
    if options.chunk_duration <= 0:
        problems.append("chunk_duration must be positive")
    return problems


def _video_duration(media: Path) -> Optional[float]:
    """How long the picture runs, by the same arithmetic the pipeline uses."""
    from ver2.video.ingest.source import probe as video_probe

    try:
        info = video_probe(str(media))
    except Exception:                                  # noqa: BLE001
        return None
    if not (info.frame_count and info.fps_trusted and info.fps):
        return None
    return info.frame_count / info.fps


def _cover(timeline: Timeline, duration: Optional[float]) -> Timeline:
    """Stretch the last span so the grid reaches ``duration``.

    A shared grid has to span the whole file, and a file is as long as its
    longest stream -- on the reference file the audio ends 16 ms before the
    video, and a grid built from the audio alone leaves the last frames outside
    every chunk.
    """
    if duration is None or not timeline.spans or duration <= timeline.duration_s:
        return timeline
    spans = timeline.spans[:-1] + [(timeline.spans[-1][0], duration)]
    return Timeline(spans, timeline.policy, timeline.params, timeline.derived_from)


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Temp file plus os.replace, so a reader never sees a torn document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _manifest_sinks(names: Sequence[str], out_dir: Path):
    built = []
    for name in names:
        if name == "file":
            built.append(FileManifestWriter(out_dir / "manifest.json"))
        else:
            from ver2.video.ingest.output import SupabaseManifestWriter

            built.append(SupabaseManifestWriter())
    return built[0] if len(built) == 1 else MultiSink(*built)


def _transcript_sinks(names: Sequence[str], out: Path):
    built = []
    for name in names:
        if name == "file":
            built.append(TranscriptDocument(out))
        else:
            from ver2.audio.output import SupabaseTranscript

            built.append(SupabaseTranscript())
    return built[0] if len(built) == 1 else MultiTranscriptSink(*built)


def process(options: Options,
            on_progress: Optional[Callable[[str, dict], None]] = None) -> Outcome:
    """Run both streams onto one grid. Raises nothing a caller must translate.

    ``on_progress(stage, detail)`` is called as each stage starts and finishes.
    A CLI prints it; a job runner stores it. Absent, the run is silent.
    """
    say = on_progress or (lambda stage, detail: None)
    problems = validate(options)
    if problems:
        raise ValueError("; ".join(problems))

    db.load_env()
    media = Path(options.media)
    video_id = options.video_id or media.stem
    out_dir = Path(options.out_root) / video_id

    audio_info = audio_source.probe(media)
    use_audio = audio_info.has_audio and options.use_audio
    if options.chunking in AUDIO_FIRST and not use_audio:
        raise ValueError(f"chunking '{options.chunking}' is derived from the "
                         f"soundtrack, and this run will read none")
    say("start", {"video_id": video_id, "audio": audio_info.as_dict(),
                  "use_audio": use_audio, "chunking": options.chunking})

    transcriber = diarizer = None
    if use_audio:
        built: dict[str, Any] = {}
        if options.audio_model:
            built["model"] = options.audio_model
        if options.language:
            built["language"] = options.language
        transcriber = transcribe_mod.build(options.transcriber, **built)
        diarizer = diarize_mod.build(options.diarizer)

    audio_result = None
    timeline: Optional[Timeline] = None
    if options.chunking in AUDIO_FIRST:
        say("audio", {"why": "deciding the grid"})
        audio_result = listen(str(media), transcriber, diarizer)
        timeline = segment_mod.build(
            options.chunking, audio_result.track.duration_s,
            audio_result.transcript, audio_result.diarization,
            min_s=options.min_chunk, max_s=options.chunk_duration,
            silence_s=options.silence)
        timeline = _cover(timeline, _video_duration(media))
        say("grid", {"chunks": len(timeline), "policy": timeline.policy,
                     "fingerprint": timeline.fingerprint()})

    store = FrameStore(out_dir / "store") if options.frame_store else None
    chunking_options: dict[str, Any] = {}
    if options.chunking == "scene":
        chunking_options = {"threshold": options.scene_threshold,
                            "min_duration_s": options.scene_min_duration}
    elif timeline is not None:
        chunking_options = {"timeline": timeline}

    say("video", {"why": "one decode pass"})
    video_result = ingest(
        str(media),
        per_second=options.per_second,
        chunking="fixed" if timeline is not None else options.chunking,
        chunking_options=chunking_options,
        chunk_duration_s=options.chunk_duration,
        sampler_specs=list(options.samplers),
        video_id=video_id,
        sink=_manifest_sinks(options.sinks, out_dir),
        store=store,
        store_scope=options.store_scope,
    )
    say("video", {"chunks": len(video_result.chunks),
                  "frames_sampled": video_result.frames_sampled})

    if timeline is None:
        timeline = Timeline(
            spans=[(c.start_ts, c.end_ts) for c in video_result.chunks],
            policy=options.chunking,
            params={"duration_s": options.chunk_duration},
            derived_from="video")
    write_json(out_dir / "timeline.json", timeline.as_dict())

    audio_stats: dict[str, Any] = {}
    if use_audio:
        if audio_result is None:
            say("audio", {"why": "cutting to the video's grid"})
            audio_result = listen(str(media), transcriber, diarizer)
        cut(audio_result, timeline)
        document = build_document(
            video_id=video_id, uri=str(media),
            transcript=audio_result.transcript,
            diarization=audio_result.diarization,
            chunks=audio_result.chunks, timeline=timeline,
            audio=audio_info.as_dict(), stats=audio_result.stats)
        _transcript_sinks(options.sinks, out_dir / "transcript.json").write(document)
        audio_stats = audio_result.stats
        say("audio", audio_stats)

    outcome = Outcome(video_id=video_id, out_dir=out_dir, timeline=timeline,
                      video_chunks=len(video_result.chunks),
                      frames_sampled=video_result.frames_sampled,
                      audio=audio_stats, used_audio=use_audio)
    say("done", outcome.as_dict())
    return outcome
