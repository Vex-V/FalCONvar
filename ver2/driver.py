"""One media file, both streams, one chunk grid.

    open the file -> split the streams -> decide the policy -> run both sides

The stage drivers each do one thing and can be run alone; this is what runs
them together, and the only thing it adds is the answer to a question neither
can answer by itself: **where do the chunk boundaries go when a file has both a
picture and a soundtrack?**

The policy decides, and the policy decides the order:

    uniform   arithmetic. Video runs, audio runs, both land on the same grid
              because both derive it from the same duration and length.
    scene     the video pass finds the cuts while decoding. Audio needs no
              boundaries to transcribe, so it could run alongside; it runs
              after only because doing both at once on one GPU is slower than
              doing them in turn.
    vad       the audio pass finds them, in the gaps between speech.
    speaker   the audio pass finds them, where the voice changes.

For `uniform` and `scene` the timeline is **derived from the chunks the video
pass actually produced**, not computed alongside them. Computing it separately
would be a second implementation of the same boundaries, and the two could
disagree -- about the final chunk's true end, about a merged short tail --
with nothing to report it. Reading it back off the result makes drift
impossible rather than unlikely.

For `vad` and `speaker` the grid cannot be derived that way, because the video
pass needs it before it starts. There the audio pass runs first and hands
`FixedChunker` a list it did not compute.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):                       # allow running as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ver2 import db
from ver2 import timeline as timeline_mod
from ver2.audio import diarize as diarize_mod
from ver2.audio import segment as segment_mod
from ver2.audio import source as audio_source
from ver2.audio import transcribe as transcribe_mod
from ver2.audio.output import (MultiTranscriptSink, TranscriptDocument,
                               build_document)
from ver2.audio.reader import cut, listen
from ver2.timeline import Timeline
from ver2.video.ingest.driver import _build_samplers
from ver2.video.ingest.output import FileManifestWriter, FrameStore, MultiSink
from ver2.video.ingest.pipeline import ingest
from ver2.video.ingest.source import UnusableSource

#: Which pass has to run first for each policy, and which module owns it.
AUDIO_FIRST = ("vad", "speaker")
POLICIES = ("uniform", "scene", "vad", "speaker")


def timeline_of(result, policy: str, params: dict[str, Any]) -> Timeline:
    """The grid the video pass actually used, read back off its chunks."""
    return Timeline(spans=[(c.start_ts, c.end_ts) for c in result.chunks],
                    policy=policy, params=params, derived_from="video")


def _video_duration(media: Path) -> Optional[float]:
    """How long the picture runs, by the same arithmetic the pipeline uses.

    Frame count over frame rate, not the container's stated duration: those are
    two independently corruptible signals and the pipeline already refuses to
    trust the second one.
    """
    from ver2.video.ingest.source import probe as video_probe

    try:
        info = video_probe(str(media))
    except Exception:                                  # noqa: BLE001
        return None
    if not (info.frame_count and info.fps_trusted and info.fps):
        return None
    return info.frame_count / info.fps


def _cover(timeline: Timeline, duration: Optional[float]) -> Timeline:
    """Stretch the last span so the grid reaches ``duration``."""
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


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("media", type=Path, help="the file to process")
    ap.add_argument("--video-id", default=None, help="default: the file stem")
    ap.add_argument("--chunking", default="uniform", choices=POLICIES,
                    help="which modality decides the chunk boundaries "
                         "(default uniform: neither, both derive the same grid)")
    ap.add_argument("--chunk-duration", type=float, default=20.0,
                    help="uniform: chunk length; everything else: the maximum")
    ap.add_argument("--min-chunk", type=float, default=5.0,
                    help="vad/speaker: shortest chunk to emit (default 5)")
    ap.add_argument("--silence", type=float, default=segment_mod.DEFAULT_SILENCE_S,
                    help="vad: a gap this long or longer is a boundary")
    ap.add_argument("--scene-threshold", type=float, default=27.0)
    ap.add_argument("--scene-min-duration", type=float, default=5.0)

    video = ap.add_argument_group("video")
    video.add_argument("--per-second", type=float, default=1.0)
    video.add_argument("--sampler", default="clip")
    video.add_argument("--every-seconds", type=float, default=3.0,
                       help="uniform/overview: seconds between kept frames")
    video.add_argument("--threshold", type=float, default=None)
    video.add_argument("--vocabulary", default=None)
    video.add_argument("--mode", default="reference")
    video.add_argument("--min-interval", type=float, default=0.0)
    video.add_argument("--max-per-chunk", type=int, default=None)
    video.add_argument("--frame-store", nargs="?", const=True, default=None)
    video.add_argument("--store-scope", default="sampled",
                       choices=("sampled", "decimated"))

    audio = ap.add_argument_group("audio")
    audio.add_argument("--transcriber", default="whisper",
                       choices=transcribe_mod.available())
    audio.add_argument("--audio-model", default=None,
                       help="model id for the transcriber")
    audio.add_argument("--language", default=None)
    audio.add_argument("--diarizer", default="pyannote",
                       choices=diarize_mod.available())
    audio.add_argument("--no-audio", action="store_true",
                       help="ignore the soundtrack even if there is one")

    ap.add_argument("--sink", default="file",
                    help="comma-separated: file, supabase (default file)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Process one media file end to end: video and audio, "
                    "on one chunk grid.")
    add_arguments(ap)
    args = ap.parse_args()

    if args.chunking == "speaker" and args.diarizer == "none":
        ap.error("--chunking speaker needs a diarizer; `--diarizer none` cannot "
                 "produce speaker boundaries")
    names = [n.strip() for n in args.sink.split(",") if n.strip()]
    if any(n not in ("file", "supabase") for n in names) or not names:
        ap.error("--sink: expected file and/or supabase")

    db.load_env()
    video_id = args.video_id or args.media.stem
    out_dir = Path("out") / video_id

    audio_info = audio_source.probe(args.media)
    use_audio = audio_info.has_audio and not args.no_audio
    if args.chunking in AUDIO_FIRST and not use_audio:
        ap.error(f"--chunking {args.chunking} is derived from the soundtrack, "
                 f"and {args.media} has none that this run will read")

    print(f"{args.media}  ({video_id})")
    print(f"  audio: " + (f"{audio_info.codec} {audio_info.rate} Hz "
                          f"{audio_info.channels}ch {audio_info.duration_s:.1f}s"
                          if audio_info.has_audio else "none")
          + (" (ignored)" if audio_info.has_audio and args.no_audio else ""))
    print(f"  grid : {args.chunking}", flush=True)

    # ---------------------------------------------------------------- audio
    audio_result = None
    transcriber = diarizer = None
    if use_audio:
        options = {"model": args.audio_model} if args.audio_model else {}
        if args.language:
            options["language"] = args.language
        transcriber = transcribe_mod.build(args.transcriber, **options)
        diarizer = diarize_mod.build(args.diarizer)

    timeline: Optional[Timeline] = None
    if args.chunking in AUDIO_FIRST:
        print("\n[audio] whole-file pass, to decide the grid", flush=True)
        audio_result = listen(str(args.media), transcriber, diarizer)
        timeline = segment_mod.build(
            args.chunking, audio_result.track.duration_s, audio_result.transcript,
            audio_result.diarization, min_s=args.min_chunk,
            max_s=args.chunk_duration, silence_s=args.silence)
        # A shared grid has to span the whole file, and a file is as long as
        # its longest stream. Containers do not promise the two agree -- on the
        # reference file the audio ends 16 ms before the video -- so a grid
        # built from the audio duration alone leaves the last video frames
        # outside every chunk, and the manifest and the timeline then disagree
        # about a boundary that neither pass actually chose.
        timeline = _cover(timeline, _video_duration(args.media))
        print(f"        {len(timeline)} chunks from {args.chunking}, "
              f"fingerprint {timeline.fingerprint()}")

    # ---------------------------------------------------------------- video
    store = None
    if args.frame_store:
        where = (out_dir / "store" if args.frame_store is True
                 else Path(args.frame_store))
        store = FrameStore(where)
    sinks = [FileManifestWriter(out_dir / "manifest.json") if n == "file"
             else _supabase_manifest() for n in names]
    sink = sinks[0] if len(sinks) == 1 else MultiSink(*sinks)

    chunking_options: dict[str, Any] = {}
    if args.chunking == "scene":
        chunking_options = {"threshold": args.scene_threshold,
                            "min_duration_s": args.scene_min_duration}
    elif timeline is not None:
        chunking_options = {"timeline": timeline}

    print("\n[video] one decode pass", flush=True)
    try:
        video_result = ingest(
            str(args.media),
            per_second=args.per_second,
            chunking="fixed" if timeline is not None else args.chunking,
            chunking_options=chunking_options,
            chunk_duration_s=args.chunk_duration,
            sampler_specs=_build_samplers(args),
            video_id=video_id, sink=sink, store=store,
            store_scope=args.store_scope,
        )
    except UnusableSource as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"        {len(video_result.chunks)} chunks, "
          f"{video_result.frames_sampled} frames sampled")

    # The grid, read back off what the video pass produced -- unless the audio
    # pass supplied it, in which case the video pass was told to honour it.
    if timeline is None:
        timeline = timeline_of(video_result, args.chunking,
                               {"duration_s": args.chunk_duration})
    write_json(out_dir / "timeline.json", timeline.as_dict())

    # ------------------------------------------------------------ transcript
    if use_audio:
        if audio_result is None:
            print("\n[audio] whole-file pass, cut to the video's grid", flush=True)
            audio_result = listen(str(args.media), transcriber, diarizer)
        cut(audio_result, timeline)
        document = build_document(
            video_id=video_id, uri=str(args.media),
            transcript=audio_result.transcript,
            diarization=audio_result.diarization,
            chunks=audio_result.chunks, timeline=timeline,
            audio=audio_info.as_dict(), stats=audio_result.stats)
        _transcript_sink(names, out_dir / "transcript.json").write(document)
        s = audio_result.stats
        print(f"        {s['segments']} segments, {s['words']} words, "
              f"{s['speakers']} speakers -> {s['chunks']} chunks "
              f"({s['chunks_with_speech']} with speech)")

    print(f"\ngrid     {len(timeline)} chunks, {timeline.policy}, "
          f"from {timeline.derived_from}, fingerprint {timeline.fingerprint()}")
    print(f"out      {out_dir}/  manifest.json, timeline.json"
          + (", transcript.json" if use_audio else "")
          + (", store/" if store else ""))
    return 0


def _transcript_sink(names: list[str], out: Path):
    built = []
    for name in names:
        if name == "file":
            built.append(TranscriptDocument(out))
        else:
            from ver2.audio.output import SupabaseTranscript

            built.append(SupabaseTranscript())
    return built[0] if len(built) == 1 else MultiTranscriptSink(*built)


def _supabase_manifest():
    from ver2.video.ingest.output import SupabaseManifestWriter

    return SupabaseManifestWriter()


if __name__ == "__main__":
    raise SystemExit(main())
