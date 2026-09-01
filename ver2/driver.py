"""Command line for a whole media file: both streams, one chunk grid.

Argument parsing and terminal output, and nothing else. The work is in
`ver2/orchestrate.py`, which the HTTP API calls directly -- a server reaching
the pipeline through an argparse module would have made this file's own
violation of "driver.py: the CLI, argparse only, no pipeline logic" permanent.

Progress reaches the terminal through `orchestrate.process`'s `on_progress`
callback, so one run can print lines here and be polled as job state there,
without either caller knowing about the other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):                       # allow running as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ver2 import orchestrate
from ver2.audio import diarize as diarize_mod
from ver2.audio import segment as segment_mod
from ver2.audio import transcribe as transcribe_mod
from ver2.orchestrate import POLICIES
from ver2.video.ingest.driver import _build_samplers
from ver2.video.ingest.source import UnusableSource


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


def options_from(args) -> orchestrate.Options:
    """An argparse namespace as the options the pipeline actually takes."""
    return orchestrate.Options(
        media=args.media, video_id=args.video_id,
        chunking=args.chunking, chunk_duration=args.chunk_duration,
        min_chunk=args.min_chunk, silence=args.silence,
        scene_threshold=args.scene_threshold,
        scene_min_duration=args.scene_min_duration,
        samplers=_build_samplers(args), per_second=args.per_second,
        frame_store=bool(args.frame_store), store_scope=args.store_scope,
        use_audio=not args.no_audio, transcriber=args.transcriber,
        audio_model=args.audio_model, language=args.language,
        diarizer=args.diarizer,
        sinks=[n.strip() for n in args.sink.split(",") if n.strip()],
    )


def report(stage: str, detail: dict) -> None:
    """One line per stage. The terminal's half of `on_progress`."""
    gap = "\n"
    if stage == "start":
        audio = detail["audio"]
        if audio["has_audio"]:
            described = (f"{audio['codec']} {audio['rate']} Hz "
                         f"{audio['channels']}ch {audio['duration_s']:.1f}s")
            if not detail["use_audio"]:
                described += " (ignored)"
        else:
            described = "none"
        print(f"  audio: {described}")
        print(f"  grid : {detail['chunking']}", flush=True)
    elif stage == "audio" and "why" in detail:
        print(f"{gap}[audio] whole-file pass, {detail['why']}", flush=True)
    elif stage == "audio":
        print(f"        {detail['segments']} segments, {detail['words']} words, "
              f"{detail['speakers']} speakers -> {detail['chunks']} chunks "
              f"({detail['chunks_with_speech']} with speech)")
    elif stage == "grid":
        print(f"        {detail['chunks']} chunks from {detail['policy']}, "
              f"fingerprint {detail['fingerprint']}")
    elif stage == "video" and "why" in detail:
        print(f"{gap}[video] {detail['why']}", flush=True)
    elif stage == "video":
        print(f"        {detail['chunks']} chunks, "
              f"{detail['frames_sampled']} frames sampled")
    elif stage == "done":
        print(f"{gap}grid     {detail['chunks']} chunks, {detail['policy']}, "
              f"from {detail['derived_from']}, "
              f"fingerprint {detail['timeline_fingerprint']}")
        print(f"out      {detail['out_dir']}/")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Process one media file end to end: video and audio, "
                    "on one chunk grid.")
    add_arguments(ap)
    args = ap.parse_args()

    options = options_from(args)
    problems = orchestrate.validate(options)
    if problems:
        ap.error("; ".join(problems))

    print(f"{args.media}  ({options.video_id or args.media.stem})")
    try:
        orchestrate.process(options, on_progress=report)
    except (UnusableSource, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
