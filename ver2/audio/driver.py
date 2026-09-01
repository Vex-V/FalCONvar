"""Command line for the audio stage: a media file in, a transcript out.

Runs audio alone. When both streams matter, `ver2.driver` orchestrates the two
and decides which one owns the chunk boundaries; this is the same work without
the video half, and it is what to reach for when the picture is irrelevant or
absent.

The chunking flags mirror the video driver's, because the grid is a property
of the run rather than of a modality: `--chunking uniform` needs neither model
pass, while `vad` and `speaker` are derived from the audio itself and so are
only available here and in the combined driver.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):                       # allow running as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ver2 import db, timeline as timeline_mod
from ver2.audio import diarize as diarize_mod
from ver2.audio import segment as segment_mod
from ver2.audio import source
from ver2.audio import transcribe as transcribe_mod
from ver2.audio.diarize.pyannote_diarizer import DiarizerUnavailable
from ver2.audio.output import (MultiTranscriptSink, TranscriptDocument,
                               build_document)
from ver2.audio.reader import Result, cut, listen
from ver2.audio.transcribe.whisper import TranscriberUnavailable

POLICIES = ("uniform", "vad", "speaker")


def report(result: Result, timeline, out: Path, names: list[str]) -> None:
    s = result.stats
    print()
    print(f"  audio        {s['duration_s']:.1f}s   rms {s['rms']:.4f}"
          + ("   SILENT -- no speech found" if s["silent"] else ""))
    print(f"  decode       {s['decode_s']:.2f}s")
    print(f"  transcribe   {s['transcribe_s']:.2f}s   {s['segments']} segments, "
          f"{s['words']} words")
    print(f"  diarize      {s['diarize_s']:.2f}s   {s['speakers']} speakers, "
          f"{s['turns']} turns, {s['speech_s']:.1f}s speech")
    print(f"  chunks       {s.get('chunks', 0)}   "
          f"({s.get('chunks_with_speech', 0)} with speech)   "
          f"{timeline.policy} grid, fingerprint {timeline.fingerprint()}")
    print()
    for chunk in result.chunks[:4]:
        who = ",".join(chunk["structured"]["speakers"]) or "--"
        print(f"    {chunk['chunk_id']:>3} {chunk['start_ts']:>7.2f}-{chunk['end_ts']:<7.2f} "
              f"{who:<12} {chunk['text'][:64]}")
    where = [str(out) if n == "file" else "supabase" for n in names]
    print("\ntranscript -> " + "\n              ".join(where))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Transcribe and diarize a media file, then cut it to a grid.")
    ap.add_argument("media", type=Path, help="video or audio file")
    ap.add_argument("--video-id", default=None,
                    help="identity for the output (default: the file stem)")
    ap.add_argument("--transcriber", default="whisper",
                    choices=transcribe_mod.available(),
                    help="which transcriber (default whisper; stub needs no model)")
    ap.add_argument("--model", default=None, help="model id for the transcriber")
    ap.add_argument("--language", default=None,
                    help="skip language detection and assume this one")
    ap.add_argument("--diarizer", default="pyannote",
                    choices=diarize_mod.available(),
                    help="which diarizer (default pyannote; `none` skips it)")
    ap.add_argument("--chunking", default="uniform", choices=POLICIES,
                    help="how to cut the transcript into chunks (default uniform)")
    ap.add_argument("--chunk-duration", type=float, default=20.0,
                    help="uniform: chunk length; vad/speaker: the maximum (default 20)")
    ap.add_argument("--min-chunk", type=float, default=5.0,
                    help="vad/speaker: shortest chunk to emit (default 5)")
    ap.add_argument("--silence", type=float, default=segment_mod.DEFAULT_SILENCE_S,
                    help="vad: a gap this long or longer is a boundary "
                         f"(default {segment_mod.DEFAULT_SILENCE_S})")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="document path (default out/<video-id>/transcript.json)")
    ap.add_argument("--sink", default="file",
                    help="comma-separated: file, supabase "
                         "(default file; `file,supabase` writes both)")
    args = ap.parse_args()

    if args.chunking == "speaker" and args.diarizer == "none":
        ap.error("--chunking speaker needs a diarizer; `--diarizer none` cannot "
                 "produce speaker boundaries")

    names = [n.strip() for n in args.sink.split(",") if n.strip()]
    if any(n not in ("file", "supabase") for n in names) or not names:
        ap.error("--sink: expected file and/or supabase")

    db.load_env()
    video_id = args.video_id or args.media.stem
    out = args.out or Path("out") / video_id / "transcript.json"

    info = source.probe(args.media)
    if not info.has_audio:
        print(f"{args.media} has no audio stream; nothing to transcribe.",
              file=sys.stderr)
        return 1
    print(f"{args.media}  ({video_id})")
    print(f"  {info.codec} {info.rate} Hz {info.channels}ch, {info.duration_s:.1f}s",
          flush=True)

    options = {"model": args.model} if args.model else {}
    if args.language:
        options["language"] = args.language
    try:
        transcriber = transcribe_mod.build(args.transcriber, **options)
        diarizer = diarize_mod.build(args.diarizer)
    except (TranscriberUnavailable, DiarizerUnavailable, TypeError) as exc:
        print(exc, file=sys.stderr)
        return 2

    result = listen(str(args.media), transcriber, diarizer)

    # The grid, after the pass: `uniform` needs only the duration, while vad
    # and speaker need the transcript that has just been produced.
    if args.chunking == "uniform":
        timeline = timeline_mod.uniform(result.track.duration_s, args.chunk_duration)
    else:
        timeline = segment_mod.build(
            args.chunking, result.track.duration_s, result.transcript,
            result.diarization, min_s=args.min_chunk,
            max_s=args.chunk_duration, silence_s=args.silence)
    cut(result, timeline)

    sink = _sink(names, out)
    document = build_document(
        video_id=video_id, uri=str(args.media), transcript=result.transcript,
        diarization=result.diarization, chunks=result.chunks, timeline=timeline,
        audio=info.as_dict(), stats=result.stats)
    sink.write(document)
    report(result, timeline, out, names)
    return 0


def _sink(names: list[str], out: Path):
    """First named is authoritative; the rest are best-effort. See ver2.fanout."""
    built = []
    for name in names:
        if name == "file":
            built.append(TranscriptDocument(out))
        else:
            from ver2.audio.output import SupabaseTranscript

            built.append(SupabaseTranscript())
    return built[0] if len(built) == 1 else MultiTranscriptSink(*built)


if __name__ == "__main__":
    raise SystemExit(main())
