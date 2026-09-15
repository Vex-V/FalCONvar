"""The listen component: `media.json` -> `transcript.raw.json`.

Defaults are named here explicitly rather than taken from the head of a
registry. `falconvar` built a form from `transcribe.available()` in order, so
the first option was `stub`; an audio-only run through it completed in 10.6 s,
reported 42 segments and 205 words, and wrote a transcript of
`[stub0.0][stub0.1]`. Nothing was wrong enough to report.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..media import load as load_media
from ..shared import env, paths
from ..shared.storage import sinks
from ..shared.contracts.documents import Produced, RawTranscript
from . import models
from .reader import listen
from .source import NoAudio

#: Named, not positional. See the module docstring.
DEFAULT_TRANSCRIBER = "whisper"
DEFAULT_DIARIZER = "pyannote"


def run(video_id: str,
        transcriber: str = DEFAULT_TRANSCRIBER,
        diarizer: str = DEFAULT_DIARIZER,
        model: Optional[str] = None,
        language: Optional[str] = None,
        sink: str | Sequence[str] = "file") -> Produced:
    """Transcribe and diarize the whole file. Writes no chunk ids."""
    # Before a model is constructed, not after: pyannote is gated and reads a
    # token from the environment at load time.
    env.load()
    media = load_media(video_id)
    if not media.has_audio:
        raise NoAudio(f"{video_id} has no audio stream")

    built: dict[str, object] = {}
    if model:
        built["model"] = model
    if language:
        built["language"] = language

    raw = listen(media.path,
                 models.transcriber(transcriber, **built),
                 models.diarizer(diarizer),
                 video_id=video_id)
    written = sinks.write(video_id, "raw_transcript", raw.as_dict(), sink)
    return Produced(
        video_id=video_id, component="audio", backend=",".join(written),
        artifacts={"raw_transcript": written.get("file", "")},
        stats={**raw.stats, "silent": raw.silent},
        skipped=["transcribe", "diarize"] if raw.silent else [],
    )


def load(video_id: str) -> RawTranscript:
    return RawTranscript.from_dict(
        sinks.read_json(paths.artifact(video_id, "raw_transcript")))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Transcribe and diarize a whole file. No chunking.")
    ap.add_argument("video_id")
    ap.add_argument("--transcriber", default=DEFAULT_TRANSCRIBER,
                    choices=sorted(models.TRANSCRIBERS))
    ap.add_argument("--diarizer", default=DEFAULT_DIARIZER,
                    choices=sorted(models.DIARIZERS))
    ap.add_argument("--model", default=None, help="whisper: tiny|base|small|...")
    ap.add_argument("--language", default=None, help="skip detection")
    ap.add_argument("--sink", default="file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        produced = run(args.video_id, args.transcriber, args.diarizer,
                       args.model, args.language, args.sink)
    except (NoAudio, KeyError, FileNotFoundError, models.ModelUnavailable,
            sinks.UnknownBackend) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    s = produced.stats
    print(f"{produced.video_id}")
    if s.get("silent"):
        print("  silent -- no model was loaded, and no speech is the finding")
    else:
        print(f"  segments     {s['segments']}   words {s['words']}")
        print(f"  speakers     {s['speakers']}   turns {s['turns']}   "
              f"speech {s['speech_s']:g}s")
        print(f"  attributed   {s['attributed']}/{s['words']} words")
        print(f"  decode       {s['decode_s']:.2f}s   "
              f"transcribe {s['transcribe_s']:.2f}s   "
              f"diarize {s['diarize_s']:.2f}s")
    print()
    print(f"raw transcript -> {produced.artifacts['raw_transcript']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
