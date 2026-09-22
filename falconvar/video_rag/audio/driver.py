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
from ...shared import env, paths
from ...shared.storage import sinks
from ...shared.contracts.documents import Produced, RawTranscript
from . import models
from .reader import listen
from .source import NoAudio

#: Named, not positional. See the module docstring.
DEFAULT_TRANSCRIBER = "whisper"
DEFAULT_DIARIZER = "pyannote"

#: This component's settings, and the constructor argument each one sets on
#: each half. Two maps rather than one because the two models are configured
#: independently and both happen to call their checkpoint `model`: `model` is
#: the transcriber's and `diarizer_model` the diarizer's, which is also how
#: `PyannoteDiarizer.config()` already reports it. `device` is in both, because
#: it is a fact about the machine rather than about either model.
SPEECH_SETTINGS = {"model": "model", "language": "language",
                   "vad_filter": "vad_filter", "compute_type": "compute_type",
                   "device": "device"}
VOICE_SETTINGS = {"diarizer_model": "model", "exclusive": "exclusive",
                  "device": "device"}


def _for(mapping: dict[str, str], takes: set[str],
         named: dict[str, object]) -> dict[str, object]:
    """The subset of ``named`` this backend can actually be told."""
    return {argument: named[setting] for setting, argument in mapping.items()
            if setting in named and argument in takes}


def audio(video_id: str,
          transcriber: str = DEFAULT_TRANSCRIBER,
          diarizer: str = DEFAULT_DIARIZER,
          model: Optional[str] = None,
          language: Optional[str] = None,
          vad_filter: Optional[bool] = None,
          compute_type: Optional[str] = None,
          device: Optional[str] = None,
          diarizer_model: Optional[str] = None,
          exclusive: Optional[bool] = None,
          sink: str | Sequence[str] = "file") -> Produced:
    """Transcribe and diarize the whole file. Writes no chunk ids.

    Every setting defaults to None, meaning *leave the backend's own default*,
    so a run that names none of them constructs exactly what it always did.

    `exclusive` is the one worth knowing about. pyannote answers with
    overlapping speech either resolved or not, and the whole grid rests on the
    resolved form: a word cannot belong to two speakers, and chunk boundaries
    derived from turns must not overlap. It was previously unreachable -- the
    diarizer was built with no arguments at all -- so the safe reading was the
    only reading. Turning it off is a deliberate choice about overlapping
    speech, not a tuning knob.

    A setting no chosen backend takes is refused rather than dropped. `stub`
    has no `language` and `none` has no `exclusive`, and quietly ignoring one
    would mean a run that reports success having transcribed under settings
    nobody asked for -- the same silence the named defaults at the top of this
    file exist to prevent.
    """
    # Before a model is constructed, not after: pyannote is gated and reads a
    # token from the environment at load time.
    env.load()
    media = load_media(video_id)
    if not media.has_audio:
        raise NoAudio(f"{video_id} has no audio stream")

    named: dict[str, object] = {
        setting: value for setting, value in
        (("model", model), ("language", language), ("vad_filter", vad_filter),
         ("compute_type", compute_type), ("device", device),
         ("diarizer_model", diarizer_model), ("exclusive", exclusive))
        if value is not None}

    speech = _for(SPEECH_SETTINGS, models.settings("transcriber", transcriber),
                  named)
    voices = _for(VOICE_SETTINGS, models.settings("diarizer", diarizer), named)
    reached = ({s for s in SPEECH_SETTINGS if SPEECH_SETTINGS[s] in speech}
               | {s for s in VOICE_SETTINGS if VOICE_SETTINGS[s] in voices})
    unreachable = sorted(set(named) - reached)
    if unreachable:
        raise ValueError(
            f"transcriber {transcriber!r} and diarizer {diarizer!r} take no "
            f"{', '.join(repr(u) for u in unreachable)}")

    raw = listen(media.path,
                 models.transcriber(transcriber, **speech),
                 models.diarizer(diarizer, **voices),
                 video_id=video_id)
    written = sinks.write(video_id, "raw_transcript", raw.as_dict(), sink)
    return Produced(
        video_id=video_id, component="audio", backend=",".join(written),
        artifacts={"raw_transcript": written.get("file", "")},
        stats={**raw.stats, "silent": raw.silent},
        skipped=["transcribe", "diarize"] if raw.silent else [],
    )


#: The uniform name every component also answers to: what a dispatch
#: table calls and what a form introspects. The same function object.
#: See `media/driver.py` for why the function is named for its component.
run = audio


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
    ap.add_argument("--no-vad-filter", dest="vad_filter", action="store_false",
                    default=None,
                    help="whisper: transcribe the whole track, including what "
                         "its voice-activity filter would drop")
    ap.add_argument("--compute-type", default=None,
                    help="whisper: float16 | int8 | ... (default float16 on "
                         "cuda, int8 on cpu)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda "
                                                   "where torch finds one)")
    ap.add_argument("--diarizer-model", default=None,
                    help="pyannote: a checkpoint id (default "
                         "pyannote/speaker-diarization-3.1). Named here rather "
                         "than read off the backend, which is imported lazily")
    ap.add_argument("--overlaps", dest="exclusive", action="store_false",
                    default=None,
                    help="pyannote: keep overlapping speech rather than "
                         "resolving it. A word then belongs to two speakers")
    ap.add_argument("--sink", default="file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        produced = run(args.video_id, args.transcriber, args.diarizer,
                       args.model, args.language, args.vad_filter,
                       args.compute_type, args.device, args.diarizer_model,
                       args.exclusive, args.sink)
    except (NoAudio, KeyError, ValueError, FileNotFoundError,
            models.ModelUnavailable, sinks.UnknownBackend) as exc:
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
