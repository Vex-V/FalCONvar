"""Whisper, through faster-whisper.

CTranslate2 rather than the reference PyTorch implementation: same weights,
several times faster, and it is what is installed. Measured on an RTX 4060,
`small` in float16 transcribed 205 s of narration in 7.7 s -- about 26x
realtime -- with 428 word-level timestamps.

The whole track goes in at once. Whisper carries context across an utterance
and detects language from the opening seconds, so feeding it windows costs
both; see `audio/__init__` for why that is structural rather than a preference.

`vad_filter` is left on. It is Silero VAD running ahead of the model to drop
non-speech regions, which saves time on sparse audio and, more usefully, stops
the decoder looping on silence -- Whisper's best-known failure. Measured on a
silent track it made no difference to the output (zero segments either way),
so it is kept as insurance rather than because it was observed to rescue
anything here.
"""

from __future__ import annotations

from typing import Any, Optional

from .. import cuda
from ..source import Track
from .base import Segment, Transcript, Word

#: Model ids change faster than this file will, so a plain default.
DEFAULT_MODEL = "small"
#: float16 on GPU, int8 on CPU -- the usual trade, and int8 on CUDA is slower
#: than float16 on this hardware rather than faster.
DEFAULT_COMPUTE = {"cuda": "float16", "cpu": "int8"}


class TranscriberUnavailable(Exception):
    """No model, no runtime, or a load that will not succeed by retrying."""


class WhisperTranscriber:
    """faster-whisper over a whole track. A ``Transcriber``."""

    name = "whisper"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        vad_filter: bool = True,
        model_obj: Any = None,
    ) -> None:
        self.model_name = model
        self.language = language
        self.vad_filter = vad_filter

        if model_obj is not None:
            self.device, self.compute_type, self._model = "given", "given", model_obj
            return

        # Before importing the runtime, not after: CTranslate2 resolves its
        # CUDA libraries lazily at the first encode, and the failure then is a
        # missing-DLL error on a machine where the DLL is present.
        cuda.enable()
        try:
            import torch
            from faster_whisper import WhisperModel
        except ImportError as exc:                       # pragma: no cover
            raise TranscriberUnavailable(
                "faster-whisper is not installed: pip install faster-whisper"
            ) from exc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type or DEFAULT_COMPUTE.get(self.device, "int8")
        try:
            self._model = WhisperModel(model, device=self.device,
                                       compute_type=self.compute_type)
        except Exception as exc:                         # noqa: BLE001
            raise TranscriberUnavailable(
                f"could not load Whisper {model!r} on {self.device} "
                f"({self.compute_type}): {exc}"
            ) from None

    def transcribe(self, track: Track) -> Transcript:
        segments, info = self._model.transcribe(
            track.samples,
            language=self.language,
            word_timestamps=True,
            vad_filter=self.vad_filter,
        )
        out: list[Segment] = []
        for seg in segments:                    # a generator: work happens here
            out.append(Segment(
                start=float(seg.start), end=float(seg.end), text=seg.text,
                words=[Word(float(w.start), float(w.end), w.word,
                            float(w.probability) if w.probability is not None else None)
                       for w in (seg.words or [])],
            ))
        return Transcript(
            language=info.language,
            language_probability=float(info.language_probability),
            duration_s=track.duration_s,
            segments=out,
            model=self.config(),
        )

    def config(self) -> dict[str, Any]:
        return {"name": self.name,
                "params": {"model": self.model_name, "device": self.device,
                           "compute_type": self.compute_type,
                           "language": self.language,
                           "vad_filter": self.vad_filter}}
