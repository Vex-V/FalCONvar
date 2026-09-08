"""faster-whisper over a whole track.

Imported only when `--transcriber whisper` is asked for: this pulls in torch
and CTranslate2, and a stub run should pay for neither.
"""

from __future__ import annotations

from typing import Any, Optional

from . import cuda
from ..models import ModelUnavailable, Segment, Transcript, Word
from ..source import Track

#: Model ids change faster than this file will, so a plain default.
DEFAULT_MODEL = "small"

#: float16 on GPU, int8 on CPU -- and int8 on CUDA is slower than float16 on
#: this hardware rather than faster.
DEFAULT_COMPUTE = {"cuda": "float16", "cpu": "int8"}


class WhisperTranscriber:
    """A `Transcriber`. Whole track in, segments with word timestamps out."""

    name = "whisper"

    def __init__(self, model: str = DEFAULT_MODEL,
                 device: Optional[str] = None,
                 compute_type: Optional[str] = None,
                 language: Optional[str] = None,
                 vad_filter: bool = True,
                 model_obj: Any = None) -> None:
        self.model_name = model
        self.language = language
        self.vad_filter = vad_filter

        if model_obj is not None:
            self.device = self.compute_type = "given"
            self._model = model_obj
            return

        # Before importing the runtime, not after. CTranslate2 asks Windows for
        # its CUDA libraries *by name* at the first encode, not at import, and
        # `nvidia-cublas-cu12` installs them where no search path points --
        # since Python 3.8 an extension's dependencies do not resolve from
        # PATH. The failure is otherwise a missing-DLL error on a machine where
        # the DLL is present.
        cuda.enable()
        try:
            import torch
            from faster_whisper import WhisperModel
        except ImportError as exc:                       # pragma: no cover
            raise ModelUnavailable(
                "faster-whisper is not installed: pip install faster-whisper"
            ) from exc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type or DEFAULT_COMPUTE.get(self.device, "int8")
        try:
            self._model = WhisperModel(model, device=self.device,
                                       compute_type=self.compute_type)
        except Exception as exc:                         # noqa: BLE001
            raise ModelUnavailable(
                f"could not load Whisper {model!r} on {self.device} "
                f"({self.compute_type}): {exc}") from None

    def transcribe(self, track: Track) -> Transcript:
        segments, info = self._model.transcribe(
            track.samples, language=self.language,
            word_timestamps=True, vad_filter=self.vad_filter)
        out: list[Segment] = []
        for seg in segments:                  # a generator: the work is here
            out.append(Segment(
                start=float(seg.start), end=float(seg.end), text=seg.text,
                words=[Word(float(w.start), float(w.end), w.word)
                       for w in (seg.words or [])]))
        return Transcript(segments=out, language=info.language,
                          language_probability=float(info.language_probability))

    def config(self) -> dict[str, Any]:
        return {"transcriber": self.name, "model": self.model_name,
                "device": self.device, "compute_type": self.compute_type,
                "language": self.language, "vad_filter": self.vad_filter}


__all__ = ["DEFAULT_MODEL", "WhisperTranscriber"]
