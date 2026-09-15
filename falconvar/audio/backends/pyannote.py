"""Speaker turns over a whole track, via pyannote.

Imported only when `--diarizer pyannote` is asked for.

**pyannote 4.x returns `DiarizeOutput`, not `Annotation`.** The 3.x recipe
`pipeline(audio).itertracks(yield_label=True)` raises `AttributeError`. The
annotation is `.speaker_diarization`; `.exclusive_speaker_diarization` is the
same with overlaps resolved, and that is what this uses -- a word cannot belong
to two speakers, and chunk boundaries derived from turns must not overlap.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ..models import Diarization, ModelUnavailable, Turn
from ..source import Track

DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
TOKEN_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _token(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    return next((os.environ[v] for v in TOKEN_VARS if os.environ.get(v)), None)


class PyannoteDiarizer:
    """A `Diarizer`. Whole track in, speaker turns out."""

    name = "pyannote"

    def __init__(self, model: str = DEFAULT_MODEL,
                 device: Optional[str] = None,
                 token: Optional[str] = None,
                 exclusive: bool = True,
                 pipeline: Any = None) -> None:
        self.model_name = model
        self.exclusive = exclusive

        if pipeline is not None:
            self.device, self._pipeline = "given", pipeline
            return
        import torch
        from pyannote.audio import Pipeline

        key = _token(token)
        if not key:
            raise ModelUnavailable(
                "no Hugging Face token: set " + " or ".join(TOKEN_VARS) +
                f" and accept the terms for {model} on huggingface.co")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._pipeline = Pipeline.from_pretrained(model, token=key)
            self._pipeline.to(torch.device(self.device))
        except Exception as exc:                         # noqa: BLE001
            raise ModelUnavailable(
                f"could not load {model!r}: {exc}\nThe model is gated -- accept "
                "its terms on huggingface.co with the account this token "
                "belongs to.") from None

    def diarize(self, track: Track) -> Diarization:
        # `reader.listen` already returns early on a silent track, so this is
        # belt and braces for a direct caller. pyannote finds zero speakers in
        # 0.1 s anyway; the point is that "no speech" stays a reported fact.
        if track.silent:
            return Diarization([])

        import torch

        output = self._pipeline({
            "waveform": torch.from_numpy(track.samples).unsqueeze(0),
            "sample_rate": track.rate,
        })
        annotation = (output.exclusive_speaker_diarization if self.exclusive
                      else output.speaker_diarization)
        turns = [Turn(str(label), float(seg.start), float(seg.end))
                 for seg, _, label in annotation.itertracks(yield_label=True)]
        turns.sort(key=lambda t: t.start)
        return Diarization(turns)

    def config(self) -> dict[str, Any]:
        return {"diarizer": self.name, "diarizer_model": self.model_name,
                "device": self.device, "exclusive": self.exclusive}


__all__ = ["DEFAULT_MODEL", "PyannoteDiarizer"]
