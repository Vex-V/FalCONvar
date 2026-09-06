"""pyannote.audio, over the whole track.

The pipeline is fetched from the Hub and gated: it needs `HF_TOKEN` in the
environment and the model's terms accepted on the hub page. That is a
first-run cost, not a per-run one.

**pyannote 4.x returns a `DiarizeOutput`, not an `Annotation`.** The 3.x
recipe -- `pipeline(audio).itertracks(yield_label=True)` -- raises
`AttributeError` here; the annotation is `.speaker_diarization`, and there is
also an `.exclusive_speaker_diarization` with overlaps resolved and a
`.speaker_embeddings` array. The last is the only route to identity across
files: labels are per-run clusterings, so `SPEAKER_00` in two videos are
unrelated strings, while their 256-dimensional embeddings are comparable.

Measured on an RTX 4060: 205 s of single-narrator audio diarized in 6.6 s,
about 31x realtime, finding one speaker over 18 turns covering 89% of the
duration. On silent CCTV it returned zero speakers in 0.1 s, which is the
right answer rather than a failure to produce one.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from ..source import Track
from .base import Diarization, Turn

DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
TOKEN_VARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")


class DiarizerUnavailable(Exception):
    """No token, no package, or a gated model this account cannot fetch."""


def _token(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    return next((os.environ[v] for v in TOKEN_VARS if os.environ.get(v)), None)


class PyannoteDiarizer:
    """Speaker turns over a whole track. A ``Diarizer``."""

    name = "pyannote"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        token: Optional[str] = None,
        exclusive: bool = True,
        pipeline: Any = None,
    ) -> None:
        self.model_name = model
        # Overlapped speech assigned to one speaker rather than several. Chunk
        # boundaries derived from turns must not overlap, and a transcript word
        # cannot belong to two speakers at once, so the exclusive reading is
        # what the rest of the pipeline can actually consume.
        self.exclusive = exclusive

        if pipeline is not None:
            self.device, self._pipeline = "given", pipeline
            return
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:                       # pragma: no cover
            raise DiarizerUnavailable(
                "pyannote.audio is not installed: pip install pyannote.audio"
            ) from exc

        key = _token(token)
        if not key:
            raise DiarizerUnavailable(
                "no Hugging Face token: set " + " or ".join(TOKEN_VARS) +
                f" and accept the terms for {model} on huggingface.co"
            )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._pipeline = Pipeline.from_pretrained(model, token=key)
            self._pipeline.to(torch.device(self.device))
        except Exception as exc:                         # noqa: BLE001
            raise DiarizerUnavailable(
                f"could not load {model!r}: {exc}\nThe model is gated -- accept "
                "its terms on huggingface.co with the account this token belongs to."
            ) from None

    def diarize(self, track: Track) -> Diarization:
        # Silence is answered without loading anything onto the GPU. pyannote
        # gets there on its own in 0.1 s, but saying so explicitly keeps "no
        # speech" a reported fact rather than an empty result to interpret.
        if track.silent:
            return Diarization(turns=[], embeddings=None, model=self.config())

        import torch

        output = self._pipeline({
            "waveform": torch.from_numpy(track.samples).unsqueeze(0),
            "sample_rate": track.rate,
        })
        annotation = (output.exclusive_speaker_diarization if self.exclusive
                      else output.speaker_diarization)
        turns = [Turn(float(seg.start), float(seg.end), str(label))
                 for seg, _, label in annotation.itertracks(yield_label=True)]
        turns.sort(key=lambda t: t.start)
        return Diarization(turns=turns,
                           embeddings=getattr(output, "speaker_embeddings", None),
                           model=self.config())

    def config(self) -> dict[str, Any]:
        return {"name": self.name,
                "params": {"model": self.model_name, "device": self.device,
                           "exclusive": self.exclusive}}
