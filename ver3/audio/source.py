"""The waveform, decoded once, whole.

PyAV rather than an ffmpeg subprocess and a temporary wav, for the same reason
the video side uses it: one decoder, one set of timestamp semantics, and no
second binary whose presence has to be checked.

Resampling happens inside the decode loop -- 16 kHz mono float32, which is what
both Whisper and pyannote want -- so there is no second pass over the samples.
Measured: 205 s of AAC decodes and resamples in about 0.3 s, ~700x realtime, so
this is never the slow part.

Memory is the one thing to watch. The whole file is held at once, which is the
price of transcription being a whole-file operation. At 16 kHz mono float32
that is 64 KB per second: an hour is 230 MB, a feature film under a gigabyte.
The video side's one-frame-in-flight discipline exists because 4485 frames at
6 MB does not fit; 16 kHz mono does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import av
import numpy as np

#: What both Whisper and pyannote are trained on. Not a parameter.
SAMPLE_RATE = 16000

#: Below this RMS a track carries no speech worth transcribing. Measured: CCTV
#: with a live but empty microphone sits at RMS 0.000221 and peak 0.0291;
#: narration at 0.1796 -- three orders of magnitude, so the threshold sits in a
#: wide gap rather than on a cliff. It exists to skip a model load and to make
#: "no speech" a reported fact, not to make a fine judgement.
SILENCE_RMS = 1e-3


class NoAudio(Exception):
    """The file carries no audio stream at all."""


@dataclass
class Track:
    """One decoded waveform, and the two numbers that say whether a model
    should be handed it."""

    samples: np.ndarray                      # float32, mono, SAMPLE_RATE
    rate: int
    rms: float
    peak: float

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.rate

    @property
    def silent(self) -> bool:
        return self.rms < SILENCE_RMS

    def as_dict(self) -> dict[str, Any]:
        return {"rate": self.rate, "duration_s": round(self.duration_s, 3),
                "samples": int(len(self.samples)), "rms": round(self.rms, 6),
                "peak": round(self.peak, 6), "silent": self.silent}


def load(path: str, rate: int = SAMPLE_RATE) -> Track:
    """Decode the whole audio stream to mono float32 at ``rate``."""
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise NoAudio(f"{path} has no audio stream")
        # Mandatory rather than an optimisation, exactly as on the video side:
        # 7.15 ms/frame without it against 3.97 with.
        stream.thread_type = "AUTO"
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
        blocks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                blocks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):            # flush
            blocks.append(out.to_ndarray().reshape(-1))

    samples = (np.concatenate(blocks) if blocks
               else np.zeros(0, dtype=np.float32)).astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return Track(samples=samples, rate=rate, rms=rms, peak=peak)


__all__ = ["SAMPLE_RATE", "SILENCE_RMS", "NoAudio", "Track", "load"]
