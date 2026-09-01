"""The waveform, decoded once, whole.

PyAV rather than an ffmpeg subprocess and a temporary wav, for the same reason
the video reader uses it: one decoder, one set of timestamp semantics, and no
second binary whose presence has to be checked. There is no ffmpeg on this
machine's PATH and nothing here needs one.

Resampling happens inside the decode loop -- 16 kHz mono float32, which is what
both Whisper and pyannote want, and asking the decoder for it avoids a second
pass over the samples. Measured on an RTX 4060 machine: 205 s of AAC decodes
and resamples in 0.29 s, about 700x realtime, so this is never the slow part.

Memory is the one thing to keep in view. A whole file is held at once, which is
the price of transcription being a whole-file operation -- but at 16 kHz mono
float32 that is 64 KB per second, so an hour is 230 MB and a feature film is
under a gigabyte. The video reader's one-frame-in-flight discipline exists
because 4485 frames x 6 MB does not fit; 16 kHz mono does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import av
import numpy as np

#: What both Whisper and pyannote are trained on. Not a parameter.
SAMPLE_RATE = 16000

#: Below this RMS a track carries no speech worth transcribing. Measured:
#: test1.mp4 is CCTV with a live but empty microphone at RMS 0.000221 and peak
#: 0.0291; Chernobyl.mp4 is narration at RMS 0.1796 -- three orders of
#: magnitude apart, so the threshold sits in a wide gap rather than on a cliff.
#: It exists to skip a model load, not to make a fine judgement: Whisper
#: returns nothing on that track anyway, and a guard that has to be precise
#: would be the wrong mechanism.
SILENCE_RMS = 1e-3


@dataclass
class AudioInfo:
    """What probing found, before anything was decoded."""

    has_audio: bool
    codec: Optional[str] = None
    rate: Optional[int] = None
    channels: Optional[int] = None
    duration_s: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return {"has_audio": self.has_audio, "codec": self.codec,
                "rate": self.rate, "channels": self.channels,
                "duration_s": round(self.duration_s, 3) if self.duration_s else None}


@dataclass
class Track:
    """One decoded waveform, and the two numbers that say whether it is worth
    handing to a model."""

    samples: np.ndarray                  # float32, mono, SAMPLE_RATE
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


def probe(path: str | Path) -> AudioInfo:
    """What the container says about its audio, without decoding any of it.

    A file with no audio stream is an ordinary outcome, not an error: half the
    footage this project has been built on is silent CCTV. It returns
    ``has_audio=False`` and the caller writes no transcript, rather than
    writing one full of nulls.
    """
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            return AudioInfo(has_audio=False)
        duration = (float(stream.duration * stream.time_base)
                    if stream.duration is not None else None)
        return AudioInfo(
            has_audio=True,
            codec=stream.codec_context.name,
            rate=stream.rate,
            channels=stream.channels,
            duration_s=duration,
        )


def load(path: str | Path, rate: int = SAMPLE_RATE) -> Track:
    """Decode the whole audio stream to mono float32 at ``rate``.

    Raises ``NoAudio`` if the file has no audio stream; check with ``probe``
    first when that is an expected case rather than a surprise.
    """
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise NoAudio(f"{path} has no audio stream")
        # Mandatory rather than an optimisation, exactly as in the video
        # reader: measured at 7.15 ms/frame without it against 3.97 with.
        stream.thread_type = "AUTO"
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
        blocks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                blocks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):          # flush the resampler
            blocks.append(out.to_ndarray().reshape(-1))

    samples = (np.concatenate(blocks) if blocks
               else np.zeros(0, dtype=np.float32)).astype(np.float32, copy=False)
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return Track(samples=samples, rate=rate, rms=rms, peak=peak)


class NoAudio(Exception):
    """The file carries no audio stream at all."""
