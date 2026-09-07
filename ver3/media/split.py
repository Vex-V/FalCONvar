"""The split itself. `driver.py` is what writes it down.

This is the fork, and it is *only* the fork. It opens a media file, works out
which of the two streams it carries, and records what each half needs to open
its own decoder. It decodes no pixels, loads no waveform, runs no model and
calls no component.

**Why a description rather than a demux.** The obvious reading of "split" is
two files on disk -- a video-only track and a wav. That is wrong here for three
reasons: it doubles the bytes for data read once; it needs an ffmpeg binary on
PATH, or a second decode pass, to produce files a decoder is about to read
straight back; and it throws away the container's timing, since `pts` and
`time_base` are how a frame is addressed later and a re-muxed file renumbers
them. So the split is a *statement about* the file, and each half opens the
original.

**Why each half opens its own container.** A decoder is a cursor with no
rewind, and the two halves are consumed at incompatible rates. Audio wants the
whole waveform at once, because transcription carries context across an
utterance and speaker identity is a clustering over the entire recording. Video
wants one frame in flight, because a five-minute 1080p file is 4485 frames at
~6 MB. Sharing one container would force one half to buffer for the other.

**Why absence is not an error.** Half the footage this was built on is silent
CCTV, and an audio-only run is a supported thing to ask for. A missing stream
is a fact -- `has_video`, `has_audio` -- and only a file carrying neither
raises. What to do about a missing half is the caller's policy.

**What is deliberately not here.** Whether the frame rate is plausible, whether
the timestamps are usable, what rotation to apply, whether the audio is silent:
each is a judgement about *decoded* data, and making it here would mean
decoding the file this component promises not to decode. It reports what the
container claims and stops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import av

from ..shared.documents import AudioStream, Media, VideoStream


class UnusableMedia(RuntimeError):
    """The file cannot be opened, or carries neither a video nor an audio stream."""


def _seconds(value: Optional[int], time_base) -> Optional[float]:
    """A stream duration in seconds, or None when the container does not say.

    None rather than 0.0: "unknown length" and "zero length" are different
    facts, and a caller that cannot tell them apart will divide by one of them.
    """
    if value is None or time_base is None:
        return None
    return float(value * time_base)


def split(path: str | Path, video_id: Optional[str] = None) -> Media:
    """Open ``path`` once and describe the two streams it carries.

    The container is opened, read for metadata and closed before returning, so
    the result holds no decoder and is safe to keep, serialise or pass around.
    """
    path = Path(path)
    if not path.exists():
        raise UnusableMedia(f"{path} does not exist")

    try:
        container = av.open(str(path))
    except Exception as exc:                                # noqa: BLE001
        raise UnusableMedia(
            f"cannot open {path} -- unsupported, missing or corrupt "
            f"({type(exc).__name__})") from None

    try:
        # The container's own duration. Carried separately rather than taken
        # from whichever stream reports one: the streams routinely disagree by
        # milliseconds, and a file is as long as its longest.
        duration = (container.duration / av.time_base
                    if container.duration is not None else None)

        v = next(iter(container.streams.video), None)
        a = next(iter(container.streams.audio), None)

        video = None if v is None else VideoStream(
            index=v.index,
            codec=v.codec_context.name,
            # `guessed_rate` before `average_rate`: containers lie about the
            # average, and the guess is derived from the timestamps themselves.
            # An H.264-in-AVI reporting 600 fps guesses 15, correctly.
            rate=float(v.guessed_rate or v.average_rate or 0) or None,
            # As a string, so the exact rational survives JSON.
            time_base=str(v.time_base) if v.time_base else None,
            width=v.codec_context.width,
            height=v.codec_context.height,
            # 0 means "the container did not count", not "no frames".
            frames=v.frames or None,
            duration_s=_seconds(v.duration, v.time_base),
        )

        audio = None if a is None else AudioStream(
            index=a.index,
            codec=a.codec_context.name,
            rate=a.rate,
            channels=a.channels,
            duration_s=_seconds(a.duration, a.time_base),
        )
        container_format = container.format.name
    finally:
        container.close()

    if video is None and audio is None:
        raise UnusableMedia(f"{path} carries neither a video nor an audio stream")

    return Media(
        video_id=video_id or path.stem,
        path=str(path),
        container_format=container_format,
        duration_s=duration,
        video=video,
        audio=audio,
    )


__all__ = ["UnusableMedia", "split"]
