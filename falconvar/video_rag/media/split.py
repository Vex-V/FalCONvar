"""Reading what streams a file carries.

Opens the container once, records what each half needs to open its own decoder,
and closes it. No pixels, no waveform, no model.

A description rather than a demux, for three reasons: writing two files doubles
the bytes for data read once, it needs ffmpeg or a second decode pass, and it
renumbers `pts`, which is how a frame is addressed later.

Each half reopens the original because a decoder is a cursor with no rewind and
the two are consumed at incompatible rates -- audio wants the whole waveform,
video wants one frame in flight.

A missing stream is a fact (`has_video`, `has_audio`), not an error; only a
file carrying neither raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import av

from ...shared.contracts.documents import AudioStream, Media, VideoStream


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
