"""Manifest in, pixels out. From the frame store, and only from the store.

Describing reads frames the store already holds. It does **not** fall back to
seeking the video, and that is a deliberate refusal rather than a missing
feature: the store exists precisely so this stage has its frames in hand, and a
fallback would quietly do the store's job while leaving it broken. The two
paths differ by about 40x -- 0.12 s against 4.97 s on the reference video -- so
a silent fallback also turns a missing directory into a mysterious slowdown.

If the store is absent or incomplete, that is a fact worth stopping for. The
frames are reconstructible byte-for-byte from the manifest and the video, and
there is a tool whose whole job is doing that:

    python -m ver2.recovery.recreate <manifest> --out <store>

Rebuilding is explicit, verifiable and reported. Guessing is none of those.

Frames are handed on exactly as stored. ``FrameStore.read_bytes`` returns the
JPEG untouched, and a describer base64s JPEG at the API boundary anyway, so
nothing here decodes or re-encodes. Nothing here downscales either: the store
was sized for the most demanding consumer, because at 1024 px a VLM misread a
burnt-in clock as 11:17:40 when it read 11:17:19, and read it correctly at
1920. Downstream can shrink an image; it cannot invent detail back.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ver2.video.ingest.output import FrameStore

REBUILD_HINT = ("Rebuild it from the manifest and the video:\n"
                "    python -m ver2.recovery.recreate <manifest> --out <store>")


class StoreUnavailable(Exception):
    """The frames this manifest names are not on this machine."""


@dataclass
class LoadedFrame:
    """One frame, ready to hand to a model."""

    index: int
    media_ts: float
    pts: Optional[int]
    score: Optional[float]
    jpeg: bytes

    @property
    def image(self) -> np.ndarray:
        """Decoded pixels, for a describer that wants an array rather than bytes."""
        return cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR)


@dataclass
class SourceStats:
    requested: int = 0
    cache_hits: int = 0
    read: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames_requested": self.requested,
            "cache_hits": self.cache_hits,
            "frames_read": self.read,
            "load_seconds": round(self.seconds, 3),
        }


class FrameSource:
    """Loads the frames a manifest points at, out of the store it names.

    The cache is per chunk and keyed by frame index, because samplers overlap:
    on the reference run 25 of 80 frames were chosen by two samplers. Both
    sampler calls should see those pixels; only one of them should pay to read
    them. ``release()`` at the chunk boundary keeps peak memory at one chunk's
    worth rather than one video's.
    """

    def __init__(self, manifest: dict[str, Any]) -> None:
        self.manifest = manifest
        self.stats = SourceStats()
        self._cache: dict[int, LoadedFrame] = {}

        config = (manifest.get("config") or {}).get("frame_store")
        if not config:
            raise StoreUnavailable(
                f"manifest for {manifest['video_id']!r} names no frame store: it "
                "was ingested without --frame-store, so no pixels were kept.\n"
                + REBUILD_HINT)

        self.dir = Path(config["dir"])
        if not self.dir.exists():
            raise StoreUnavailable(
                f"frame store {self.dir} does not exist on this machine.\n"
                + REBUILD_HINT)

        self.store = FrameStore(
            self.dir,
            quality=config.get("quality", 85),
            suffix="." + config.get("format", "jpg"),
        )
        self.scope = config.get("scope", "sampled")

    def _load(self, record: dict[str, Any]) -> LoadedFrame:
        index = record["index"]
        self.stats.requested += 1
        if index in self._cache:
            self.stats.cache_hits += 1
            return self._cache[index]

        started = time.perf_counter()
        data = self.store.read_bytes(index)
        self.stats.seconds += time.perf_counter() - started
        if data is None:
            # Describing 8 of the 9 frames a chunk names would look entirely
            # normal in the output -- right shape, plausible count, no null --
            # so a gap has to stop the run rather than shrink a description.
            raise StoreUnavailable(
                f"frame {index} is named by the manifest but missing from "
                f"{self.dir}. The store is incomplete.\n" + REBUILD_HINT)

        self.stats.read += 1
        frame = LoadedFrame(
            index=index,
            media_ts=record["media_ts"],
            pts=record.get("pts"),
            score=record.get("score"),
            jpeg=data,
        )
        self._cache[index] = frame
        return frame

    def images_for(self, chunk: dict[str, Any], sampler: str) -> list[LoadedFrame]:
        """Every frame this sampler kept in this chunk, in manifest order.

        All of them or none: a short list is never returned, because a
        description covering fewer frames than it claims is indistinguishable
        from a correct one once it is written down.
        """
        block = chunk["samplers"][sampler]
        return [self._load(record) for record in block["frames"]]

    def release(self) -> None:
        """Drop the chunk's pixels. Call at every chunk boundary."""
        self._cache.clear()

    def close(self) -> None:
        self.release()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
