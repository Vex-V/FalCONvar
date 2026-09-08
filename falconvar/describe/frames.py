"""Reading pixels back out of the frame store.

**The store and nothing else.** There is no seek-the-video fallback, and that
is deliberate: the store exists so this component has its frames in hand, and a
fallback would quietly do the store's job while leaving it broken -- silently,
since the output is identical and only about 40x slower. A missing store, or a
frame the store lacks, raises and names the fix.

**A short frame list is never returned.** A description covering 8 of the 9
frames it claims is indistinguishable from a correct one once written down.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..shared import paths
from ..shared.documents import Manifest


class StoreUnavailable(RuntimeError):
    """The frame store is missing, or lacks a frame the manifest names."""


@dataclass
class LoadedFrame:
    """One frame, as bytes ready to send."""

    index: int
    media_ts: float
    jpeg: bytes

    @property
    def size(self) -> int:
        return len(self.jpeg)


class FrameSource:
    """Frames for one (chunk, sampler), read from the store.

    Caches by index, because a frame two samplers chose is *described* twice --
    they are different questions -- but only ever read once.
    """

    def __init__(self, video_id: str, manifest: Manifest) -> None:
        self.video_id = video_id
        self.manifest = manifest
        self.root = paths.artifact(video_id, "store")
        self._cache: dict[int, bytes] = {}
        if not self.root.exists():
            raise StoreUnavailable(
                f"no frame store at {self.root}. Re-run ingest with a frame "
                f"store, or rebuild it from the manifest and the video.")

    def _read(self, index: int) -> bytes:
        if index in self._cache:
            return self._cache[index]
        path = self.root / f"{index:07d}.jpg"
        if not path.exists():
            raise StoreUnavailable(
                f"{path} is missing, but the manifest names frame {index}. "
                f"The store and the manifest disagree.")
        data = path.read_bytes()
        self._cache[index] = data
        return data

    def images_for(self, chunk_id: int, sampler_id: str) -> list[LoadedFrame]:
        records = self.manifest.frames_of(chunk_id, sampler_id)
        if not records:
            return []
        loaded = [LoadedFrame(r["index"], r["media_ts"], self._read(r["index"]))
                  for r in records]
        if len(loaded) != len(records):        # unreachable; the read raises
            raise StoreUnavailable(
                f"chunk {chunk_id}/{sampler_id}: {len(loaded)} of "
                f"{len(records)} frames available")
        return loaded

    def close(self) -> None:
        self._cache.clear()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = ["FrameSource", "LoadedFrame", "StoreUnavailable"]
