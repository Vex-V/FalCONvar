"""The frame store: sampled pixels written at ingest, keyed by frame index.

Ingest already visits every frame in order and already holds the pixels, so
writing them costs ~4.8 ms each against ~167 ms to seek one back out later.
The store is what makes the VLM handoff fast, and on a live source it is the
only thing that works at all -- a camera feed cannot be seeked back into.

Keyed by *frame index*, not by sampler. Samplers overlap heavily: measured on
a three-sampler run, 138 picks covered 93 distinct frames, so keying per
sampler would write a third of the store twice.

The manifest remains authoritative. Everything here is reconstructible from it
plus the source video, which is why the store can be deleted, pruned, or
rewritten at a different resolution without losing anything.

What this is *not* is a cache for retuning thresholds. Re-running a sampler
over stored JPEGs does not reproduce a production run: q85 at 1920 px perturbs
pixels by 1.6/255, three times the 0.5/255 decoder difference that already
moves the detection samplers by 12-15%. Sweeping a threshold wants a cache of
what the model *produced* -- CLIP embeddings are 2 KB a frame against 320 KB
of JPEG, and exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class FrameStore:
    """A directory of encoded frames, addressed by source frame index."""

    def __init__(
        self,
        directory: str | Path,
        max_width: Optional[int] = 1920,
        quality: int = 85,
        suffix: str = ".jpg",
    ) -> None:
        # The directory itself, not a root to append a video id to. The
        # manifest records this path verbatim, and a reader that has the
        # manifest should be able to hand it straight back.
        self.dir = Path(directory)
        # Sized for the most demanding consumer rather than the average one:
        # at 1024 px a VLM misread a burnt-in clock as 11:17:40 when it said
        # 11:17:19, and read it correctly at 1920. Downstream can downscale;
        # it cannot invent detail back.
        self.max_width = max_width
        self.quality = quality
        self.suffix = suffix
        self.written = 0
        self.bytes_written = 0

    def path_for(self, index: int) -> Path:
        return self.dir / f"{index:07d}{self.suffix}"

    def write(self, index: int, image: np.ndarray, overwrite: bool = False) -> Optional[Path]:
        """Encode and store one frame. Returns None if it was already there."""
        path = self.path_for(index)
        if path.exists() and not overwrite:
            return None
        if self.max_width and image.shape[1] > self.max_width:
            h = int(round(image.shape[0] * self.max_width / image.shape[1]))
            image = cv2.resize(image, (self.max_width, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(self.suffix, image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise RuntimeError(f"failed to encode frame {index}")
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buf)
        self.written += 1
        self.bytes_written += len(buf)
        return path

    def read(self, index: int) -> Optional[np.ndarray]:
        path = self.path_for(index)
        if not path.exists():
            return None
        return cv2.imread(str(path))

    def read_bytes(self, index: int) -> Optional[bytes]:
        """The encoded bytes, for a consumer that wants to forward them as-is.

        A describer base64s JPEG anyway, so reading these avoids a decode and
        a re-encode that would otherwise both happen at the API boundary.
        """
        path = self.path_for(index)
        return path.read_bytes() if path.exists() else None

    def config(self) -> dict:
        """Recorded in the manifest so a reader knows what it is getting."""
        return {
            "dir": str(self.dir).replace("\\", "/"),
            "format": self.suffix.lstrip("."),
            "max_width": self.max_width,
            "quality": self.quality,
            "key": "frame_index",
        }
