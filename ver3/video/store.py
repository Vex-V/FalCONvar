"""The frames themselves, on disk, addressed by read index.

Describing reads this and nothing else. There is deliberately no
seek-the-video fallback: the store exists so that stage has its frames in
hand, and a fallback would quietly do the store's job while leaving it broken
-- silently, since the output is identical and only ~40x slower.

Named by `index`, the reader's count over every frame, because that is the one
address that survives the manifest being copied somewhere else. `pts` is the
exact address *within* a container; the index is what a filename can hold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


class FrameStore:
    """Writes one JPEG per kept frame. Nothing else touches this directory."""

    def __init__(self, root: Path, quality: int = 95,
                 fmt: str = "jpg") -> None:
        self.root = Path(root)
        self.quality = quality
        self.format = fmt
        self.written = 0
        self.bytes_written = 0

    def path_for(self, index: int) -> Path:
        return self.root / f"{index:07d}.{self.format}"

    def write(self, index: int, image: Optional[np.ndarray]) -> Optional[Path]:
        if image is None:
            return None
        import cv2

        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(index)
        params = ([int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                  if self.format in ("jpg", "jpeg") else [])
        if not cv2.imwrite(str(path), image, params):
            raise OSError(f"could not write {path}")
        self.written += 1
        self.bytes_written += path.stat().st_size
        return path

    def prune(self, keep: set[int]) -> list[int]:
        """Delete stored frames no longer named, and say which.

        A store accumulates across runs: ingesting with `uniform` and then with
        `clip` leaves the first run's frames behind, because nothing tells the
        directory that a new manifest supersedes the old one. That is disk
        wasted on frames no manifest addresses, and it makes a byte-comparison
        against the store report orphans that are not mismatches.

        Not automatic. Deleting frames is the one irreversible thing this
        component can do, and a caller who is about to describe from an older
        manifest wants them kept -- so it happens when asked.
        """
        if not self.root.exists():
            return []
        removed = []
        for path in sorted(self.root.glob(f"*.{self.format}")):
            try:
                index = int(path.stem)
            except ValueError:
                continue                     # not ours; leave it alone
            if index not in keep:
                path.unlink()
                removed.append(index)
        return removed

    def config(self) -> dict[str, Any]:
        return {"root": str(self.root), "format": self.format,
                "quality": self.quality}


__all__ = ["FrameStore"]
