"""What an ingest run leaves behind.

Two artifacts with different standing. The manifest is the record: it says
which frames were chosen, why, and how to address them again. The frame store
is a cache of pixels the run already had in hand -- faster to read back than
seeking the video, and the only option at all on a live source, but always
reconstructible from the manifest.

Ingest owns the format. recovery/ deliberately does not import from here --
it re-derives what it needs from the manifest, so a manifest and a video are
enough to rebuild a store without this package present at all.
"""

from .manifest import MANIFEST_VERSION, ManifestWriter
from .store import FrameStore

__all__ = ["MANIFEST_VERSION", "FrameStore", "ManifestWriter"]
