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

from .base import ManifestSink
from .manifest import MANIFEST_VERSION, FileManifestWriter
from .multi import MultiSink
from .store import FrameStore

# Imported lazily by name: supabase-py is optional, and a file run must not
# pay for it or fail without it.
def __getattr__(name: str):
    if name == "SupabaseManifestWriter":
        from .supabase_manifest import SupabaseManifestWriter
        return SupabaseManifestWriter
    raise AttributeError(name)

__all__ = [
    "MANIFEST_VERSION",
    "FileManifestWriter",
    "FrameStore",
    "ManifestSink",
    "MultiSink",
    "SupabaseManifestWriter",
]
