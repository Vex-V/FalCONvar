"""Where the work comes from: a manifest, a chunk stream, and pixels.

Three inputs and no more. `describe` reads a **manifest** to learn which
frames matter, reads those frames out of the **frame store**, and -- when
following a live run -- reads **chunk rows** as ingest writes them. It reaches
for nothing else: not the video, not the pipeline, not any ingest internal
beyond the store class itself. What it cannot get from those three, it fails
for, rather than reconstructing behind your back.
"""

from .follow import client_from_env, follow_chunks
from .frames import FrameSource, LoadedFrame, StoreUnavailable
from .manifest import from_file, from_supabase, header

__all__ = [
    "FrameSource",
    "LoadedFrame",
    "StoreUnavailable",
    "client_from_env",
    "follow_chunks",
    "from_file",
    "from_supabase",
    "header",
]
