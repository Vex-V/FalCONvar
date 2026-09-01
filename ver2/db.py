"""The Supabase connection, and the reads more than one stage needs.

Owned by no stage, imported by three. Before this existed, the same twenty
lines of "read the url and key, complain usefully if they are missing, build a
client" appeared in four places -- the manifest sink, the description sink, the
describe stage's chunk follower and the pgvector index -- and `retrieve` had to
import from `describe` to borrow one of them. That is an import between two
modules that have nothing to say to each other, created entirely by where a
helper happened to live.

Two keys, two jobs, and the distinction is the reason this is not one function:

  **secret**       bypasses row-level security. Writes. Never leaves the
                   machine that ingests or describes.
  **publishable**  read-only under RLS. What a consumer holds, and what the
                   recovery kit is handed.

`recovery/` deliberately does not import this. It re-implements the same three
requests with `urllib`, because its whole claim is that a manifest, a video and
three files are enough -- and a shared helper, however small, would make that
claim false.
"""

from __future__ import annotations

import os
from typing import Any, Optional

SECRET_VARS = ("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_KEY")
PUBLISHABLE_VARS = ("SUPABASE_PUBLISHABLE_KEY", "SUPABASE_ANON_KEY")


def load_env() -> None:
    """Load a .env if python-dotenv is installed. A convenience, never required."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _first(names: tuple[str, ...]) -> Optional[str]:
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


def client_from_env(url: Optional[str] = None, key: Optional[str] = None,
                    write: bool = True) -> Any:
    """A Supabase client. ``write=False`` uses the read-only publishable key."""
    url = url or os.environ.get("SUPABASE_URL")
    names = SECRET_VARS if write else PUBLISHABLE_VARS
    key = key or _first(names)
    if not url or not key:
        need = "SUPABASE_URL and " + " or ".join(names)
        raise RuntimeError(
            f"{need} must be set. Writes need the secret key (it bypasses "
            "row-level security); reads should use the publishable one."
        )
    from supabase import create_client

    return create_client(url, key)


def fetch_manifest(client: Any, video_id: str) -> dict[str, Any]:
    """The whole manifest, reassembled server-side by ``export_manifest``.

    Server-side so there is no second implementation of the manifest format to
    drift from the one the file writer produces.
    """
    document = client.rpc("export_video_manifest", {"p_video_id": video_id}).execute().data
    if not document:
        raise SystemExit(f"no manifest in Supabase for video_id {video_id!r}")
    return document


def fetch_descriptions(client: Any, video_id: str) -> list[dict[str, Any]]:
    """Every description row for one video, in (chunk, sampler) order."""
    return client.rpc("export_video_descriptions",
                      {"p_video_id": video_id}).execute().data or []


def fetch_transcript(client: Any, video_id: str) -> Optional[dict[str, Any]]:
    """The transcript document, reassembled by ``export_audio_transcript``.

    None when the video has no transcript, which is an ordinary outcome: half
    the footage this project runs on is silent CCTV.
    """
    document = client.rpc("export_audio_transcript",
                          {"p_video_id": video_id}).execute().data
    return document or None


def manifest_header(client: Any, video_id: str) -> dict[str, Any]:
    """The manifest without its chunks -- all a follower has at the start.

    Ingest claims its ``video_id`` before decoding a frame, so everything about
    a run except its results is available while the run is still going.
    """
    rows = (client.table("video_manifests")
            .select("video_id,complete,manifest_version,source,config,stats")
            .eq("video_id", video_id).execute().data)
    if not rows:
        raise SystemExit(
            f"no video_id {video_id!r} in Supabase yet. A follower has to start "
            "after ingest has claimed the id, which it does before decoding."
        )
    row = rows[0]
    return {
        "manifest_version": row["manifest_version"],
        "video_id": row["video_id"],
        "complete": row["complete"],
        "source": row["source"],
        "config": row["config"],
        "stats": row["stats"],
        "chunks": [],
    }
