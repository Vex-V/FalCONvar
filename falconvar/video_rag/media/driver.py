"""The media component: a file path -> `media.json`.

The only component that takes a path rather than a video id, because it is the
one that establishes the id. Everything after it is addressed by `video_id`
and a backend, and `shared/paths.py` resolves the rest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from ...shared import paths
from ...shared.storage import sinks
from ...shared.contracts.documents import Media, Produced
from .split import UnusableMedia, split


def media(path: str | Path, video_id: Optional[str] = None,
          sink: str | Sequence[str] = "file") -> Produced:
    """Describe the file, write `media.json`, report what was written."""
    described = split(path, video_id)
    written = sinks.write(described.video_id, "media", described.as_dict(), sink)
    return Produced(
        video_id=described.video_id,
        component="media",
        backend=",".join(written),
        artifacts={"media": written.get("file", "")},
        stats={"has_video": described.has_video, "has_audio": described.has_audio,
               "duration_s": described.duration_s,
               "container": described.container_format},
        # What this file does NOT carry. A later component reads this rather
        # than opening the file again to find out.
        skipped=([] if described.has_video else ["video"])
                + ([] if described.has_audio else ["audio"]),
    )


#: The uniform name every component also answers to: what a dispatch
#: table calls and what a form introspects. The same function object.
#: Named for the component, so a traceback frame says which one failed;
#: eight functions called `run` all read the same in a stack.
run = media


def load(video_id: str) -> Media:
    """Read back what `run` wrote. Every later component starts here."""
    return Media.from_dict(sinks.read_json(paths.artifact(video_id, "media")))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Describe the two streams a media file carries.")
    ap.add_argument("media", type=Path)
    ap.add_argument("--video-id", default=None, help="defaults to the filename stem")
    ap.add_argument("--sink", default="file", help="file | supabase | both")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        produced = run(args.media, args.video_id, args.sink)
    except (UnusableMedia, sinks.UnknownBackend) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    m = load(produced.video_id)
    length = f"  {m.duration_s:.3f}s" if m.duration_s else ""
    print(f"{m.path}  [{m.container_format}]{length}")
    if m.video:
        v = m.video
        rate = f"  {v.rate:g} fps" if v.rate else ""
        print(f"  video  {v.codec}  {v.width}x{v.height}{rate}")
        print(f"         time_base={v.time_base}  frames={v.frames}  "
              f"duration={v.duration_s}")
    else:
        print("  video  -- none")
    if m.audio:
        a = m.audio
        print(f"  audio  {a.codec}  {a.rate} Hz  {a.channels} ch")
        print(f"         duration={a.duration_s}")
    else:
        print("  audio  -- none")
    print()
    print(f"media -> {produced.artifacts['media']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
