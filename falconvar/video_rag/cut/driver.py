"""The cut component: `transcript.raw.json` + `timeline.json` -> `transcript.json`."""

from __future__ import annotations

from typing import Optional, Sequence

from ..boundaries import load as load_timeline
from ..audio import load as load_raw
from ..shared import paths
from ..shared.storage import sinks
from ..shared.contracts.documents import Produced, Transcript
from .cutter import stats_for, to_chunks


def run(video_id: str, sink: str | Sequence[str] = "file") -> Produced:
    """Apply the grid. Costs no model and can be repeated at will."""
    raw = load_raw(video_id)
    timeline = load_timeline(video_id)
    chunks = to_chunks(raw, timeline)
    stats = stats_for(raw, chunks)

    transcript = Transcript(
        video_id=video_id,
        timeline_fingerprint=timeline.fingerprint(),
        model=raw.model,
        chunks=chunks,
        stats=stats,
    )
    written = sinks.write(video_id, "transcript", transcript.as_dict(), sink)
    return Produced(
        video_id=video_id, component="cut", backend=",".join(written),
        artifacts={"transcript": written.get("file", "")},
        stats={**stats, "timeline_fingerprint": transcript.timeline_fingerprint},
    )


def load(video_id: str) -> Transcript:
    return Transcript.from_dict(
        sinks.read_json(paths.artifact(video_id, "transcript")))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Cut a transcript to the grid.")
    ap.add_argument("video_id")
    ap.add_argument("--sink", default="file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        produced = run(args.video_id, args.sink)
    except (FileNotFoundError, ValueError, sinks.UnknownBackend) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    s = produced.stats
    print(f"{produced.video_id}")
    print(f"  chunks       {s['chunks']}   ({s['chunks_with_speech']} with speech)")
    print(f"  words        {s['words']}/{s['words_in_transcript']} placed")
    if s["words_outside_grid"]:
        print(f"  outside grid {s['words_outside_grid']}")
    print(f"  speakers     {s['speakers']}")
    print(f"\ntranscript -> {produced.artifacts['transcript']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
