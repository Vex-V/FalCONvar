"""Command line for the describe stage.

Argument parsing, sink assembly and the run summary. The reader itself is in
reader.py and does not import this.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

if __package__ in (None, ""):                       # allow running as a script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ver2.video.describe import describers as describers_mod
from ver2.video.describe import input as input_mod
from ver2.video.describe.input import StoreUnavailable
from ver2.video.describe.output import DescriptionDocument, MultiDescriptionSink
from ver2.video.describe.vlm import DescriberUnavailable
from ver2.video.describe.reader import Result, describe


def report(result: Result, show: int = 4) -> None:
    stats = result.document.get("stats", {})
    print()
    print(f"  chunks       {result.chunks_seen}")
    print(f"  described    {result.described}")
    if result.skipped:
        print(f"  skipped      {result.skipped}   (already described)")
    print(f"  elapsed      {result.elapsed_s:.2f}s")
    print()
    frames = result.frames
    print(f"  frames       {frames['frames_requested']} requested   "
          f"{frames['cache_hits']} served from cache, "
          f"{frames['frames_read']} read from the store")
    for chunk in result.document.get("chunks", [])[:show]:
        for sampler, block in chunk["samplers"].items():
            text = (block.get("description") or "")[:88]
            print(f"    {chunk['chunk_id']:>3} {sampler:<10} {text}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Read the frames a manifest names and describe them.")
    ap.add_argument("manifest", nargs="?", type=Path, default=None,
                    help="manifest json; omit when using --video-id")
    ap.add_argument("--video-id", default=None,
                    help="fetch the manifest from Supabase instead of a file")
    ap.add_argument("--follow", action="store_true",
                    help="describe chunks as ingest writes them, ending when the "
                         "run reports itself complete (implies --video-id)")
    ap.add_argument("--poll", type=float, default=2.0,
                    help="seconds between polls while following (default 2)")
    ap.add_argument("--describer", default="stub",
                    choices=describers_mod.available(),
                    help="which describer to run (default stub: no model, no "
                         "network, no cost)")
    ap.add_argument("--model", default=None,
                    help="model id for a model-backed describer "
                         "(default gpt-5.4-mini for --describer openai)")
    ap.add_argument("--max-output-tokens", type=int, default=None,
                    help="cap on the description length (default 700)")
    ap.add_argument("--sampler", default=None,
                    help="comma-separated subset of samplers to describe")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many descriptions")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="document path (default out/<video-id>/descriptions.json)")
    ap.add_argument("--sink", default="file",
                    help="comma-separated destinations: file, supabase "
                         "(default file; `file,supabase` writes both)")
    args = ap.parse_args()

    if not args.manifest and not args.video_id:
        ap.error("give a manifest path or --video-id")
    if args.follow and not args.video_id:
        ap.error("--follow needs --video-id: it reads the chunk stream from Supabase")

    names = [n.strip() for n in args.sink.split(",") if n.strip()]
    unknown = [n for n in names if n not in ("file", "supabase")]
    if unknown or not names:
        ap.error(f"--sink: unknown destination {unknown or [args.sink]}, "
                 "expected file and/or supabase")
    # Unconditionally: Supabase, the model key, or neither -- a CLI reading the
    # project's .env is expected either way, and guessing which flags need it
    # is how --describer openai came to not see its own key.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    # The manifest first: everything else is named after what it says.
    feed = None
    if args.manifest:
        manifest = input_mod.from_file(args.manifest)
    else:
        client = input_mod.client_from_env()
        if args.follow:
            manifest = input_mod.header(client, args.video_id)
            feed = input_mod.follow_chunks(client, args.video_id, poll_s=args.poll)
        else:
            manifest = input_mod.from_supabase(client, args.video_id)

    video_id = manifest["video_id"]
    if args.out is None:
        # Beside the manifest and the store it read, under the video's own
        # directory: out/<video-id>/descriptions.json.
        args.out = Path("out") / video_id / "descriptions.json"

    built = []
    for n in names:
        if n == "file":
            built.append(DescriptionDocument(args.out))
        else:
            from ver2.video.describe.output import SupabaseDescriptions

            built.append(SupabaseDescriptions())
    sink = built[0] if len(built) == 1 else MultiDescriptionSink(*built)

    print(f"{manifest['source']['uri']}  ({video_id})", flush=True)
    if args.follow:
        print("  following the chunk stream; ends when ingest reports complete")

    options: dict[str, Any] = {}
    if args.model is not None:
        options["model"] = args.model
    if args.max_output_tokens is not None:
        options["max_output_tokens"] = args.max_output_tokens

    try:
        describer = describers_mod.build(args.describer, **options)
    except (DescriberUnavailable, TypeError) as exc:
        print(exc, file=sys.stderr)
        return 2

    try:
        result = describe(
            manifest,
            describer=describer,
            sink=sink,
            samplers=[s.strip() for s in args.sampler.split(",")] if args.sampler else None,
            feed=feed,
            limit=args.limit,
        )
    except (StoreUnavailable, DescriberUnavailable) as exc:
        print("", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    report(result)
    where = [str(args.out) if n == "file" else
             f"supabase: descriptions where video_id = {video_id}"
             for n in names]
    print("\ndescriptions -> " + ("\n                ").join(where))
    if not result.document.get("complete", False):
        print("  note: complete = false -- some (chunk, sampler) pairs have no "
              "description yet", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
