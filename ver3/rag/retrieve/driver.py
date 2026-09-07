"""The retrieve component: a query -> ranked moments. Writes nothing.

**The query is embedded with the embedder that built the index.** Not a
default, an argument. A mismatch across widths fails loudly; a mismatch between
two models of the same width returns a well-formed ranking that means nothing,
which is why the embedder key is in the index filename -- a wrong name searches
a file that does not exist rather than the wrong vectors.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ...boundaries import load as load_timeline
from ...shared import paths
from ..embed import embedders as embedders_mod
from ..embed.driver import DEFAULT_EMBEDDER
from ..embed.index import LocalIndex
from .search import Moment, to_moments


def search(query: str, video_id: str, embedder: str = DEFAULT_EMBEDDER,
           model: Optional[str] = None, moments: int = 5,
           sampler: Optional[str] = None,
           candidates: int = 20) -> list[Moment]:
    built = embedders_mod.build(embedder, **({"model": model} if model else {}))
    index = LocalIndex(video_id, built.key)
    if not index.units:
        raise FileNotFoundError(
            f"{video_id}: no index for {built.key}. Run embed with this "
            f"embedder first -- a different one writes a different file.")
    vector = built.embed([query])[0]
    hits = index.search(vector, query, candidates, sampler)
    timeline = load_timeline(video_id)
    return to_moments(hits, video_id, timeline.spans, moments)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Search one video's moments.")
    ap.add_argument("query")
    ap.add_argument("video_id")
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER,
                    choices=embedders_mod.available())
    ap.add_argument("--model", default=None)
    ap.add_argument("--moments", type=int, default=5)
    ap.add_argument("--sampler", default=None,
                    help="narrow to one question's answers. Note this gives up "
                         "the agreement signal: a chunk can then contribute at "
                         "most one term, so scores halve")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        found = search(args.query, args.video_id, args.embedder, args.model,
                       args.moments, args.sampler)
    except (KeyError, ValueError, FileNotFoundError,
            embedders_mod.EmbedderUnavailable) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps([m.as_dict() for m in found], indent=2))
        return 0

    print(f"{args.query!r} in {args.video_id}")
    for moment in found:
        marks = " ".join(
            f"{h['sampler_id']}(v{h['dense_rank']}"
            + (f",t{h['text_rank']}" if h["text_rank"] else ",tNone") + ")"
            for h in moment.hits)
        print(f"\n  chunk {moment.chunk_id}  "
              f"{moment.start_ts:.1f}-{moment.end_ts:.1f}s   "
              f"score {moment.score:.4f}")
        print(f"    {marks}")
        for hit in moment.hits[:2]:
            text = hit["content"].replace("\n", " ")
            print(f"    {hit['sampler_id']}: {text[:110]}"
                  + ("..." if len(text) > 110 else ""))
    if not found:
        print("  no matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
