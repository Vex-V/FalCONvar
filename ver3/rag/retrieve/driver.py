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
from ..embed.driver import DEFAULT_EMBEDDER, DEFAULT_INDEX
from ..embed import indexes as backends
from .search import Moment, to_moments


def search(query: str, video_id: str, embedder: str = DEFAULT_EMBEDDER,
           model: Optional[str] = None, moments: int = 5,
           sampler: Optional[str] = None, index_name: str = DEFAULT_INDEX,
           candidates: int = 20) -> tuple[list[Moment], list[str]]:
    """Ranked moments, and anything the caller should be told about the ranking.

    The notes are not decoration. A dense-only backend returns hits that look
    exactly like fused ones minus a `t` marker, which reads as "no lexical
    match for this query" rather than "this index cannot have one".
    """
    built = embedders_mod.build(embedder, **({"model": model} if model else {}))
    index = backends.build(index_name, video_id, built.key)

    notes: list[str] = []
    if not backends.has_lexical(index_name):
        notes.append(f"{index_name} is dense-only: this ranking has no lexical "
                     "half, which is worth 0.429 against 0.714 top-1 on the "
                     "reference corpus")

    vector = built.embed([query])[0]
    hits = index.search(vector, query, candidates, sampler)
    if getattr(index, "degraded", None):
        notes.append(index.degraded)
    if not hits:
        raise FileNotFoundError(
            f"{video_id}: nothing indexed for {built.key} in {index_name!r}. "
            f"Run embed with this embedder and index first -- a different "
            f"embedder writes a different collection.")
    if sampler is not None:
        notes.append("filtering to one sampler gives up the agreement signal: "
                     "a chunk can then contribute at most one term, so scores "
                     "roughly halve")
    timeline = load_timeline(video_id)
    return to_moments(hits, video_id, timeline.spans, moments), notes


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
                    help="narrow to one question's answers")
    ap.add_argument("--index", default=DEFAULT_INDEX, dest="index_name",
                    choices=backends.available())
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        found, notes = search(args.query, args.video_id, args.embedder,
                              args.model, args.moments, args.sampler,
                              args.index_name)
    except (KeyError, ValueError, FileNotFoundError,
            embedders_mod.EmbedderUnavailable) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps([m.as_dict() for m in found], indent=2))
        return 0

    print(f"{args.query!r} in {args.video_id}   [{args.index_name}]")
    for note in notes:
        print(f"  note: {note}")
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
