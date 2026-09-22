"""The retrieve component: a query -> ranked moments. Writes nothing.

**The query is embedded with the embedder that built the index.** Not a
default, an argument. A mismatch across widths fails loudly; a mismatch between
two models of the same width returns a well-formed ranking that means nothing,
which is why the embedder key is in the index filename -- a wrong name searches
a file that does not exist rather than the wrong vectors.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..boundaries import load as load_timeline
from ...shared import paths
from ..embed import embedders as embedders_mod
from ..embed.driver import DEFAULT_INDEX
from ..embed import indexes as backends
from .search import Moment, to_moments


def scope_of(video_id: Any = None,
             video_ids: Optional[Sequence[str]] = None) -> Optional[list[str]]:
    """The set of videos to search. `None` means every one.

    One video, three, or all of them is the same question asked over a
    different set -- so the scope is a set, and the single-video case is the
    one-element case rather than a second endpoint. `video_id` stays accepted
    as the shorthand it always was.
    """
    if video_ids is not None:
        chosen = [v for v in video_ids if v]
        return chosen or None
    if isinstance(video_id, str) and video_id:
        return [video_id]
    if isinstance(video_id, (list, tuple, set)):
        chosen = [v for v in video_id if v]
        return chosen or None
    return None


def spans_of(video_id: str) -> list[tuple[float, float]]:
    """The grid, from the local file or from `chunks` if there is none.

    A span is stored in exactly one place -- the grid -- and everything else
    joins on `chunk_id`. But `load_timeline` reads `timeline.json`, so a
    deployment holding every row in Postgres and no output directory could not
    search at all: measured, moving one file aside turned a working query into
    `404 No such file or directory`. The database is the other copy of the same
    grid, so it is the fallback rather than a second source of truth.
    """
    try:
        return load_timeline(video_id).spans
    except FileNotFoundError:
        from ...shared.storage import db
        rows = (db.client(write=False).table("chunks")
                .select("chunk_id,start_ts,end_ts")
                .eq("video_id", video_id).order("chunk_id").execute().data or [])
        return [(float(r["start_ts"]), float(r["end_ts"])) for r in rows]


def chunks_in(spans: Sequence[tuple[float, float]],
              after: Optional[float], before: Optional[float]) -> list[int]:
    """Which chunks overlap a time window.

    Time is not a field on a vector and deliberately is not one: a span lives
    in the grid, so seconds are resolved to chunk ids here and the filter that
    reaches the store is the one that already existed. That keeps one mechanism
    for both backends and needs no Qdrant payload change -- payload is written
    only on upsert, so adding a field there would need a forced re-index.
    """
    lo = float("-inf") if after is None else after
    hi = float("inf") if before is None else before
    return [i for i, (start, end) in enumerate(spans) if end > lo and start < hi]


def search(query: str, video_id: Any = None, embedder: Optional[str] = None,
           moments: int = 5,
           sampler: Optional[str] = None, index_name: str = DEFAULT_INDEX,
           candidates: int = 20,
           question: Optional[str] = None,
           strategy: Optional[str] = None,
           chunk_ids: Optional[Sequence[int]] = None,
           window: int = 0,
           after: Optional[float] = None,
           before: Optional[float] = None,
           structured: Optional[dict[str, Any]] = None,
           video_ids: Optional[Sequence[str]] = None
           ) -> tuple[list[Moment], list[str]]:
    """Ranked moments, and anything the caller should be told about the ranking.

    The notes are not decoration. A dense-only backend returns hits that look
    exactly like fused ones minus a `t` marker, which reads as "no lexical
    match for this query" rather than "this index cannot have one".
    """
    if not (query or "").strip():
        # Refused here rather than at the provider. An empty string reached
        # OpenAI and came back `400 Invalid 'input'` -- an error about the
        # request body, from inside the embedder, for a mistake the caller made
        # and can fix. A search for nothing has no answer worth inventing.
        raise ValueError("a search needs a query; this one is empty")

    scope = scope_of(video_id, video_ids)
    built = embedders_mod.build(embedder)
    # The index is built with one id only so `stored_hashes`/`prune` keep
    # working for the writer; the scope a *search* uses is passed per call.
    index = backends.build(index_name, scope[0] if scope else "", built.key)
    try:
        return _search(built, index, index_name, query, scope, moments,
                       sampler, question, candidates, strategy, chunk_ids,
                       window, after, before, structured)
    finally:
        # Embedded Qdrant holds an exclusive folder lock. A CLI run never
        # notices -- the process exits -- but a server that does not release it
        # fails every later search with "already accessed by another instance",
        # on a route that worked a minute earlier. `finally`, because the
        # not-indexed raise below is exactly the exit that would leak it.
        backends.release(index)


def _search(built, index, index_name: str, query: str,
            scope: Optional[Sequence[str]],
            moments: int, sampler: Optional[str], question: Optional[str],
            candidates: int, strategy: Optional[str],
            chunk_ids: Optional[Sequence[int]], window: int,
            after: Optional[float], before: Optional[float],
            structured: Optional[dict[str, Any]]
            ) -> tuple[list[Moment], list[str]]:
    notes: list[str] = []
    if not backends.has_lexical(index_name):
        notes.append(f"{index_name} is dense-only: this ranking has no lexical "
                     "half, which is worth 0.429 against 0.714 top-1 on the "
                     "reference corpus")

    # A grid per video in scope. `chunk_id` is an index into ONE video's grid,
    # so it means nothing without knowing whose.
    known = list(scope) if scope else _all_videos()
    spans = {vid: spans_of(vid) for vid in known}
    one = spans.get(known[0], []) if len(known) == 1 else spans

    # A time window and a chunk set are the same filter. Resolved before the
    # query so exactly one mechanism reaches the store.
    wanted: Optional[set[int]] = None
    if chunk_ids:
        # Truthiness, not `is not None`: an empty list is no constraint, which
        # is what an empty `video_ids` already meant. Two list-valued filters
        # reading an empty list opposite ways is a difference nobody could
        # guess -- `chunk_ids=[]` returned nothing while `video_ids=[]`
        # returned everything.
        wanted = set(int(c) for c in chunk_ids)
    if after is not None or before is not None:
        # Over every grid in scope: the same second is a different chunk id in
        # each video, so the union is what a window means across a set.
        in_window: set[int] = set()
        for vid in known:
            in_window |= set(chunks_in(spans.get(vid, []), after, before))
        wanted = in_window if wanted is None else (wanted & in_window)
    if wanted is not None and window:
        # A neighbourhood, because "more context around chunk 6" is almost
        # always 5, 6, 7. Chunk ids are contiguous over the grid, so a window is
        # arithmetic rather than another query.
        longest = max((len(v) for v in spans.values()), default=0)
        widened = {c + step for c in wanted
                   for step in range(-window, window + 1)}
        wanted = {c for c in widened if 0 <= c < longest} or wanted
    narrowed = sorted(wanted) if wanted is not None else None
    if narrowed is not None and not narrowed:
        return [], notes + ["no chunk matches that window"]

    # The query side: e5, nomic and bge embed a question differently from the
    # passage it should find, and a query embedded as a document loses recall
    # with no error anywhere.
    vector = embedders_mod.query_vector(built, query)
    hits = index.search(vector, query, candidates, sampler, question,
                        strategy, narrowed, structured, scope)
    if not hits:
        if any(f is not None for f in (sampler, question, strategy,
                                       narrowed, structured, scope)):
            # Nothing matched a filter is a different answer from nothing
            # indexed, and only one of them is worth re-running `embed` over.
            #
            # `scope` counts as a filter: naming a video that holds nothing is
            # "no such video in this index", not "this deployment has never
            # embedded anything" -- and the raise below tells you to run
            # `embed`, which would not help.
            #
            # Named with the embedder: a space holds only what was embedded
            # with that model, so searching with one that never indexed these
            # videos comes back empty in exactly this way.
            return [], notes + [f"nothing matched those filters in {built.key} -- "
                                "if that embedder never indexed these videos, "
                                "embed them with it first"]
        where = ", ".join(scope) if scope else "any video"
        raise FileNotFoundError(
            f"{where}: nothing indexed for {built.key} in {index_name!r}. "
            f"Run embed with this embedder and index first -- a different "
            f"embedder writes a different collection.")
    if any(f is not None for f in (sampler, question, strategy, structured)):
        notes.append("filtering gives up the agreement signal: a chunk "
                     "contributes fewer terms, so scores fall -- to a single "
                     "1/(k+rank) when only one unit per chunk survives")
    if scope is None or len(known) > 1:
        # More than one video in one ranking is the case the chunk aggregation
        # was designed for and has never been exercised: a chunk contributes
        # one term per sampler that described it, so a video described by more
        # samplers would win on count if the score summed rather than taking
        # the best plus a discounted second.
        notes.append(f"scope is {len(known)} videos: moments are keyed by "
                     "(video_id, chunk_id), since a chunk id only means "
                     "something inside one grid")
    return to_moments(hits, known[0] if len(known) == 1 else "",
                      one, moments), notes


def _all_videos() -> list[str]:
    """Every video this deployment has a grid for.

    From disk, falling back to `timelines` -- the same two-copies-of-one-grid
    rule `spans_of` follows, so a database-only deployment still resolves a
    span.
    """
    from ...shared import paths
    found = paths.videos()
    if found:
        return found
    try:
        from ...shared.storage import db
        rows = (db.client(write=False).table("timelines")
                .select("video_id").execute().data or [])
        return [r["video_id"] for r in rows]
    except Exception:                                    # noqa: BLE001
        return []


def videos(query: str, embedder: Optional[str] = None, limit: int = 5
           ) -> list[dict[str, Any]]:
    """Which video is this about. A different question from which moment.

    `search` answers *which twenty seconds*, and every filter it takes narrows
    inside one video. This ranks whole videos by their summary, from
    `video_embeddings` -- so a caller can find the video first and then search
    inside it, which is the two-step a single index cannot serve: a whole-video
    "moment" beside real ones would be a result nobody can play.

    Postgres only, because that is where the summaries are written.
    """
    from ..embed.indexes.supabase import search_videos

    built = embedders_mod.build(embedder)
    # The query side: e5, nomic and bge embed a question differently from the
    # passage it should find, and a query embedded as a document loses recall
    # with no error anywhere.
    vector = embedders_mod.query_vector(built, query)
    return search_videos(vector, built.key, limit)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Search one video's moments.")
    ap.add_argument("query")
    ap.add_argument("video_id")
    ap.add_argument("--embedder", default=None,
                    help="the one that built the index: a provider or "
                         "provider/model; default FALCONVAR_EMBEDDER, then openai")
    ap.add_argument("--moments", type=int, default=5)
    ap.add_argument("--sampler", default=None,
                    help="narrow to one pairing, e.g. `clip:text`")
    ap.add_argument("--question", default=None,
                    help="narrow to one question across every sampler that "
                         "asked it, e.g. `text`")
    ap.add_argument("--strategy", default=None,
                    help="one sampler's whole output, e.g. `clip`")
    ap.add_argument("--chunks", default=None, dest="chunk_ids",
                    help="comma-separated chunk ids to search within")
    ap.add_argument("--window", type=int, default=0,
                    help="widen --chunks by N neighbours each side")
    ap.add_argument("--after", type=float, default=None,
                    help="seconds; resolved to chunk ids through the grid")
    ap.add_argument("--before", type=float, default=None, help="seconds")
    ap.add_argument("--where", default=None, metavar="FIELD=VALUE",
                    help="exact structured values, comma-separated, e.g. "
                         "`severity=severe,environment=outdoor`")
    ap.add_argument("--candidates", type=int, default=20,
                    help="units ranked per half before fusion (default 20)")
    ap.add_argument("--index", default=DEFAULT_INDEX, dest="index_name",
                    choices=backends.available())
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    chunk_ids = ([int(c) for c in args.chunk_ids.split(",") if c.strip()]
                 if args.chunk_ids else None)
    structured = None
    if args.where:
        structured = dict(pair.split("=", 1) for pair in args.where.split(",")
                          if "=" in pair)

    try:
        found, notes = search(args.query, args.video_id, args.embedder,
                              args.moments, args.sampler,
                              args.index_name, args.candidates,
                              question=args.question, strategy=args.strategy,
                              chunk_ids=chunk_ids, window=args.window,
                              after=args.after, before=args.before,
                              structured=structured)
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
