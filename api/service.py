"""The pipeline, as functions a request handler can call.

Everything here is a thin translation between HTTP-shaped values (strings from
a form, a video id from a path) and the library. No route handling, no
FastAPI, no argparse -- so the same functions serve the API, a notebook, or a
test, and none of them has to learn the pipeline twice.

The one thing this file genuinely owns is **building samplers from strings**.
The CLI does it in `video/ingest/driver._build_samplers` from an argparse
namespace; a form post arrives as a list and a dict instead. Rather than
construct a fake namespace, the rules live here once, in the shape a request
actually has.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from ver2 import aggregate as aggregate_mod
from ver2 import db, orchestrate
from ver2.embed import defaults as embed_defaults
from ver2.embed import embedders as embedders_mod
from ver2.embed import index as index_mod
from ver2.embed import units as units_mod
from ver2.aggregate.output import AggregateDocuments, MultiAggregateSink
from ver2.aggregate.reader import aggregate as run_aggregate, context_for
from ver2.embed import summaries as summaries_mod
from ver2.embed.indexer import index_units
from ver2.retrieve.search import search as search_moments
from ver2.video.describe import describers as describers_mod
from ver2.video.describe.output import DescriptionDocument, MultiDescriptionSink
from ver2.video.describe.reader import describe as run_describe
from ver2.video.ingest import samplers as samplers_mod

OUT_ROOT = Path("out")
UPLOADS = Path("uploads")


def build_samplers(names: Sequence[str],
                   settings: Optional[dict[str, Any]] = None) -> list:
    """Sampler objects from `["yolo", "uniform:text"]` plus a settings dict.

    The colon form is the same one the CLI takes, and every sampler accepts it:
    `name:prompt` pairs a strategy for choosing frames with a question to ask
    about them. `yolo:overview` keeps frames where the people changed and asks
    for prose; unpaired, the question is the sampler's own name.

    An unknown question is a ValueError here, which `main.py` turns into a 422 --
    rather than a run that quietly asks the scene question and bills for it.
    """
    from ver2.video.describe.vlm import prompts
    settings = settings or {}
    rate = {"min_interval_s": settings.get("min_interval", 0.0),
            "max_per_chunk": settings.get("max_per_chunk")}
    tuned = ({} if settings.get("threshold") is None
             else {"threshold": settings["threshold"]})
    # Absent, the positional samplers keep the default stride of 1. Passed,
    # it reaches every one of them: `overview` is a prompt rather than a
    # cadence, so `overview` and `uniform:overview` -- the same question under
    # two spellings -- must keep the same frames.
    stride = ({} if settings.get("every_frames") is None
              else {"every_n": settings["every_frames"]})

    built = []
    for entry in names:
        name, _, prompt = entry.partition(":")
        if prompt and prompt not in prompts.QUESTIONS:
            raise ValueError(
                f"{entry!r}: unknown question {prompt!r}; "
                f"known: {', '.join(prompts.QUESTIONS)}")
        ask = {"prompt": prompt} if prompt else {}
        if name == "uniform":
            built.append(samplers_mod.build(name, **ask, **stride, **rate))
        elif name == "objects":
            vocab = settings.get("vocabulary")
            if isinstance(vocab, str):
                vocab = [v.strip() for v in vocab.split(",") if v.strip()]
            built.append(samplers_mod.build(name, vocabulary=vocab or None,
                                            **ask, **tuned, **rate))
        elif name == "clip":
            built.append(samplers_mod.build(name, mode=settings.get("mode",
                                                                    "reference"),
                                            **ask, **tuned, **rate))
        else:
            built.append(samplers_mod.build(name, **ask, **tuned, **rate))
    return built


def ingest(options: orchestrate.Options, on_progress=None) -> dict[str, Any]:
    """Both streams onto one grid. Returns what a caller can show."""
    return orchestrate.process(options, on_progress=on_progress).as_dict()


def describe(video_id: str, describer: str = "openai",
             model: Optional[str] = None, sinks: Sequence[str] = ("file",),
             limit: Optional[int] = None, out_root: Path = OUT_ROOT) -> dict[str, Any]:
    """Ask a VLM about every (chunk, sampler) the manifest names."""
    # The key lives in .env, and a describer that cannot find it fails at
    # construction -- four stages into a job, with no terminal to have seen it.
    db.load_env()
    out_dir = Path(out_root) / video_id
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    built = []
    for name in sinks:
        if name == "file":
            built.append(DescriptionDocument(out_dir / "descriptions.json"))
        else:
            from ver2.video.describe.output import SupabaseDescriptions

            built.append(SupabaseDescriptions())
    sink = built[0] if len(built) == 1 else MultiDescriptionSink(*built)

    options: dict[str, Any] = {"model": model} if model else {}
    result = run_describe(manifest,
                          describer=describers_mod.build(describer, **options),
                          sink=sink, limit=limit)
    return {"video_id": video_id, "chunks": result.chunks_seen,
            "described": result.described, "skipped": result.skipped,
            "elapsed_s": round(result.elapsed_s, 2),
            "complete": bool(result.document.get("complete"))}


def embed(video_id: str, embedder: Optional[str] = None,
          model: Optional[str] = None, indexes: Optional[Sequence[str]] = None,
          out_root: Path = OUT_ROOT) -> dict[str, Any]:
    """Embed descriptions and, when there is one, the transcript.

    Both land in the same index with different `sampler` values, which is what
    makes a single query able to match what was seen and what was said.
    """
    out_dir = Path(out_root) / video_id
    units, sources = [], []
    descriptions = out_dir / "descriptions.json"
    if descriptions.exists():
        found = units_mod.from_document(
            json.loads(descriptions.read_text(encoding="utf-8")))
        units += found
        sources.append(f"{len(found)} descriptions")
    transcript = out_dir / "transcript.json"
    if transcript.exists():
        spoken = units_mod.from_transcript(
            json.loads(transcript.read_text(encoding="utf-8")))
        units += spoken
        sources.append(f"{len(spoken)} transcript chunks")
    if not units:
        raise FileNotFoundError(
            f"nothing to embed for {video_id}: no descriptions or transcript")

    db.load_env()
    emb = embedders_mod.build(embedder or embed_defaults.embedder(),
                              **({"model": model} if model else {}))
    names = list(indexes or embed_defaults.index().split(","))
    result = index_units(units, emb, index_mod.build(names),
                         video_id=video_id)
    return {**result.as_dict(), "video_id": video_id, "read": sources}


def search(query: str, video_id: Optional[str] = None,
           sampler: Optional[str] = None, moments: int = 5, limit: int = 20,
           embedder: Optional[str] = None, model: Optional[str] = None,
           indexes: Optional[Sequence[str]] = None) -> list[dict[str, Any]]:
    """One question, ranked moments back."""
    db.load_env()
    emb = embedders_mod.build(embedder or embed_defaults.embedder(),
                              **({"model": model} if model else {}))
    names = list(indexes or embed_defaults.index().split(","))
    found = search_moments(query, emb, index_mod.build(names),
                           video_id=video_id, limit=limit, moments=moments,
                           sampler=sampler)
    return [m.as_dict() for m in found]


def videos(out_root: Path = OUT_ROOT) -> list[dict[str, Any]]:
    """What has been processed, read off disk rather than remembered.

    Disk is the record. A restarted server forgets which jobs ran; it does not
    forget what they produced, and this is why.
    """
    root = Path(out_root)
    if not root.is_dir():
        return []
    out = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if directory.name == "qdrant":
            continue
        artifacts = {name: (directory / f"{name}.json").exists()
                     for name in ("manifest", "timeline", "descriptions",
                                  "transcript")}
        entry: dict[str, Any] = {"video_id": directory.name, "has": artifacts,
                                 "frames": 0}
        store = directory / "store"
        if store.is_dir():
            entry["frames"] = sum(1 for _ in store.glob("*.jpg"))
        timeline = directory / "timeline.json"
        if timeline.exists():
            grid = json.loads(timeline.read_text(encoding="utf-8"))
            entry["chunks"] = len(grid.get("chunks", []))
            entry["policy"] = grid.get("policy")
            entry["timeline_fingerprint"] = grid.get("fingerprint")
        out.append(entry)
    return out


def artifact(video_id: str, name: str, out_root: Path = OUT_ROOT) -> dict[str, Any]:
    """One of the JSON documents a video produced."""
    path = Path(out_root) / video_id / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{video_id} has no {name}.json")
    return json.loads(path.read_text(encoding="utf-8"))


#: Every JSON document a video can produce, and what each one answers. The
#: order is the order the pipeline produces them, which is also the order a
#: consumer most often wants to read them in.
ARTIFACTS = {
    "manifest": "which frames were kept, why, and how to fetch them again",
    "timeline": "the chunk grid: spans, the policy, and its fingerprint",
    "descriptions": "one VLM answer per (chunk, sampler)",
    "transcript": "words, speakers and turns, cut to the same grid",
}


def exports(video_id: str, out_root: Path = OUT_ROOT) -> dict[str, Any]:
    """Every document this video can hand to another service, with its URL.

    A caller that wants "the summaries" should not have to know that summaries
    live under `aggregates/` while descriptions live one level up. This is the
    one place that mapping is written down, and both the bundle route and the
    browser's download links read it.
    """
    directory = Path(out_root) / video_id
    if not directory.is_dir():
        raise FileNotFoundError(f"no such video: {video_id}")

    found: list[dict[str, Any]] = []
    for name, what in ARTIFACTS.items():
        path = directory / f"{name}.json"
        if path.exists():
            found.append({"kind": "artifact", "name": name, "about": what,
                          "url": f"/videos/{video_id}/{name}",
                          "bytes": path.stat().st_size})
    for path in sorted((directory / "aggregates").glob("*.json")):
        found.append({"kind": "aggregate", "name": path.stem,
                      "about": aggregate_mod.about(path.stem),
                      "url": f"/videos/{video_id}/aggregates/{path.stem}",
                      "bytes": path.stat().st_size})
    return {"video_id": video_id, "exports": found,
            "bundle": f"/videos/{video_id}/export"}


def bundle(video_id: str, out_root: Path = OUT_ROOT) -> dict[str, Any]:
    """Everything a video produced, as one document.

    Four requests plus one per aggregate is a lot of round trips for a consumer
    that wants the lot; this is that, once. Absent documents are absent keys
    rather than nulls, so the shape says what the run actually did -- an
    audio-only video has no `manifest`, and that is information.
    """
    directory = Path(out_root) / video_id
    if not directory.is_dir():
        raise FileNotFoundError(f"no such video: {video_id}")
    out: dict[str, Any] = {"video_id": video_id}
    for name in ARTIFACTS:
        path = directory / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    found = aggregates(video_id, out_root)
    if found:
        out["aggregates"] = found
    return out


def frame_path(video_id: str, index: int, out_root: Path = OUT_ROOT) -> Path:
    """The JPEG a moment cites as evidence, straight out of the frame store."""
    path = Path(out_root) / video_id / "store" / f"{index:07d}.jpg"
    if not path.exists():
        raise FileNotFoundError(
            f"{video_id} has no frame {index}. The store is written only when "
            "a run asks for it, and holds only the frames a sampler kept.")
    return path


def available() -> dict[str, Any]:
    """What this deployment can be asked for. Read from the registries, so a
    new sampler or embedder appears here without anyone editing a list."""
    from ver2.audio import diarize as diarize_mod
    from ver2.audio import transcribe as transcribe_mod

    from ver2.video.describe.vlm import prompts

    return {
        "samplers": samplers_mod.available(),
        # Any sampler may be paired with any of these as `name:prompt`. Published
        # so a client can offer the pairing without hard-coding the list, and so
        # `samplers` stays a list of strategies rather than growing an entry per
        # question the way `overview` once was.
        "prompts": prompts.questions(),
        # Pairings worth offering as they stand. The `samplers` chips send their
        # own text through as `samplers=`, so a pairing listed here is selectable
        # with no client change -- which is what keeps `overview` reachable from
        # the browser now that it is a question rather than a sampler.
        "pairings": ["uniform:overview", "uniform:text"],
        "chunking": list(orchestrate.POLICIES),
        "describers": describers_mod.available(),
        "transcribers": transcribe_mod.available(),
        "diarizers": diarize_mod.available(),
        "embedders": embedders_mod.available(),
        "indexes": list(index_mod.BACKENDS),
        "sinks": list(orchestrate.SINKS),
        "aggregators": {name: {"tier": aggregate_mod.TIER_OF.get(name, "llm"),
                               "about": aggregate_mod.about(name)}
                        for name in aggregate_mod.available()},
        "tiers": list(aggregate_mod.TIERS),
        "artifacts": dict(ARTIFACTS),
        # Read off the Options dataclass rather than restated, because a
        # registry list is alphabetical and `stub` sorts before `whisper`: a
        # form that offers the list in order defaults to the stub and produces
        # a transcript of `[stub0.0]` that no stage reports as wrong.
        "defaults": {"embedder": embed_defaults.embedder(),
                     "index": embed_defaults.index(),
                     "chunking": orchestrate.Options.chunking,
                     "transcriber": orchestrate.Options.transcriber,
                     "diarizer": orchestrate.Options.diarizer,
                     "describer": "openai"},
    }


def aggregate(video_id: str, tier: str = "free",
              aggregators: Optional[Sequence[str]] = None,
              sinks: Sequence[str] = ("file",), force: bool = False,
              embedder: Optional[str] = None,
              indexes: Optional[Sequence[str]] = None,
              out_root: Path = OUT_ROOT, on_progress=None) -> dict[str, Any]:
    """Build video-level structure, and embed the summary if one was made."""
    db.load_env()
    ctx = context_for(video_id, out_root)
    if not ctx.chunks:
        raise FileNotFoundError(
            f"nothing to aggregate for {video_id}: no descriptions or transcript")

    names = list(aggregators) if aggregators else aggregate_mod.by_tier(tier)

    # Only novelty reads the index, so a free pass needs no key.
    if "novelty" in names:
        try:
            ctx.embedder = embedders_mod.build(embedder or embed_defaults.embedder())
            ctx.index = index_mod.build(
                list(indexes or embed_defaults.index().split(",")))
        except Exception:                               # noqa: BLE001
            pass                                        # novelty reports nothing

    out_dir = Path(out_root) / video_id / "aggregates"
    documents = AggregateDocuments(out_dir)
    built: list[Any] = []
    for name in sinks:
        if name == "file":
            built.append(documents)
        else:
            from ver2.aggregate.output import SupabaseAggregates

            built.append(SupabaseAggregates())
    sink = built[0] if len(built) == 1 else MultiAggregateSink(*built)

    result = run_aggregate(ctx, names, sink=sink, on_progress=on_progress,
                           existing=documents.existing(), force=force)
    out = result.as_dict()

    # The summary is the one aggregate worth a vector: it answers "which
    # video", which no per-chunk vector can. Indexed here rather than in a
    # separate call so a caller cannot end up with a summary nothing can find.
    if "summary" in result.produced or force:
        try:
            emb = embedders_mod.build(embedder or embed_defaults.embedder())
            out["summary_indexed"] = summaries_mod.index_summary(
                video_id, emb, out_root=out_root, force=force)
        except Exception as exc:                        # noqa: BLE001
            out["summary_indexed"] = {"error": str(exc)}
    return out


def aggregates(video_id: str, out_root: Path = OUT_ROOT) -> dict[str, Any]:
    """Every aggregate a video has, keyed by id."""
    directory = Path(out_root) / video_id / "aggregates"
    if not directory.is_dir():
        return {}
    out = {}
    for path in sorted(directory.glob("*.json")):
        out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return out


def search_videos(query: str, limit: int = 10, embedder: Optional[str] = None,
                  model: Optional[str] = None) -> list[dict[str, Any]]:
    """Which video, rather than which moment."""
    db.load_env()
    emb = embedders_mod.build(embedder or embed_defaults.embedder(),
                              **({"model": model} if model else {}))
    return summaries_mod.search(query, emb, limit=limit)
