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

from ver2 import db, orchestrate
from ver2.embed import defaults as embed_defaults
from ver2.embed import embedders as embedders_mod
from ver2.embed import index as index_mod
from ver2.embed import units as units_mod
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

    The colon form is the same one the CLI takes: `uniform:text` keeps a frame
    on a clock and has describe ask the *text* question about it. Only the
    positional samplers accept it -- a change sampler already implies its own
    question, so naming a second one would be two answers to one thing.
    """
    settings = settings or {}
    rate = {"min_interval_s": settings.get("min_interval", 0.0),
            "max_per_chunk": settings.get("max_per_chunk")}
    tuned = ({} if settings.get("threshold") is None
             else {"threshold": settings["threshold"]})
    every_s = settings.get("every_seconds", 3.0)

    built = []
    for entry in names:
        name, _, prompt = entry.partition(":")
        if prompt and name not in ("uniform", "overview"):
            raise ValueError(
                f"{entry!r}: only 'uniform' takes a question after a colon; "
                "a change sampler already implies its own question")
        if name == "uniform":
            built.append(samplers_mod.build(name, every_s=every_s,
                                            prompt=prompt or None, **rate))
        elif name == "overview":
            built.append(samplers_mod.build(name, every_s=every_s, **rate))
        elif name == "objects":
            vocab = settings.get("vocabulary")
            if isinstance(vocab, str):
                vocab = [v.strip() for v in vocab.split(",") if v.strip()]
            built.append(samplers_mod.build(name, vocabulary=vocab or None,
                                            **tuned, **rate))
        elif name == "clip":
            built.append(samplers_mod.build(name, mode=settings.get("mode",
                                                                    "reference"),
                                            **tuned, **rate))
        else:
            built.append(samplers_mod.build(name, **tuned, **rate))
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

    return {
        "samplers": samplers_mod.available(),
        "chunking": list(orchestrate.POLICIES),
        "describers": describers_mod.available(),
        "transcribers": transcribe_mod.available(),
        "diarizers": diarize_mod.available(),
        "embedders": embedders_mod.available(),
        "indexes": list(index_mod.BACKENDS),
        "sinks": list(orchestrate.SINKS),
        "defaults": {"embedder": embed_defaults.embedder(),
                     "index": embed_defaults.index()},
    }
