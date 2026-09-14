"""The embed component: descriptions + transcript -> vectors.

Embeds **only what changed**, keyed by a hash of the unit's own text. A re-run
against unchanged documents costs nothing and says so.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ...shared import paths, sinks
from ...shared.documents import Produced
from . import embedders as embedders_mod
from . import readable
from . import units as units_mod
from . import indexes as backends

def _embed_video(video_id: str, built, names: Sequence[str]) -> int:
    """Embed the video-level summary into `video_embeddings`. Postgres only.

    `falconvar.video_embeddings` existed in the DDL before anything wrote it:
    a complete `--tier llm` run left it at 0 rows while every other table was
    exactly full. This is what fills it.

    Postgres only for now: Qdrant would need a second collection with its own
    name, and the whole-video corpus is one row per video, which is not a size
    that needs a vector database.
    """
    if "supabase" not in names:
        return 0
    from ...shared import paths, sinks
    from .indexes.supabase import write_video_unit

    path = paths.artifact(video_id, "aggregates") / "summary.json"
    if not path.exists():
        return 0
    try:
        payload = sinks.read_json(path).get("payload") or {}
        unit = units_mod.from_summary(video_id, payload)
        if unit is None:
            return 0
        unit.vector = built.embed([unit.content])[0]
        return write_video_unit(unit, built.key)
    except Exception:                                    # noqa: BLE001
        return 0


def collect(video_id: str) -> list[units_mod.Unit]:
    """Every embeddable unit this video has, from both modalities."""
    out: list[units_mod.Unit] = []
    if paths.exists(video_id, "descriptions"):
        from ...describe import load as load_descriptions
        out += units_mod.from_descriptions(load_descriptions(video_id))
    if paths.exists(video_id, "transcript"):
        from ...cut import load as load_transcript
        out += units_mod.from_transcript(load_transcript(video_id))
    return out


DEFAULT_INDEX = "qdrant"


def run(video_id: str, embedder: Optional[str] = None,
        model: Optional[str] = None,
        batch: int = 64,
        index_name: str | Sequence[str] = DEFAULT_INDEX,
        sink: str | Sequence[str] = "file") -> Produced:
    """Embed what changed, into one or more indexes.

    Several indexes take the same vectors: embedding is the paid half and the
    backends are the cheap one, so writing to `local,qdrant` costs one set of
    API calls rather than two.

    `embedder` is a provider or `provider/model`; None resolves through
    `shared.providers` -- FALCONVAR_EMBEDDER, then openai -- exactly as
    `retrieve` resolves it, so the two cannot disagree about the space.
    """
    built = embedders_mod.build(embedder, model=model)
    names = ([n.strip() for n in index_name.split(",") if n.strip()]
             if isinstance(index_name, str) else list(index_name))
    indexes = [(n, backends.build(n, video_id, built.key)) for n in names]
    try:
        return _embed(built, names, indexes, video_id, batch)
    finally:
        # Every one, in a `finally`: a failure part-way through would otherwise
        # leave an embedded Qdrant lock held for the life of the process.
        for _, target in indexes:
            backends.release(target)


def _embed(built, names, indexes, video_id: str, batch: int) -> Produced:

    wanted = collect(video_id)
    if not wanted:
        raise FileNotFoundError(
            f"{video_id}: nothing to embed -- no descriptions and no transcript")

    # Per index, not once. A unit needs embedding if *any* target lacks it or
    # holds a different hash -- otherwise adding a second backend later would
    # fill it only with what changed since, a subset nothing reports as
    # incomplete. This is also why the vectors must exist before any upsert:
    # an unchanged unit carries `vector=None`, and a backend handed one writes
    # nothing and says it succeeded.
    stored = {name: target.stored_hashes() for name, target in indexes}
    needs: dict[str, list[units_mod.Unit]] = {name: [] for name, _ in indexes}
    for unit in wanted:
        for name in needs:
            if stored[name].get(unit.key) != unit.text_hash:
                needs[name].append(unit)

    changed = [u for u in wanted
               if any(u in needs[name] for name in needs)]
    for start in range(0, len(changed), batch):
        window = changed[start:start + batch]
        for unit, vector in zip(window, built.embed([u.content for u in window])):
            unit.vector = vector

    live = {u.key for u in wanted}
    artifacts: dict[str, str] = {}

    # Written whatever the index, because it answers a question no index can:
    # what text did this chunk actually contribute. Free -- the units are
    # already in hand and the vectors are left out.
    from ...boundaries import load as load_timeline
    try:
        fingerprint = load_timeline(video_id).fingerprint()
    except Exception:                                    # noqa: BLE001
        fingerprint = ""
    artifacts["embedded"] = readable.write(video_id, wanted, fingerprint)

    dropped = 0
    for name, target in indexes:
        target.upsert(needs[name])
        dropped += target.prune(live)
        artifacts[name] = str(target.save())

    # The whole-video vector, if this video has a summary to make one from.
    # After the moments, and best-effort: it answers a different question, and
    # losing it must not fail a run whose moments are already written.
    videos = _embed_video(video_id, built, names)

    return Produced(
        video_id=video_id, component="embed", backend=",".join(names),
        artifacts=artifacts,
        stats={"units": len(wanted), "embedded": len(changed),
               "video_units": videos,
               "unchanged": len(wanted) - len(changed), "pruned": dropped,
               "embedder": built.key, "indexes": names,
               "samplers": sorted({u.sampler_id for u in wanted})},
    )


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Embed what changed.")
    ap.add_argument("video_id")
    ap.add_argument("--embedder", default=None,
                    help="a provider or provider/model; default FALCONVAR_EMBEDDER, "
                         f"then openai. Known: {', '.join(embedders_mod.available())}")
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--index", default=DEFAULT_INDEX, dest="index_name",
                    help=f"comma-separated; known: "
                         f"{', '.join(backends.available())}")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        produced = run(args.video_id, args.embedder, args.model, args.batch,
                       args.index_name)
    except (KeyError, ValueError, FileNotFoundError,
            embedders_mod.EmbedderUnavailable) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    s = produced.stats
    print(f"{produced.video_id}   {s['embedder']}")
    print(f"  units        {s['units']}   from {', '.join(s['samplers'])}")
    print(f"  embedded     {s['embedded']}   ({s['unchanged']} unchanged)")
    if s["pruned"]:
        print(f"  pruned       {s['pruned']}   (chunks that no longer exist)")
    print()
    for name, where in produced.artifacts.items():
        print(f"  {name:<12} -> {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
