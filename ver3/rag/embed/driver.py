"""The embed component: descriptions + transcript -> vectors.

Embeds **only what changed**, keyed by a hash of the unit's own text. A re-run
against unchanged documents costs nothing and says so.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ...shared import paths, sinks
from ...shared.documents import Produced
from . import embedders as embedders_mod
from . import units as units_mod
from . import indexes as backends

DEFAULT_EMBEDDER = "openai"


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


DEFAULT_INDEX = "local"


def run(video_id: str, embedder: str = DEFAULT_EMBEDDER,
        model: Optional[str] = None,
        batch: int = 64,
        index_name: str | Sequence[str] = DEFAULT_INDEX,
        sink: str | Sequence[str] = "file") -> Produced:
    """Embed what changed, into one or more indexes.

    Several indexes take the same vectors: embedding is the paid half and the
    backends are the cheap one, so writing to `local,qdrant` costs one set of
    API calls rather than two.
    """
    built = embedders_mod.build(embedder, **({"model": model} if model else {}))
    names = ([n.strip() for n in index_name.split(",") if n.strip()]
             if isinstance(index_name, str) else list(index_name))
    indexes = [(n, backends.build(n, video_id, built.key)) for n in names]
    index = indexes[0][1]

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
    dropped = 0
    for name, target in indexes:
        target.upsert(needs[name])
        dropped += target.prune(live)
        artifacts[name] = str(target.save())

    return Produced(
        video_id=video_id, component="embed", backend=",".join(names),
        artifacts=artifacts,
        stats={"units": len(wanted), "embedded": len(changed),
               "unchanged": len(wanted) - len(changed), "pruned": dropped,
               "embedder": built.key, "indexes": names,
               "samplers": sorted({u.sampler_id for u in wanted})},
    )


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Embed what changed.")
    ap.add_argument("video_id")
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER,
                    choices=embedders_mod.available())
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
