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
from .index import LocalIndex

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


def run(video_id: str, embedder: str = DEFAULT_EMBEDDER,
        model: Optional[str] = None,
        batch: int = 64,
        sink: str | Sequence[str] = "file") -> Produced:
    built = embedders_mod.build(embedder, **({"model": model} if model else {}))
    index = LocalIndex(video_id, built.key)

    wanted = collect(video_id)
    if not wanted:
        raise FileNotFoundError(
            f"{video_id}: nothing to embed -- no descriptions and no transcript")

    stored = index.stored_hashes()
    changed = [u for u in wanted if stored.get(u.key) != u.text_hash]

    for start in range(0, len(changed), batch):
        window = changed[start:start + batch]
        for unit, vector in zip(window, built.embed([u.content for u in window])):
            unit.vector = vector

    index.upsert(changed)
    dropped = index.prune({u.key for u in wanted})
    path = index.save()

    return Produced(
        video_id=video_id, component="embed", backend="file",
        artifacts={"index": str(path)},
        stats={"units": len(wanted), "embedded": len(changed),
               "unchanged": len(wanted) - len(changed), "pruned": dropped,
               "embedder": built.key,
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
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        produced = run(args.video_id, args.embedder, args.model, args.batch)
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
    print(f"\nindex -> {produced.artifacts['index']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
