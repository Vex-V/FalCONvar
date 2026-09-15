"""The aggregates driver: what video_rag extracted -> `aggregates/<name>.json`.

Each aggregator writes its own file, so a run that dies partway leaves the
results it did produce rather than none -- the same reason tiers run cheapest
first.

**Reaches video_rag through its driver, never its components.** Documents,
identity declarations, an embedder and the whole-video vector all come from
`video_rag.driver`, imported inside the functions that need them so importing
this package loads nothing from the other tier. video_rag, for its part, never
reads what this writes: the summary is handed to it, not left for it to find.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..shared import paths
from ..shared.storage import sinks
from ..shared.contracts.documents import Aggregate, Produced, fingerprint_of
from . import TIER_OF, available, resolve
from .base import TIERS, Context, missing, resolve_order


def context_for(video_id: str) -> Context:
    """Every document video_rag wrote for this video. Missing ones are None."""
    from ..video_rag import driver as video_rag

    found = video_rag.documents(video_id)
    descriptions = found["descriptions"]
    identity: dict[str, dict[str, list[str]]] = {}
    if descriptions is not None:
        asked = {block.get("question") or sampler_id
                 for chunk in descriptions.chunks
                 for sampler_id, block in (chunk.get("samplers") or {}).items()}
        identity = {q: declared for q in sorted(asked)
                    if (declared := video_rag.identity_of(q))}
    return Context(video_id, found["timeline"], found["manifest"], descriptions,
                   found["transcript"], identity)


#: Who made an `llm` aggregate that does not say. Before providers existed
#: every one came from OpenAI's default -- `run` had no way to name another --
#: so reading a missing field as this is a fact, not a guess. It is also what
#: stops the first run after that change paying to rebuild every summary.
LEGACY_MODEL = "openai:gpt-5.4-mini"


def made_by(document: Aggregate) -> Optional[str]:
    """`provider:model` for an aggregate a model wrote; None for arithmetic."""
    recorded = document.stats.get("model")
    if recorded is None and document.tier == "llm":
        return LEGACY_MODEL
    return recorded


def validate(tier: str = "free", llm: Optional[str] = None) -> list[str]:
    """What stops a run before it starts, as messages."""
    if tier not in TIERS:
        return [f"tier must be one of {', '.join(TIERS)}"]
    if tier != "llm":
        return []
    from ..shared.models import providers
    return providers.problems("llm", llm)


def run(video_id: str, tier: str = "free",
        only: Optional[Sequence[str]] = None,
        force: bool = False,
        sink: str | Sequence[str] = "file",
        llm: Optional[str] = None,
        embedder: Optional[str] = None,
        index: Optional[str] = None) -> Produced:
    """Run every aggregator up to ``tier``, cheapest first.

    `llm` is who answers the paid tier and `embedder` who embeds for it -- each
    a provider or `provider/model`, None resolving through
    `shared.models.providers`. `index` naming `supabase` also stores the
    summary as the video's vector, which `/search level=video` ranks.
    """
    if tier not in TIERS:
        raise KeyError(f"unknown tier {tier!r}; known: {', '.join(TIERS)}")
    ceiling = TIERS.index(tier)
    names = [n for n in (only or available())
             if TIERS.index(TIER_OF[n]) <= ceiling]
    context = context_for(video_id)
    fingerprint = context.inputs_fingerprint()

    directory = paths.artifact(video_id, "aggregates")
    produced: dict[str, str] = {}
    skipped: dict[str, str] = {}
    ran: list[str] = []
    current = 0
    models: set[str] = set()

    for name in resolve_order(names, TIER_OF):
        # Only the paid tier takes a provider, and only an aggregator that
        # embeds takes an embedder. The local models name their own
        # checkpoints, and arithmetic needs neither.
        cls = resolve(name)
        if TIER_OF[name] != "llm":
            aggregator = cls()
        elif getattr(cls, "embeds", False):
            aggregator = cls(llm, embedder)
        else:
            aggregator = cls(llm)
        author = getattr(aggregator, "model_key", None)
        # An aggregator whose answer depends on more than the chunk text --
        # what the shapes declare as identity -- folds that in, so a changed
        # declaration rebuilds it rather than reusing a stale answer. The rest
        # keep the fingerprint they always had.
        inputs_of = getattr(aggregator, "inputs_of", None)
        expected = (fingerprint if inputs_of is None else
                    fingerprint_of({"text": fingerprint, "declared": inputs_of(context)}))
        why = missing(aggregator, context, ran)
        if why is not None:
            skipped[name] = why
            continue

        path = directory / f"{name}.json"

        # The fingerprint governs whether to RECOMPUTE, not whether to write.
        # Those are different questions: a run that adds a backend has nothing
        # to recompute and everything to write, and conflating them means the
        # new destination silently stays empty while the run reports success.
        # Exactly the bug `embed` had across two indexes.
        #
        # And the model is part of "current". Same text, different model is a
        # different answer the caller asked for -- reusing the stored one would
        # report success for a switch that never happened, which is the silent
        # no-op `describe` keys its resume on the describer to avoid.
        stored = None
        if not force and path.exists():
            candidate = Aggregate.from_dict(sinks.read_json(path))
            if (candidate.inputs_fingerprint == expected
                    and made_by(candidate) == author):
                stored = candidate

        if stored is not None:
            current += 1
            document = stored
        else:
            payload = aggregator.run(context)
            document = Aggregate(video_id=video_id, aggregate_id=name,
                                 tier=aggregator.tier, payload=payload,
                                 inputs_fingerprint=expected,
                                 stats={"about": aggregator.about,
                                        **({"model": author} if author else {})})
        if author:
            models.add(author)

        # Not `sinks.write`: that resolves one path per artifact name, and each
        # aggregator writes its own file under `aggregates/`. The file half is
        # therefore explicit here, and the row half goes through the same
        # writer every other component uses -- so `--sink supabase` means the
        # same thing for this component as for the rest.
        backends = sinks.parse(sink)
        if "file" in backends and stored is None:
            produced[name] = str(sinks.write_json(path, document.as_dict()))
        elif "file" in backends:
            produced[name] = str(path)
        if "supabase" in backends:
            from ..shared.storage import rows
            rows.WRITERS["aggregate"](video_id, document.as_dict())
            produced.setdefault(name, "aggregate@supabase")
        ran.append(name)

    # The whole-video vector, from the summary this run has in hand. It lived in
    # `embed`, which read `summary.json` -- a file that on a first run did not
    # exist yet, because embed runs before aggregate.
    video_units = 0
    if "summary" in ran and index:
        from ..video_rag import driver as video_rag
        video_units = video_rag.index_video_summary(
            video_id, load(video_id, "summary").payload, embedder, index)

    return Produced(
        video_id=video_id, component="aggregate",
        backend=",".join(sinks.parse(sink)),
        artifacts=produced,
        stats={"tier": tier, "ran": len(ran), "current": current,
               "computed": len(ran) - current, "aggregates": ran,
               "models": sorted(models), "video_units": video_units,
               "skipped": skipped},
        skipped=sorted(skipped),
    )


def load(video_id: str, name: str) -> Aggregate:
    path = paths.artifact(video_id, "aggregates") / f"{name}.json"
    return Aggregate.from_dict(sinks.read_json(path))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Higher-level answers over what video_rag extracted.")
    ap.add_argument("video_id")
    ap.add_argument("--tier", default="free", choices=TIERS,
                    help="a cost ceiling; cheaper tiers still run")
    ap.add_argument("--only", default=None, help="comma-separated aggregator names")
    ap.add_argument("--force", action="store_true", help="rebuild what is current")
    ap.add_argument("--sink", default="file", help="file | supabase | both")
    ap.add_argument("--llm", default=None,
                    help="who answers the llm tier: a provider or provider/model; "
                         "default FALCONVAR_LLM, then openai")
    ap.add_argument("--embedder", default=None,
                    help="who embeds for it: a provider or provider/model; "
                         "default FALCONVAR_EMBEDDER, then openai")
    ap.add_argument("--index", default=None,
                    help="`supabase` also stores the summary as the video's vector")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    only = ([n.strip() for n in args.only.split(",") if n.strip()]
            if args.only else None)
    try:
        produced = run(args.video_id, args.tier, only, args.force, args.sink,
                       args.llm, args.embedder, args.index)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    s = produced.stats
    print(f"{produced.video_id}   tier={s['tier']}")
    print(f"  ran          {s['ran']}   ({s['computed']} computed, "
          f"{s['current']} reused)   -> {produced.backend}")
    for name in s["aggregates"]:
        print(f"    {name}")
    for name, why in s["skipped"].items():
        print(f"    {name:<12} -- skipped: {why}")
    if s["video_units"]:
        print(f"  video vector  {s['video_units']}   -> video_embeddings")
    print(f"\naggregates -> {paths.artifact(produced.video_id, 'aggregates')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
