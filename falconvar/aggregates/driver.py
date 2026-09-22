"""The aggregates driver: what video_rag extracted -> `aggregates/<answer>.json`.

Each answer writes its own file, so a run that dies partway leaves the results
it did produce rather than none -- the same reason tiers run cheapest first.

**An answer is one aggregator over one input.** `summary` over its default
input is `summary.json`. `--input summary=clip:hazards[severity],clip:hazards[hazards]`
is two answers, `summary~severity` and `summary~hazards`. A link profile's id
carries a colon, `entities:people`, and Windows refuses one in a filename, so on
disk it is `entities.people.json`.

**Reaches video_rag through its driver, never its components.** Documents,
the question vocabulary, an embedder and the whole-video vector all come from
`video_rag.driver`, imported inside the functions that need them so importing
this package loads nothing from the other tier. video_rag, for its part, never
reads what this writes: the summary is handed to it, not left for it to find.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..shared import paths
from ..shared.contracts.documents import Aggregate, Produced, fingerprint_of
from ..shared.storage import sinks
from . import available, build, definitions, expand, kind_of, takes_inputs, tier_of
from .base import TIERS, Context, missing
from .inputs import (InputError, answer_id, answer_of_file, check, filename,
                     labels, parse)
from .inputs import DEFAULT as DEFAULT_INPUT


def context_for(video_id: str) -> Context:
    """Every document video_rag wrote for this video. Missing ones are None."""
    from ..video_rag import driver as video_rag

    found = video_rag.documents(video_id)
    return Context(video_id, found["timeline"], found["manifest"],
                   found["descriptions"], found["transcript"])


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


def parse_inputs(inputs: Any) -> dict[str, str]:
    """`{aggregator: selection}`, from a dict or `name=selection;name=selection`.

    The string form is what a CLI flag or a form field hands over. Split on the
    first `=`, so a label inside the selection survives.
    """
    if not inputs:
        return {}
    if isinstance(inputs, dict):
        return {str(k).strip(): str(v).strip() for k, v in inputs.items()}
    out: dict[str, str] = {}
    for part in str(inputs).split(";"):
        if not part.strip():
            continue
        name, sep, selection = part.partition("=")
        if not sep:
            raise InputError(f"{part.strip()!r}: an input is `aggregator=selection`")
        out[name.strip()] = selection.strip()
    return out


def default_selection(name: str) -> str:
    return (DEFAULT_INPUT if kind_of(name) is None
            else definitions.default_selection(name))


def _plan_problems(tier: str, inputs: Any, only: Any) -> list[str]:
    """Everything wrong with what a run asks for, short of who answers it."""
    if tier not in TIERS:
        return [f"tier must be one of {', '.join(TIERS)}"]
    try:
        selections = parse_inputs(inputs)
    except InputError as exc:
        return [str(exc)]
    known = set(available())
    wanted = expand(only)
    problems = [f"unknown aggregator {name!r}; known: {', '.join(available())}"
                for name in wanted if name not in known]
    vocabulary = None
    for name, selection in selections.items():
        if name not in known:
            problems.append(f"an input for {name!r}, which is not an aggregator")
            continue
        if not takes_inputs(name):
            problems.append(f"{name} counts what extraction produced; it reads no input")
            continue
        if name not in wanted:
            problems.append(f"an input for {name}, which `only` leaves out")
        elif TIERS.index(tier_of(name)) > TIERS.index(tier):
            problems.append(f"an input for {name}, a {tier_of(name)} aggregator; "
                            f"this run stops at {tier}")
        try:
            parsed = parse(selection)
            labels(parsed)
        except InputError as exc:
            problems.append(f"{name}: {exc}")
            continue
        if vocabulary is None:
            from ..video_rag import driver as video_rag
            vocabulary = video_rag.vocabulary()
        problems += [f"{name}: {p}" for p in check(parsed, vocabulary)]
        if kind_of(name) == "link":
            for one in parsed:
                try:
                    definitions.selection(definitions.locate(name)[1], one)
                except InputError as exc:
                    problems.append(f"{name}: {exc}")
    return problems


def validate(tier: str = "free", llm: Optional[str] = None, inputs: Any = None,
             only: Any = None, embedder: Optional[str] = None) -> list[str]:
    """What stops a run before it starts, as messages."""
    problems = _plan_problems(tier, inputs, only)
    if problems or tier != "llm":
        return problems
    from ..shared.models import providers
    problems += providers.problems("llm", llm)
    if any(kind_of(n) == "link" for n in expand(only)):
        problems += providers.problems("embed", embedder)
    return problems


def run(video_id: str, tier: str = "free",
        only: Optional[Sequence[str] | str] = None,
        force: bool = False,
        sink: str | Sequence[str] = "file",
        llm: Optional[str] = None,
        embedder: Optional[str] = None,
        index: Optional[str] = None,
        inputs: Optional[dict[str, str] | str] = None) -> Produced:
    """Run every aggregator up to ``tier``, cheapest first.

    `inputs` is `{aggregator: selection}` -- or `name=selection;...` -- and an
    aggregator not named reads its own default. `llm` is who answers the paid
    tier and `embedder` who embeds for the link profiles, each a provider or
    `provider/model`. `index` naming `supabase` also stores the summary as the
    video's vector, which `/search level=video` ranks.
    """
    problems = _plan_problems(tier, inputs, only)
    if problems:
        raise ValueError("; ".join(problems))
    selections = parse_inputs(inputs)
    ceiling = TIERS.index(tier)
    names = sorted((n for n in expand(only) if TIERS.index(tier_of(n)) <= ceiling),
                   key=lambda n: TIERS.index(tier_of(n)))
    context = context_for(video_id)
    whole = context.inputs_fingerprint()
    grid = context.timeline.fingerprint()

    directory = paths.artifact(video_id, "aggregates")
    backends = sinks.parse(sink)
    produced: dict[str, str] = {}
    skipped: dict[str, str] = {}
    ran: list[str] = []
    current = 0
    models: set[str] = set()
    used: dict[str, str] = {}

    for name in names:
        aggregator = build(name, llm, embedder)
        why = missing(aggregator, context)
        if why is not None:
            skipped[name] = why
            continue
        author = getattr(aggregator, "model_key", None)

        # (answer id, input, what it read, the fingerprint a stored copy needs)
        answers: list[tuple[str, Any, Any, str]] = []
        if not takes_inputs(name):
            answers.append((name, None, None, whole))
        else:
            parsed = parse(selections.get(name) or default_selection(name))
            for one, label in zip(parsed, labels(parsed)):
                answer = answer_id(name, label)
                read = aggregator.read(context, one)
                if read.empty:
                    skipped[answer] = read.why_empty
                    continue
                # What was read and what it was asked with -- never what was
                # merely available, so a summary of the transcript is not
                # rebuilt because a description changed.
                answers.append((answer, one, read, fingerprint_of({
                    "timeline": grid, "read": read.fingerprint(),
                    "version": aggregator.version})))

        for answer, one, read, expected in answers:
            path = directory / filename(answer)

            # The fingerprint governs whether to RECOMPUTE, not whether to
            # write. A run that adds a backend has nothing to recompute and
            # everything to write -- exactly the bug `embed` had across two
            # indexes. And the model is part of "current": same text, different
            # model is a different answer the caller asked for.
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
                payload = (aggregator.run(context) if read is None
                           else aggregator.run(context, read))
                stats: dict[str, Any] = {"about": aggregator.about,
                                         **({"model": author} if author else {})}
                if read is not None:
                    stats.update(inputs=str(one), version=aggregator.version,
                                 read_chars=read.chars)
                document = Aggregate(video_id=video_id, aggregate_id=answer,
                                     tier=aggregator.tier, payload=payload,
                                     inputs_fingerprint=expected, stats=stats)
            if author:
                models.add(author)
            if kind_of(name) is not None:
                used[name] = aggregator.version

            # Not `sinks.write`: that resolves one path per artifact name, and
            # each answer writes its own file under `aggregates/`. The row half
            # goes through the same writer every other component uses.
            if "file" in backends:
                produced[answer] = str(sinks.write_json(path, document.as_dict())
                                       if stored is None else path)
            if "supabase" in backends:
                from ..shared.storage import rows
                rows.WRITERS["aggregate"](video_id, document.as_dict())
                produced.setdefault(answer, "aggregate@supabase")
            ran.append(answer)

    recorded = _record_definitions(used, backends)

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
        backend=",".join(backends),
        artifacts=produced,
        stats={"tier": tier, "ran": len(ran), "current": current,
               "computed": len(ran) - current, "aggregates": ran,
               "models": sorted(models), "video_units": video_units,
               "skipped": skipped, **recorded},
        skipped=sorted(skipped),
    )


def _record_definitions(used: dict[str, str], backends: Sequence[str]) -> dict[str, Any]:
    """Provenance for Postgres: what each definition said at the version used.

    Reported, never raised -- the answers already landed, and a missing table
    must read as a missing table rather than as a run with nothing to record.
    """
    if "supabase" not in backends or not used:
        return {}
    from ..shared.storage import rows
    entries = []
    for name, version in sorted(used.items()):
        section, definition = definitions.locate(name)
        entry = definitions.get(section, definition)
        entries.append({"name": name, "version": version, "kind": kind_of(name),
                        "definition": {k: v for k, v in entry.items() if k != "builtin"},
                        "builtin": bool(entry.get("builtin"))})
    try:
        return {"definitions_recorded": rows.write_definitions(entries)}
    except Exception as exc:                              # noqa: BLE001
        return {"definitions_recorded": 0,
                "definitions_error": f"{type(exc).__name__}: {exc}"[:300]}


def load(video_id: str, name: str) -> Aggregate:
    path = paths.artifact(video_id, "aggregates") / filename(name)
    return Aggregate.from_dict(sinks.read_json(path))


def answers(video_id: str) -> list[str]:
    """Every answer this video has on disk, by id."""
    directory = paths.artifact(video_id, "aggregates")
    if not directory.exists():
        return []
    return [answer_of_file(p.stem) for p in sorted(directory.glob("*.json"))]


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Higher-level answers over what video_rag extracted.")
    ap.add_argument("video_id", nargs="?")
    ap.add_argument("--tier", default="free", choices=TIERS,
                    help="a cost ceiling; cheaper tiers still run")
    ap.add_argument("--only", default=None,
                    help="comma-separated ids, e.g. summary,entities:people; "
                         "`entities` means every link profile")
    ap.add_argument("--input", action="append", default=[], metavar="NAME=SELECTION",
                    help="repeatable. What one aggregator reads, e.g. "
                         "summary=transcript+clip:activity or "
                         "summary=clip:hazards[severity],clip:hazards[hazards]")
    ap.add_argument("--force", action="store_true", help="rebuild what is current")
    ap.add_argument("--sink", default="file", help="file | supabase | both")
    ap.add_argument("--llm", default=None,
                    help="who answers the llm tier: a provider or provider/model; "
                         "default FALCONVAR_LLM, then openai")
    ap.add_argument("--embedder", default=None,
                    help="who embeds for the link profiles: a provider or "
                         "provider/model; default FALCONVAR_EMBEDDER, then openai")
    ap.add_argument("--index", default=None,
                    help="`supabase` also stores the summary as the video's vector")
    ap.add_argument("--list", action="store_true",
                    help="every aggregator, its tier and what it reads by default")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for name in available():
            kind = kind_of(name) or "code"
            reads = default_selection(name) if takes_inputs(name) else "-"
            from . import about
            print(f"{name:<22} {tier_of(name):<6} {kind:<6} {reads:<16} {about(name)}")
        for where, found in definitions.load()["problems"].items():
            print(f"  ignored {where}: {'; '.join(found)}")
        return 0
    if not args.video_id:
        ap.error("video_id is required unless --list")

    try:
        produced = run(args.video_id, args.tier, args.only, args.force, args.sink,
                       args.llm, args.embedder, args.index, ";".join(args.input))
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
        print(f"    {name:<22} -- skipped: {why}")
    if s["video_units"]:
        print(f"  video vector  {s['video_units']}   -> video_embeddings")
    print(f"\naggregates -> {paths.artifact(produced.video_id, 'aggregates')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
