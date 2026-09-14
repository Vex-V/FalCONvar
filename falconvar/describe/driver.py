"""The describe component: `manifest.json` + `store/` -> `descriptions.json`."""

from __future__ import annotations

from typing import Optional, Sequence

from ..boundaries import load as load_timeline
from ..video import load as load_manifest
from ..shared import env, paths, sinks
from ..shared.documents import Descriptions, Produced
from . import base, library, prompts
from .backends import stub  # noqa: F401  -- self-registers
from .frames import FrameSource, StoreUnavailable

def run(video_id: str, describer: Optional[str] = None,
        model: Optional[str] = None,
        samplers: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        resume: bool = True,
        sink: str | Sequence[str] = "file") -> Produced:
    """One call per (chunk, sampler). The expensive stage.

    `describer` is a provider or `provider/model`; None resolves through
    `shared.providers` -- FALCONVAR_DESCRIBER, then openai.
    """
    env.load()
    manifest = load_manifest(video_id)
    timeline = load_timeline(video_id)

    if manifest.timeline_fingerprint != timeline.fingerprint():
        raise ValueError(
            f"{video_id}: the manifest was built on a different grid "
            f"({manifest.timeline_fingerprint} vs {timeline.fingerprint()}). "
            "Re-run ingest against the current timeline.")

    known = prompts.questions()
    unknown = sorted({q for s in manifest.config.get("samplers", [])
                      for q in prompts.questions_of(s, s.get("id", ""))
                      if q not in known})
    if unknown:
        raise ValueError(
            f"manifest names unknown question(s) {', '.join(unknown)}; "
            f"known: {', '.join(prompts.questions())}")

    existing = None
    if resume and paths.exists(video_id, "descriptions"):
        existing = load(video_id)

    built = base.build(describer, model=model)
    from .reader import describe

    with FrameSource(video_id, manifest) as source:
        document = describe(manifest, timeline, built, source,
                            samplers, existing, limit)

    written = sinks.write(video_id, "descriptions", document.as_dict(), sink)
    return Produced(
        video_id=video_id, component="describe", backend=",".join(written),
        artifacts={"descriptions": written.get("file", "")},
        stats={**document.stats, "describer": built.name,
               "model": (document.model.get("params") or {}).get("model", built.name),
               **_record_prompts(document.model.get("prompts") or {}, sink)},
    )


def _record_prompts(versions: dict[str, str],
                    sink: str | Sequence[str]) -> dict[str, object]:
    """Append what each question said, at the version this run asked it under.

    `descriptions.model` already records `{question: hash}`, which lets a
    reader *detect* that an answer came from a different prompt version. It
    cannot recover what that version said -- edit an instruction and the old
    text is gone -- so the row is what makes a description's provenance
    readable rather than merely comparable.

    Best-effort and after the descriptions are written, exactly as the Postgres
    half of `sinks.write` is: this is provenance, and losing it must not fail a
    stage that has already paid for its answers.

    But the failure is **reported**, never swallowed. Returning a bare 0 made a
    missing column read exactly like a run with nothing to record -- and the
    first version of this did precisely that, hiding a `PGRST204` behind a
    number that looked ordinary. `sinks.write` takes the same stance: continue,
    and say what went wrong.
    """
    if "supabase" not in sinks.parse(sink) or not versions:
        return {"prompts_recorded": 0}
    from ..shared import rows
    entries = []
    for name, version in sorted(versions.items()):
        entry = library.load()["questions"].get(name) or {}
        shape = library.shape_of(name)
        entries.append({
            "name": name, "version": version,
            "instruction": library.instruction_of(name),
            "shape": shape,
            "summary": shape.get("summary", "standard"),
            "builtin": bool(entry.get("builtin")),
            "about": entry.get("about") or None,
        })
    try:
        return {"prompts_recorded": rows.write_prompts(entries)}
    except Exception as exc:                              # noqa: BLE001
        return {"prompts_recorded": 0,
                "prompts_error": f"{type(exc).__name__}: {exc}"[:300]}


def load(video_id: str) -> Descriptions:
    return Descriptions.from_dict(
        sinks.read_json(paths.artifact(video_id, "descriptions")))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Describe every (chunk, sampler).")
    ap.add_argument("video_id")
    ap.add_argument("--describer", default=None,
                    help="a provider or provider/model; default "
                         f"FALCONVAR_DESCRIBER, then openai. Known: "
                         f"{', '.join(base.available())}")
    ap.add_argument("--model", default=None,
                    help="blank: the provider's default chat model")
    ap.add_argument("--sampler", default=None,
                    help="comma-separated subset to describe")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N calls. Costs money, so this exists")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--sink", default="file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    samplers = ([s.strip() for s in args.sampler.split(",") if s.strip()]
                if args.sampler else None)
    try:
        produced = run(args.video_id, args.describer, args.model, samplers,
                       args.limit, not args.no_resume, args.sink)
    except (KeyError, ValueError, FileNotFoundError, StoreUnavailable,
            base.DescriberUnavailable, sinks.UnknownBackend) as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        print(json.dumps(produced.as_dict(), indent=2))
        return 0

    s = produced.stats
    print(f"{produced.video_id}   {s['describer']} ({s['model']})")
    print(f"  described    {s['described']}")
    print(f"  skipped      {s['skipped']}   (already current)")
    print(f"  chunks       {s['chunks']}")
    print(f"  elapsed      {s['elapsed_s']:.2f}s")
    print(f"\ndescriptions -> {produced.artifacts['descriptions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
