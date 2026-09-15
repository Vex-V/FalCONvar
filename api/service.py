"""The pipeline in terms a request can supply.

Every component has the same signature, so `COMPONENTS` is a dispatch table
rather than one function per stage and a single route serves all of them.

Nothing here does pipeline work: it resolves names to callables, validates
against the registries, and reads what is on disk.

`available()` reads the registries and the defaults off `workflow.Options`, so
a sampler or an index added to the package appears in `/capabilities` without
anyone editing a list -- and a form built from it cannot default to whichever
option sorts first.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from falconvar import aggregates, workflow
from falconvar.video_rag import (audio, boundaries, cut, describe, embed, media,
                                 retrieve, video)
from falconvar.video_rag.describe import library, prompts
from falconvar.shared import paths
from falconvar.shared.models import providers
from falconvar.shared.storage import sinks
from falconvar.shared.contracts.documents import Produced
from falconvar.video_rag.video import samplers as samplers_mod

#: Where an upload is parked until a run reads it.
UPLOADS = paths.UPLOADS


@functools.wraps(boundaries.evidence)
def _boundaries_evidence(video_id: str, **kwargs) -> Produced:
    """`evidence` returns None when a policy needs none; a route needs a
    `Produced` either way, so the skip is reported rather than absent.

    `functools.wraps` carries the wrapped signature through, so `parameters()`
    publishes what `evidence` actually takes instead of the `**kwargs` this
    wrapper is written with.
    """
    produced = boundaries.evidence(video_id, **kwargs)
    if produced is None:
        return Produced(video_id=video_id, component="boundaries.evidence",
                        backend="none", stats={"needed": False},
                        skipped=["evidence"])
    return produced


#: component name -> the callable a request can invoke. The only place that
#: knows a component's public entry point, and the reason one route runs any of
#: them. Order matches `workflow.COMPONENTS`.
COMPONENTS: dict[str, Callable[..., Produced]] = {
    "audio": audio.run,
    "boundaries.evidence": _boundaries_evidence,
    "boundaries": boundaries.run,
    "video": video.run,
    "cut": cut.run,
    "describe": describe.run,
    "embed": embed.run,
    "aggregate": aggregates.run,
}


def register(source: Path, video_id: str,
             sink: str | Sequence[str] = "file") -> Produced:
    """Read what streams the file carries, and nothing else.

    The one component a caller cannot reach through `run/{component}`, because
    until it has run there is no video id to address. Everything after it is
    the caller's to sequence.
    """
    return media.run(source, video_id, sink)


def parameters() -> dict[str, Any]:
    """Every component's tunable parameters, read off its signature.

    Introspected rather than listed, for the reason `defaults` is read off
    `workflow.Options`: a restated list is a second copy to keep in step, and
    when it drifts a form offers a parameter the component does not take or
    hides one it does.

    `video_id` is omitted -- it is the address, not a setting. Types are the
    annotation as written, which is what a form needs to pick a widget.
    """
    import inspect

    out: dict[str, Any] = {}
    for name, fn in COMPONENTS.items():
        fields = []
        for arg, param in inspect.signature(fn).parameters.items():
            if arg in ("video_id", "self") or arg.startswith("*"):
                continue
            fields.append({
                "name": arg,
                "type": (param.annotation if isinstance(param.annotation, str)
                         else getattr(param.annotation, "__name__", "any")),
                "default": (None if param.default is inspect.Parameter.empty
                            else param.default),
                "required": param.default is inspect.Parameter.empty,
            })
        out[name] = fields
    return out


def run_component(name: str, video_id: str, **params) -> Produced:
    """Run one component. Raises KeyError for an unknown name."""
    if name not in COMPONENTS:
        raise KeyError(f"unknown component {name!r}; "
                       f"known: {', '.join(COMPONENTS)}")
    if not paths.exists(video_id, "media"):
        raise FileNotFoundError(
            f"{video_id} has no media.json -- upload it first, or it is not a "
            f"video this deployment has seen")
    return COMPONENTS[name](video_id, **params)


def run_workflow(options: workflow.Options,
                 on_step: Optional[Callable] = None) -> workflow.Run:
    return workflow.process(options, on_step=on_step)


# ----------------------------------------------------------------- reading

#: Artifact -> one line on what it holds. Published so a client can label a
#: download without hard-coding the list.
ARTIFACTS: dict[str, str] = {
    "media": "what the file is: the two streams and their addressing",
    "raw_transcript": "words, segments and speaker turns, before any grid",
    "cuts": "boundary evidence, and the score series it was thresholded from",
    "timeline": "THE GRID: every chunk's span, and the policy that chose it",
    "manifest": "which frames were kept, by which sampler, and why",
    "transcript": "what was said, cut to the grid",
    "descriptions": "one model answer per (chunk, sampler)",
    "embedded": "the text that went into the index, without the vectors",
}


def videos() -> list[dict[str, Any]]:
    """Every video with an output directory.

    Read from disk rather than remembered, so a restarted server still knows
    everything it produced.
    """
    out = []
    for video_id in paths.videos():
        present = paths.present(video_id)
        entry: dict[str, Any] = {"video_id": video_id, "artifacts": present}
        if "media" in present:
            described = media.load(video_id)
            entry.update(duration_s=described.duration_s,
                         has_video=described.has_video,
                         has_audio=described.has_audio)
        if "timeline" in present:
            grid = boundaries.load(video_id)
            entry.update(policy=grid.policy, chunks=len(grid),
                         timeline_fingerprint=grid.fingerprint())
        out.append(entry)
    return out


def artifact(video_id: str, name: str) -> dict[str, Any]:
    """One document, as parsed JSON."""
    path = paths.artifact(video_id, name)
    if not path.exists():
        raise FileNotFoundError(f"{video_id} has no {name}")
    return sinks.read_json(path)


def exports(video_id: str) -> dict[str, Any]:
    """What this video can hand over, read from disk.

    Only what exists, never what could exist: an audio-only video advertises no
    manifest rather than offering a link that 404s, because a broken link reads
    as breakage rather than as a stage that never ran.
    """
    present = set(paths.present(video_id))
    documents = [{"name": name, "about": about,
                  "url": f"/videos/{video_id}/artifacts/{name}"}
                 for name, about in ARTIFACTS.items() if name in present]
    listed = []
    directory = paths.artifact(video_id, "aggregates")
    if directory.exists():
        listed = [{"name": p.stem,
                   "about": aggregates.ABOUT.get(p.stem, ""),
                   "url": f"/videos/{video_id}/aggregates/{p.stem}"}
                  for p in sorted(directory.glob("*.json"))]
    return {"video_id": video_id, "documents": documents,
            "aggregates": listed,
            "frames": f"/videos/{video_id}/frames/{{index}}"
                      if "store" in present else None}


def frame_path(video_id: str, index: int) -> Path:
    path = paths.artifact(video_id, "store") / f"{index:07d}.jpg"
    if not path.exists():
        raise FileNotFoundError(
            f"{video_id} has no frame {index} -- the manifest names the frames "
            f"that exist")
    return path


def search(query: str, video_id: Any = None, **params) -> dict[str, Any]:
    """Moments, and the notes about how they were ranked.

    The notes are top-level as well as on each moment. Carried only on the
    moments, an empty result had nowhere to put them -- so "nothing matched
    those filters", the one answer an empty result exists to give, reached the
    caller as a bare `[]`.
    """
    moments, notes = retrieve.search(query, video_id, **params)
    return {"moments": [{**m.as_dict(), "notes": notes} for m in moments],
            "notes": notes}


def search_videos(query: str, **params) -> list[dict[str, Any]]:
    """Which video, rather than which moment. See `retrieve.videos`."""
    return retrieve.videos(query, **params)


# ------------------------------------------------------------ capabilities

#: What `/search` narrows by. Published rather than restated in a client, for
#: the same reason `parameters` is introspected: a form built from a second
#: copy offers a filter the route does not take, or hides one it does.
SEARCH_FILTERS: list[dict[str, Any]] = [
    {"name": "video_ids", "type": "list[str]",
     "about": "which videos to search. Omit for every video; one id, three or "
              "all is the same question over a different set"},
    {"name": "level", "type": "str",
     "about": "`moment` (default) ranks chunks; `video` ranks whole videos by "
              "their summary, from `video_embeddings`"},
    {"name": "sampler", "type": "str",
     "about": "one PAIRING, e.g. `clip:text`"},
    {"name": "question", "type": "str",
     "about": "one question across every sampler that asked it, e.g. `text`"},
    {"name": "strategy", "type": "str",
     "about": "one sampler's whole output, e.g. `clip`. Not a prefix of "
              "`sampler`: a bare id means the question IS the strategy name"},
    {"name": "chunk_ids", "type": "list[int]",
     "about": "the drill-down: search, read the ids back, ask for more"},
    {"name": "window", "type": "int",
     "about": "widen `chunk_ids` by N neighbours each side"},
    {"name": "after", "type": "float",
     "about": "seconds. Resolved to chunk ids through the grid"},
    {"name": "before", "type": "float", "about": "seconds"},
    {"name": "structured", "type": "dict",
     "about": "exact structured values. Only meaningful where a shape fixed "
              "the vocabulary with `one_of`"},
    {"name": "candidates", "type": "int",
     "about": "units ranked per half before fusion. Measured: 5 truncates the "
              "fusion, 20 and 100 agree"},
]


def filterable() -> dict[str, list[str]]:
    """Structured fields whose values are a vocabulary, and what it is.

    The only fields worth offering as a filter. A free-text field is
    filterable in the mechanical sense and useless in practice -- one video
    produced `cashier`, `customer` and `cashier or customer near checkout`, and
    a filter for the first matched all three. `one_of` is what makes the
    difference, so this reads the shapes rather than guessing.
    """
    out: dict[str, list[str]] = {}
    for shape in library.shapes().values():
        for field, spec in (shape.get("fields") or {}).items():
            values = spec.get("enum")
            if values is None:
                items = spec.get("items")
                values = items.get("enum") if isinstance(items, dict) else None
            if values:
                out.setdefault(field, sorted(set(out.get(field, [])) | set(values)))
    return out


def available() -> dict[str, Any]:
    """What this deployment can be asked for, read from the registries.

    So a sampler, an index or an aggregator added to `falconvar` appears here
    without anyone editing a list -- and the defaults are read off
    `workflow.Options` rather than restated, because a form that offers a
    registry in alphabetical order defaults to `stub` and produces a run that
    looks complete and says nothing.
    """
    from falconvar.video_rag.audio import models as audio_models

    return {
        "components": list(workflow.COMPONENTS),
        "samplers": samplers_mod.available(),
        "prompts": prompts.questions(),
        "shapes": sorted(library.shapes()),
        "pairings": ["uniform:overview", "uniform:text", "yolo:overview",
                     "clip:[text,scene]", "clip:text+scene"],
        "policies": sorted(boundaries.POLICIES),
        "describers": describe.available(),
        "embedders": embed.available(),
        "llms": providers.names("llm"),
        # Every provider, whether it can run here and why not, and the
        # variables a default is read from. Names of keys, never keys.
        "models": providers.catalog(),
        "indexes": embed.indexes.available(),
        "sinks": list(sinks.BACKENDS),
        "transcribers": sorted(audio_models.TRANSCRIBERS),
        "diarizers": sorted(audio_models.DIARIZERS),
        "aggregators": {name: {"tier": aggregates.TIER_OF[name],
                               "about": aggregates.about(name)}
                        for name in aggregates.available()},
        "tiers": list(aggregates.TIERS),
        "artifacts": dict(ARTIFACTS),
        # What a search may narrow by, and which structured values are a
        # vocabulary rather than free text.
        "search": {"filters": SEARCH_FILTERS,
                   "structured_fields": filterable(),
                   "levels": ["moment", "video"]},
        # What each component may be tuned with, for a form that configures a
        # run stage by stage rather than accepting the workflow's defaults.
        "parameters": parameters(),
        "defaults": {
            "policy": workflow.Options.policy,
            "sampler": workflow.Options.sampler,
            "index": workflow.Options.index,
            "tier": workflow.Options.tier,
            "sink": workflow.Options.sink,
            # Resolved now rather than read off the dataclass, whose fields
            # are None until a run resolves them: a form defaulting to None
            # would show nothing where the real answer is `openai`.
            **providers.defaults(),
        },
    }


# ------------------------------------------------------------------- prompts

def prompt_list() -> dict[str, Any]:
    """Every question, marked built-in or custom, with the shape it answers in.

    The shape is resolved rather than just named, so a caller sees which keys
    an answer will carry without having to fetch the shape separately.
    """
    builtin = library.builtin_shapes()
    entries = []
    for name in library.questions():
        entry = library.question(name)
        entries.append({
            "name": name,
            "builtin": bool(entry.get("builtin")),
            "shape": entry.get("shape"),
            "fields": library.fields_of(name),
            "about": entry.get("about", ""),
            "instruction": entry.get("instruction", ""),
        })
    return {
        "prompts": entries,
        "shapes": {name: {"fallback": bool(shape.get("fallback")),
                          "builtin": name in builtin,
                          "fields": sorted(shape.get("fields") or {}),
                          "summary": shape.get("summary"),
                          "identity": library.shape_identity(name)}
                   for name, shape in sorted(library.shapes().items())},
        "field_types": list(library.FIELD_TYPES),
        "limits": {"fields": library.MAX_FIELDS,
                   "nested_keys": library.MAX_NESTED_KEYS,
                   "one_of": library.MAX_ENUM},
        "custom_file": str(paths.PROMPTS),
    }


def prompt_get(name: str) -> dict[str, Any]:
    """One question, with the exact response schema a call would be given.

    Exact, not indicative: a call's schema depends on nothing but its question.
    """
    entry = library.question(name)
    return {
        "name": name,
        "builtin": bool(entry.get("builtin")),
        "shape": entry.get("shape"),
        "fields": library.fields_of(name),
        "about": entry.get("about", ""),
        "instruction": entry.get("instruction", ""),
        "schema": prompts.schema_for(name),
        "version": prompts.version_of(name),
    }


def prompt_add(name: str, instruction: str, shape: str = "scene",
               about: str = "", fields: Optional[dict[str, Any]] = None,
               summary: str = "standard") -> dict[str, Any]:
    """Add or replace a custom question. Built-ins are refused.

    ``fields`` makes the question carry its own shape rather than naming a
    shipped one; the shape is then stored under the question's own name.
    """
    library.add(name, instruction, shape=shape, about=about,
                fields=fields, summary=summary)
    return prompt_get(name)


def prompt_remove(name: str) -> None:
    library.remove(name)


__all__ = ["ARTIFACTS", "COMPONENTS", "UPLOADS", "artifact", "available",
           "parameters", "register",
           "exports", "frame_path", "prompt_add", "prompt_get", "prompt_list",
           "prompt_remove", "run_component", "run_workflow", "search",
           "search_videos", "videos"]
