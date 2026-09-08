"""The pipeline in terms a request can supply.

**Every component has the same signature**, so this is a dispatch table rather
than one function per stage. `falconvar`'s API needed a separate route and a
separate handler for describe, embed and aggregate because those stages took
different arguments and returned different things; here they all take a
`video_id` and keyword arguments and return a `Produced`, so one route serves
all of them and adding a component adds a row.

Nothing here does pipeline work. It resolves names to callables, validates what
a request asked for against the registries, and reads what is on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from ver3 import aggregate, audio, boundaries, cut, describe, media, video
from ver3 import workflow
from ver3.describe import prompts
from ver3.rag import embed, retrieve
from ver3.shared import paths, sinks
from ver3.shared.documents import Produced
from ver3.video import samplers as samplers_mod

#: Where an upload is parked until a run reads it.
UPLOADS = paths.UPLOADS


def _boundaries_evidence(video_id: str, **kwargs) -> Produced:
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
    "aggregate": aggregate.run,
}


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
    aggregates = []
    directory = paths.artifact(video_id, "aggregates")
    if directory.exists():
        aggregates = [{"name": p.stem,
                       "about": aggregate.ABOUT.get(p.stem, ""),
                       "url": f"/videos/{video_id}/aggregates/{p.stem}"}
                      for p in sorted(directory.glob("*.json"))]
    return {"video_id": video_id, "documents": documents,
            "aggregates": aggregates,
            "frames": f"/videos/{video_id}/frames/{{index}}"
                      if "store" in present else None}


def frame_path(video_id: str, index: int) -> Path:
    path = paths.artifact(video_id, "store") / f"{index:07d}.jpg"
    if not path.exists():
        raise FileNotFoundError(
            f"{video_id} has no frame {index} -- the manifest names the frames "
            f"that exist")
    return path


def search(query: str, video_id: str, **params) -> list[dict[str, Any]]:
    moments, notes = retrieve.search(query, video_id, **params)
    return [{**m.as_dict(), "notes": notes} for m in moments]


# ------------------------------------------------------------ capabilities

def available() -> dict[str, Any]:
    """What this deployment can be asked for, read from the registries.

    So a sampler, an index or an aggregator added to `ver3` appears here
    without anyone editing a list -- and the defaults are read off
    `workflow.Options` rather than restated, because a form that offers a
    registry in alphabetical order defaults to `stub` and produces a run that
    looks complete and says nothing.
    """
    from ver3.audio import models as audio_models

    return {
        "components": list(workflow.COMPONENTS),
        "samplers": samplers_mod.available(),
        "prompts": prompts.questions(),
        "pairings": ["uniform:overview", "uniform:text", "yolo:overview"],
        "policies": sorted(boundaries.POLICIES),
        "describers": describe.available(),
        "embedders": embed.available(),
        "indexes": embed.indexes.available(),
        "sinks": list(sinks.BACKENDS),
        "transcribers": sorted(audio_models.TRANSCRIBERS),
        "diarizers": sorted(audio_models.DIARIZERS),
        "aggregators": {name: {"tier": aggregate.TIER_OF[name],
                               "about": aggregate.about(name)}
                        for name in aggregate.available()},
        "tiers": list(aggregate.TIERS),
        "artifacts": dict(ARTIFACTS),
        "defaults": {
            "policy": workflow.Options.policy,
            "sampler": workflow.Options.sampler,
            "describer": workflow.Options.describer,
            "embedder": workflow.Options.embedder,
            "index": workflow.Options.index,
            "tier": workflow.Options.tier,
            "sink": workflow.Options.sink,
        },
    }


__all__ = ["ARTIFACTS", "COMPONENTS", "UPLOADS", "artifact", "available",
           "exports", "frame_path", "run_component", "run_workflow", "search",
           "videos"]
