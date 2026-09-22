"""Both ways to run FalCONvar, side by side, on the same video.

    python example.py [path/to/video.mp4]

Run as a script, this does each in turn and compares what they produced. The
point is that they are the same eight steps -- `video_rag()` is a loop over the
component calls, not a different engine -- and that the difference is *tuning*,
not capability.

**Level 1 -- the components.** Every one is `describe.run(video_id, ...) ->
Produced`, and `describe.load(video_id)` reads the result back. Two directions,
not two steps: nothing below calls `load` to make the pipeline work, because
each `run` resolves its own inputs through the previous components' `load`.
That is why nothing is passed between these calls except an id and settings.

Per-stage tuning lives here and nowhere else: a scene threshold, a sampler's
`confidence`, an audio `compute_type`.

The module is the unit, always -- `embed.run`, never a bare `embed`. `load`
appears in six of the eight components and `build` in three, so a bare-name
style needs an alias the moment a caller wants a second thing from the same
module. `boundaries` is the clearest case: `evidence` finds candidate cuts by
decoding, `run` turns them into the grid, and `retune` re-thresholds the cached
scores without decoding at all -- three ways in, one of which is `run`.

**Level 2 -- the pipeline.** `video_rag()` takes one policy and one sampler and
uses defaults for the rest. It decides order and whether a step runs at all --
`use_audio` is and-ed with whether the file actually carries a soundtrack.

Neither needs a checkout: `falconvar.configure(data_root=...)` says where to
write, and without it an installed copy writes under `~/.falconvar`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Level 1: the components. One import line for all of them, and every call
# site says which component it is in -- `run` is the work, `load` reads the
# result back, and `boundaries` has two more ways in besides.
from falconvar.video_rag import (audio, boundaries, cut, describe, embed,
                                 media, retrieve, video)

# Level 2: the whole extraction, as one call.
from falconvar.video_rag import video_rag

SOURCE = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/test.mp4")

#: Cheap backends, so the example costs nothing to run. Swap for
#: `describer="openai"` and `embedder="local"` to do it for real.
DESCRIBER, EMBEDDER, INDEX = "stub", "hash", "qdrant"


def by_components(video_id: str) -> str:
    """The pipeline, written out. Eight calls, in dependency order.

    The order is not a convention -- `boundaries` under `vad` needs `audio` to
    have finished, under `scene` needs the evidence pass, and `video` and `cut`
    both need the grid. Get it wrong and a component raises `FileNotFoundError`
    naming the artifact it wanted, rather than answering something wrong.
    """
    # 1 · what the file is. The one component that takes a path, because it is
    #     the one that mints the id.
    first = media.run(SOURCE, video_id=video_id)

    # `Produced` is a receipt: what was written, and a few headline numbers.
    # These three are in it, so there is nothing to read back. Reach for
    # `media.load(video_id)` when you want the typed `Media` document instead.
    info = first.stats
    print(f"  media      {info['duration_s']:.1f}s  "
          f"audio={info['has_audio']} video={info['has_video']}")

    # 2 · the soundtrack, scanned whole. Tuned per stage, which is the reason
    #     to be down here rather than calling `video_rag()`.
    if info["has_audio"]:
        audio.run(video_id, transcriber="stub", diarizer="none")

    # 3 · boundary evidence. Returns None when the policy needs none -- that is
    #     an answer, not a failure.
    evidence = boundaries.evidence(video_id, "uniform")
    print(f"  evidence   {'skipped -- uniform is arithmetic' if evidence is None else evidence.stats}")

    # 4 · THE GRID. Everything downstream reads it; nothing edits it.
    grid = boundaries.run(video_id, "uniform", chunk_s=20.0, min_s=5.0)
    print(f"  boundaries {grid.stats['chunks']} chunks  "
          f"({grid.stats['shortest_s']:.1f}-{grid.stats['longest_s']:.1f}s)")

    # 5 · the picture, onto that grid.
    if info["has_video"]:
        kept = video.run(video_id, sampler="uniform", per_second=1.0, every_n=5)
        print(f"  video      {kept.stats['frames_sampled']} frames sampled")

    # 6 · the transcript, onto the same grid. Cheap and repeatable: Whisper
    #     timestamped every word, so re-cutting runs no model.
    if info["has_audio"]:
        cut.run(video_id)

    # 7 · one answer per (chunk, sampler:question).
    if info["has_video"]:
        describe.run(video_id, describer=DESCRIBER)

    # 8 · vectors, from both modalities.
    done = embed.run(video_id, embedder=EMBEDDER, index_name=INDEX)
    print(f"  embed      {done.stats['units']} units -> {done.stats['embedder']}")
    return video_id


def whole_pipeline(video_id: str) -> str:
    """The same eight steps, as one call.

    `on_step` is how a caller follows it: announced by name before a component
    runs, and again with its `Produced` after. A stage set only on completion
    names the *previous* component throughout the longest stage of the run.
    """
    def announce(component, produced):
        if produced is None:                      # before it runs
            print(f"  {component:<20} running...")
        else:                                     # after, with what it wrote
            print(f"  {component:<20} -> {', '.join(produced.artifacts) or '-'}")

    run = video_rag(SOURCE, video_id=video_id, policy="uniform",
                    sampler="uniform", describer=DESCRIBER,
                    embedder=EMBEDDER, index=INDEX, on_step=announce)
    if run.skipped:
        for name, why in run.skipped.items():
            print(f"  {name:<10} -- skipped: {why}")
    return run.video_id


def show(video_id: str) -> dict:
    """Read the results back. `load` is the other half of every component."""
    timeline = boundaries.load(video_id)
    described = describe.load(video_id)
    return {"chunks": len(timeline),
            "described": len(described.chunks),
            "answers": sorted({sid for c in described.chunks
                               for sid in c.get("samplers", {})})}


if __name__ == "__main__":
    if not SOURCE.exists():
        raise SystemExit(f"no such file: {SOURCE}")

    print(f"\n=== level 1: the components, one call each ===")
    t = time.perf_counter()
    a = by_components("example-components")
    by_hand = time.perf_counter() - t

    print(f"\n=== level 2: video_rag(), the same eight steps ===")
    t = time.perf_counter()
    b = whole_pipeline("example-pipeline")
    whole = time.perf_counter() - t

    print(f"\n=== what each produced ===")
    for name, vid, secs in (("components", a, by_hand), ("video_rag()", b, whole)):
        print(f"  {name:<12} {show(vid)}   {secs:.1f}s")
    print("  Same grid, same chunks, same answers -- they are the same eight")
    print("  steps. The time differs only because the hand-written one chose")
    print('  `transcriber="stub"`, which `video_rag()` has no argument for.')
    print("  That is the whole trade between the two levels.")

    moments, notes = retrieve.search("what is on screen", video_id=b,
                                     embedder=EMBEDDER, index_name=INDEX,
                                     moments=2)
    print(f"\n=== searching what the pipeline built ===")
    for m in moments:
        print(f"  chunk {m.chunk_id:<3} {m.start_ts:6.1f}-{m.end_ts:6.1f}s  "
              f"score {m.score:.4f}")
    if notes:
        print(f"  notes: {notes}")
