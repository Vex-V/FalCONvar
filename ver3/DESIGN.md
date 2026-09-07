# ver3 — components, artifacts, schema

What each component reads, what it produces, and the shape of both halves of
every artifact: the local JSON a run writes under `data/out/<video-id>/`, and
the Postgres table the same facts land in.

Written before the code on purpose. In `falconvar` the data flow was discovered
by building it, and two things that should have been one — the grid, stored in
both `video_chunks` and `audio_chunks` — ended up as two.

---

## The three rules everything below follows

**1. The grid is a component, not a side effect.**

In `falconvar` the chunk grid is produced *by* whichever pass happened to run
first, and read back out of its results afterwards. That is why there are four
ordering cases in `orchestrate.process`, a `Chunker` protocol and a `Timeline`
class describing the same idea, and a `FixedChunker` to bridge them.

Here the grid is its own step with its own artifact. Everything that needs
boundaries reads `timeline.json`; nothing derives them as a byproduct. There is
no "who runs first" logic in the driver, because the answer falls out of what
the chosen policy depends on:

| policy | the grid step needs | so what must run before it |
|---|---|---|
| `uniform` | a duration | **nothing** — it is arithmetic over `media.json` |
| `scene` | decoded frames | a scene-detection pass over the picture |
| `vad` | a finished transcript | the audio pass |
| `speaker` | a finished transcript + diarization | the audio pass |

`uniform` is the case worth noticing: it needs neither modality to have run.
Both `falconvar` and the "video first unless audio-only" rule make the video
pass produce it, which is work done for nothing.

**2. Cascade follows cost.**

Everything downstream is keyed by `(video_id, chunk_id)`. Whether a table
cascades from the grid is decided by one question — *if this were deleted,
what would it cost to rebuild?*

- **Rebuildable from the source file in seconds** → foreign key, `on delete
  cascade`. Re-deriving a grid should take its dependent rows with it.
- **Cost inference or a paid API call** → **no foreign key.** It carries
  `timeline_fingerprint` instead, and staleness is a comparison a reader makes.

This is not a preference. `falconvar` learned it the hard way: ingest replaces
a manifest wholesale, and re-ingesting costs 20 seconds where describing costs
money. A cascade from the grid into `descriptions` means retuning a scene
threshold silently destroys everything a VLM was paid to produce.

**3. A transcript is stored before it is cut.**

`listen` writes `transcript.raw.json` — words, segments, speaker turns, no
chunks. `cut` reads that plus a timeline and writes `transcript.json`.

Two files rather than one because re-cutting is free and transcription is not.
Whisper timestamps every word, so a raw transcript can be re-cut to any grid,
any number of times, without touching a model. Making that a file rather than
an in-process function call is what lets a grid change without re-running
inference — and the asymmetry it encodes (audio conforms cheaply, video cannot)
is the reason the ordering rules work at all.

---

## Layout

One file per component. **A package means plurality, not importance** — a
registry of interchangeable implementations. `falconvar` is 121 files partly
because everything got a package whether or not it had alternatives; four
near-identical `output/` packages (`base`, `document`, `multi`, `supabase`,
once each for ingest, describe, audio and aggregate) are ~19 files doing one
job four times.

```
FalCONvar/
  api/
  data/                      everything a run writes
  db/
    json/                    generated document schemas
    supabase/                DDL
  recovery/                  STANDALONE. imports nothing from ver3
  ver3/
    workflow.py              resolves the dependency chain, calls each driver
    paths.py                 where things live.        imports nothing
    documents.py             what every artifact IS.   imports nothing
    sinks.py                 write a document to file and/or Postgres
    db.py                    the Postgres client

    media/              1    split
    listen/             2    source · models · align · transcribe · diarize
    boundaries/         3+4
      scenes.py              video evidence: content scores -> cuts
      speech.py              audio evidence: vad / speaker cuts
      grid.py                evidence + duration -> spans, guards applied
    ingest/             5    reader · decimate · store · samplers/
    cut/                6
    describe/           7    prompts · describers
    rag/
      embed/            8
      retrieve/        10
    aggregate/          9
```

Every component is a directory holding `driver.py` (`run()` + `main()`), a
`__main__.py` so `python -m ver3.<component>` works, and whatever modules the
work itself needs. `__main__.py` rather than running `driver.py` directly:
importing the package already loads `driver`, so `python -m ver3.x.driver`
executes it twice and Python warns about exactly that.

The root holds `workflow.py` and nothing else that runs.

**`boundaries/` owns every cut derivation, including the audio ones.** In
`falconvar` the vad and speaker policies live in `audio/segment/`, which makes
the audio module know that chunking exists. Moved here, `listen/` is purely
"produce a transcript" and has no idea a grid will ever be laid over it.

The two evidence sources have nothing in common but their output: `scenes.py`
decodes video and is the expensive one, `speech.py` reads a finished transcript
and is arithmetic. `grid.py` takes cuts from either.

Note this reads `transcript.raw.json` as a **file**, never by importing
`listen`. That is what keeps the import graph acyclic while the run order flips
between policies -- on a `vad` run `listen` precedes `boundaries`, on a `scene`
run it does not, and neither imports the other in either case.

`boundaries` rather than `chunker`: `falconvar` used "chunker" for the
streaming protocol this design deletes, and reusing the word would import the
old meaning along with it.

---

## Contracts

Every component has the same signature. This is what makes them individually
testable -- feed the input file, verify the output file -- and what lets the
API be a thin layer over the same calls the CLI makes.

```python
run(video_id, source="file", sink="file", **params) -> Produced
```

**Addressed by `video_id` and a backend, never by assembled paths.** `paths.py`
is the only module that knows an artifact's filename, so a caller never
concatenates one. An explicit path is accepted as an override for a one-off
file, but it is the exception rather than the calling convention.

**`run()` is the callable; `main()` is a shim over it.** Never the reverse.
`falconvar` had to undo exactly this: `driver.py` held the run inline, so a
server would have had to import an argparse module to reach the work behind it.
`workflow.py` and the API call `run()`; only a human calls `main()`. A
single-file component gets both and stays a single file -- `python -m
ver3.media` works on a module with an `if __name__ == "__main__"` block.

**Every component returns `Produced`**, listing what it actually wrote:

```python
Produced(video_id="chernobyl", backend="file",
         artifacts={"manifest": ..., "store": ...},
         stats={"frames_sampled": 61})
```

A bare path would be shorter but says less. `Produced` reports the *actual*
output: no `store` on a `--no-frame-store` run, no `transcript` on a silent
file, and the stats a progress callback wants.

### documents.py is a leaf, and that is the point

Every artifact's shape lives in one module that imports nothing. Putting a
document's dataclass inside the component that produces it would mean `cut.py`
importing `boundaries` to read a timeline -- growing exactly the edges the
file-handoff exists to remove.

|  |  |
|---|---|
| `documents.py` | what a thing **is** — `Timeline`, `Manifest`, `RawTranscript`, `Produced`; `as_dict`/`from_dict`, `fingerprint()`, `index_at()` |
| `boundaries/grid.py` | what a thing **does** — `uniform()`, `from_cuts()`, `enforce()` |

`Timeline.index_at()` is shape. `enforce(min_s, max_s)` is derivation, and only
that side knows policies exist.

### Schemas are generated, not maintained

The dataclasses in `documents.py` are the single source of truth.
`db/json/*.schema.json` is generated from them and checked in; CI regenerates
and fails on a non-empty diff, so the three descriptions of a document --
dataclass, JSON Schema, SQL DDL -- cannot drift apart silently.

Components validate on read, because a file-handoff contract that nothing
checks is an assumption rather than a contract. **Header and structure always;
the per-frame arrays only under a flag** -- a three-hour manifest holds tens of
thousands of frame records, and validating each on every read would make the
contract layer a tax on the hot path.

---

## The chain

```
                     media file
                          │
                     ┌────▼────┐
                     │ 1 split │                    → media.json          │ videos
                     └────┬────┘
                          │
        ┌─────────────────┼──────────────────┐
        │                 │                  │
   policy=vad        policy=scene       policy=uniform
   policy=speaker         │                  │
        │                 │                  │
   ┌────▼─────┐     ┌─────▼──────┐           │
   │ 2 listen │     │ 3 scenes   │           │      → transcript.raw.json │ transcripts
   └────┬─────┘     └─────┬──────┘           │      → cuts.json          │ (none)
        │                 │                  │
        └────────────────►┼◄─────────────────┘
                     ┌────▼────┐
                     │ 4 grid  │                    → timeline.json       │ timelines, chunks
                     └────┬────┘
                          │
              ┌───────────┴───────────┐
              │                       │
        ┌─────▼──────┐          ┌─────▼─────┐
        │ 5 ingest   │          │ 6 cut     │        → manifest.json      │ manifests, chunk_samplers
        │  + store/  │          │           │        → transcript.json    │ transcript_chunks
        └─────┬──────┘          └─────┬─────┘
              │                       │
        ┌─────▼──────┐                │
        │ 7 describe │                │              → descriptions.json  │ descriptions
        └─────┬──────┘                │
              └───────────┬───────────┘
                          │
              ┌───────────┴────────────┐
              │                        │
        ┌─────▼─────┐          ┌───────▼─────┐
        │ 8 embed   │          │ 9 aggregate │       → (vectors)          │ embeddings
        └─────┬─────┘          └───────┬─────┘       → aggregates/*.json  │ aggregates,
              │                        │                                  │ video_embeddings
        ┌─────▼──────┐                 │
        │ 10 retrieve│◄────────────────┘
        └────────────┘                                → (no artifact)
```

`listen` runs before the grid only when the policy needs it. On a `uniform` or
`scene` run it can go anywhere after `split` — including concurrently with the
video pass, since it needs no boundaries.

---

## Component by component

### 1 · split

| | |
|---|---|
| **reads** | the media file |
| **produces** | `media.json` · table `videos` |
| **cost** | one container open, no decode |

Already built: `ver3/main.py`. Opens the file once, records what both halves
will need, closes it. The one place the file is inspected as a whole.

```json
{
  "video_id": "chernobyl",
  "path": "media/Chernobyl.mp4",
  "container_format": "mov,mp4,m4a,3gp,3g2,mj2",
  "duration_s": 205.28,
  "video": {
    "index": 0, "codec": "h264", "rate": 25.0, "time_base": "1/12800",
    "width": 1280, "height": 720, "frames": 5132, "duration_s": 205.28
  },
  "audio": {
    "index": 1, "codec": "aac", "rate": 44100, "channels": 2,
    "duration_s": 205.264
  }
}
```

`duration_s` on the whole is the container's, and it is the one both halves
must agree to use: the streams differ by 16 ms on this file, and a grid built
from the shorter one leaves the tail of the longer outside every chunk.

---

### 2 · listen

| | |
|---|---|
| **reads** | `media.json` |
| **produces** | `transcript.raw.json` · table `transcripts` |
| **cost** | Whisper + pyannote over the whole file. ~37x realtime |

Decode whole → transcribe → diarize → attribute. **No chunking.** The whole
file at once is not an optimisation: Whisper carries context across an
utterance, and speaker labels come from clustering over the entire recording,
so a windowed run produces speakers that are not merely misaligned but
unnameable.

```json
{
  "video_id": "chernobyl",
  "audio": { "codec": "aac", "rate": 44100, "channels": 2, "duration_s": 205.264 },
  "track":  { "duration_s": 205.264, "rms": 0.1796, "peak": 0.83, "silent": false },
  "model":  { "transcriber": "whisper", "model": "small", "language": "en",
              "diarizer": "pyannote", "version": "3.1" },
  "segments": [ { "start": 5.9, "end": 11.2, "speaker": "SPEAKER_00",
                  "text": "At 1.23 a.m., reactor 4 exploded." } ],
  "words":    [ { "start": 5.9, "end": 6.1, "text": "At", "speaker": "SPEAKER_00" } ],
  "turns":    [ { "speaker": "SPEAKER_00", "start": 5.9, "end": 42.0 } ],
  "stats":    { "segments": 34, "words": 428, "speakers": 1, "turns": 18,
                "speech_s": 182.9 }
}
```

A silent track is a result, not an error: `silent: true`, empty `segments`, and
no model is loaded at all. Every segment keeps `speaker: null` when there is no
diarizer — truthful, where labelling everything `SPEAKER_00` is not.

---

### 3 · boundaries/scenes *(only when `policy = scene`)*

| | |
|---|---|
| **reads** | `media.json` |
| **produces** | `cuts.json` |
| **cost** | one decode pass, detection at ~0.6 ms/frame on a downscaled copy |

**This is the one real departure from `falconvar`, and it costs a second decode
of the picture.** Today scene detection runs *inside* the ingest pass, which is
why the streaming `Chunker` protocol exists at all — `observe()` at native
rate, `bounds_of` returning an open end, and a correction branch for grids that
arrived from elsewhere.

Pulling it out buys three things:

- The grid is always an input to ingest, never an output. `Chunker`,
  `observe`, the open-ended `bounds_of` and `FixedChunker` all disappear;
  ingest asks `timeline.index_at(ts)` and nothing else.
- Retuning `--scene-threshold` costs one cheap pass instead of a full ingest
  with CLIP and YOLO running.
- Every policy produces a timeline the same way, so there is one code path
  instead of four.

**Measured cost.** Decoding is nearly free; converting pixels is the whole
expense. Per frame at 1280x720 on this machine: bare decode **0.40 ms**,
full-res bgr24 conversion **6.2 ms**, reformat to 320w + ContentDetector
**2.2 ms** amortised at stride 5.

That ratio, not the number of passes, is what decides the cost. Extrapolated to
three hours of 720p25 (270k frames), with the detector strided 5 and sampling
decimated to 1/s:

| | 3 h |
|---|---|
| ingest alone, converting only decimated frames (4%) | **2.99 min** |
| scene pass alone, reformatting straight to 320w | **11.88 min** |
| **separate — both passes** | **14.86 min** |
| fused in one pass, also converting lazily | 11.97 min |
| bare decode — the only work separate does twice | **1.82 min** |

**So separating costs about 3 minutes on a three-hour video**, against a
describe stage measured in hours. It is not a performance decision either way.

What *is* a performance decision is lazy conversion. `falconvar` converts every
frame to full-res BGR inside `read_frames` because `observe()` needs pixels and
the reader is the only thing holding them -- 37 min for the same file against
11.97. A 12x penalty that exists purely because detection was fused into the
pass. ver3 converts a frame only when something has asked for it.

**Stride the detector, and recalibrate when you do.** A cut is a discontinuity
that persists, so frames N apart still show it -- every real cut survived every
stride tested (1, 2, 5, 10, 25). What rises is false positives: threshold 27 is
calibrated for *adjacent* frames, and ordinary camera motion across five of
them exceeds it. `min_s` absorbs some by merging cuts arriving too close
together; the threshold still needs its own calibration, and `calibrate.py`'s
"report, do not choose" stance is the right model.

**What makes the pass worth having is caching the scores, not the cuts.**
`ContentDetector` exposes `_frame_score` and publishes `content_val`, so
`cuts.json` records the per-frame series and re-thresholding becomes arithmetic
over a cached array. Same split `calibrate.py` already uses for samplers -- run
the model once per frame, compare thousands of times. Fused, every threshold
change costs a full re-ingest with CLIP and YOLO loaded; separate, it costs
seconds.

The "one decode pass" rule in `falconvar` was an argument about *samplers*, not
about grid derivation, and it still holds: all samplers still share one pass.

A live source cannot do two passes — but it cannot do `vad` or `speaker`
either, both of which need a finished transcript. Streaming falls back to
`uniform`, which needs no pass at all.

```json
{
  "video_id": "chernobyl",
  "detector": "content",
  "params": { "threshold": 27.0, "detect_width": 320, "fps": 25.0, "stride": 5 },
  "cuts": [12.44, 38.9, 71.2, 94.4],
  "scores": { "stride": 5, "metric": "content_val",
              "values": [0.0, 3.1, 2.8, 41.7, 2.2] },
  "stats": { "frames_seen": 5132, "frames_scored": 1027, "elapsed_s": 19.7 }
}
```

Raw cuts, no guards applied: the minimum and maximum chunk lengths belong to
the grid step, so `scene`, `vad` and `speaker` are all guarded by the same code
with the same meaning.

`scores` is the point of the file. `cuts` is a thresholding of it, so keeping
the series means a different threshold is arithmetic rather than another pass
over the video.

**Measured, once built.** Chernobyl at stride 5: 5132 frames read, 1027 scored,
5.63 s, 1.10 ms/frame -- about 5 min extrapolated to three hours. Retuning the
threshold over the cached series produced cut lists **identical** to a full
re-run at 45 and at 20, in 0.12 ms against 5.9 s: a 40,000x speedup for the
same answer, because `detect` and `rethreshold` both call `cuts_from_scores`.
`cuts.json` is 29 KB for 1027 scores, so a three-hour file at stride 5 is
about 5.4 MB.

**A stride double-fires, and `min_s` is what absorbs it.** At threshold 45 the
pass found cuts at 27.0 and 27.2 -- one scene change caught on two consecutive
scored frames 0.2 s apart -- and another at 204.8, 0.48 s before the end. Both
open a span below `min_s`, and `grid.enforce` merged both away, leaving nine
chunks none shorter than 5 s. So the guard that exists for voice activity turns
out to cover the stride's characteristic failure too. Note the direction:
`enforce` merges a short chunk *backwards*, so the boundary it removes is that
chunk's **start**.

---

### 4 · boundaries/grid

| | |
|---|---|
| **reads** | `media.json`, plus `cuts.json` *or* `transcript.raw.json` depending on policy |
| **produces** | `timeline.json` · tables `timelines`, `chunks` |
| **cost** | arithmetic |

The only component that decides boundaries. Takes interior cut times from
whichever source the policy names, applies `min_s` and `max_s`, merges a short
tail, and writes the spans.

Both guards are mandatory for any content-derived policy. Voice activity cuts
on every pause — every second or two on conversational audio, which shreds the
video into chunks too short to describe. A monologue yields zero cuts and one
chunk covering the file. `max_s` splits **evenly**, not into fixed bites: 30 s
bites off a 62.5 s span leave 2.5 s, so a guard whose job is preventing tiny
chunks would create one.

```json
{
  "timeline_version": 1,
  "video_id": "chernobyl",
  "policy": "vad",
  "derived_from": "audio",
  "params": { "silence_s": 0.65, "min_s": 5.0, "max_s": 30.0 },
  "fingerprint": "a3f9c21e8b40d7e5",
  "duration_s": 205.28,
  "chunks": [
    { "chunk_id": 0, "start_ts": 0.0,   "end_ts": 16.2 },
    { "chunk_id": 1, "start_ts": 16.2,  "end_ts": 41.7 }
  ]
}
```

The last span always reaches `media.json`'s container duration, so the grid
covers the longer stream. `fingerprint` is a hash of the rounded spans plus
policy and params, and is what every expensive downstream table carries to
detect that it was built against a different grid.

---

### 5 · ingest

| | |
|---|---|
| **reads** | `media.json`, `timeline.json` |
| **produces** | `manifest.json`, `store/` · tables `manifests`, `chunk_samplers` |
| **cost** | one decode pass; samplers run models on decimated frames |

One decode, every sampler fed from it. Per frame: decimate on media time, ask
the timeline which chunk this is, reset samplers at a boundary, offer the frame
to each sampler, release the pixels.

Because the grid arrives as data, ingest **never** corrects a boundary. No
final-`end_ts` fixup and no tail merge — both were `falconvar` working around
having derived the grid itself, and both silently edited spans another pass had
chosen.

```json
{
  "manifest_version": 3,
  "video_id": "chernobyl",
  "timeline_fingerprint": "a3f9c21e8b40d7e5",
  "manifest_fingerprint": "77c1de0a94b2",
  "config": {
    "decimator": { "per_second": 1.0 },
    "samplers": [
      { "id": "yolo:overview", "name": "yolo", "prompt": "overview",
        "min_interval_s": 0.0, "max_per_chunk": null, "threshold": 0.83 }
    ],
    "frame_store": { "root": "store", "format": "jpg", "quality": 95,
                     "scope": "sampled" }
  },
  "stats": { "frames_read": 5132, "frames_decimated": 205, "frames_sampled": 61 },
  "chunks": [
    { "chunk_id": 0, "decimated_frames": 16,
      "samplers": { "yolo:overview": { "frame_count": 2, "frames": [
        { "index": 0,   "media_ts": 0.0,  "chunk_local_index": 0, "pts": 0 },
        { "index": 150, "media_ts": 6.0,  "chunk_local_index": 6, "pts": 76800,
          "score": 0.7412 }
      ] } } }
  ]
}
```

`start_ts`/`end_ts` are **not** repeated here — they are the grid's, and one
copy is the whole point. A reader joins on `chunk_id`.

---

### 6 · cut

| | |
|---|---|
| **reads** | `transcript.raw.json`, `timeline.json` |
| **produces** | `transcript.json` · tables `transcripts`, `transcript_chunks` |
| **cost** | arithmetic |

A word belongs to the chunk containing its **midpoint** — the same rule used to
attribute a word to a speaker, because a word straddling a boundary belongs to
whichever side holds more of it, and every word must land in exactly one chunk
or the text is duplicated or dropped.

Chunks with no speech are **kept, with empty text**. The grid is shared, so
`chunk_id` must mean the same thing here as in the manifest; dropping the quiet
ones renumbers everything after them.

```json
{
  "video_id": "chernobyl",
  "timeline_fingerprint": "a3f9c21e8b40d7e5",
  "chunks": [
    { "chunk_id": 0, "text": "", "word_count": 0,
      "structured": { "speakers": [] }, "turns": [] },
    { "chunk_id": 1, "text": "At 1.23 a.m., reactor 4 exploded.",
      "word_count": 7, "structured": { "speakers": ["SPEAKER_00"] },
      "turns": [ { "speaker": "SPEAKER_00", "start": 16.4, "end": 21.0,
                   "text": "At 1.23 a.m., reactor 4 exploded." } ] }
  ]
}
```

Only `speakers` goes into `structured`, never `turns`: `turns[].text` *is* the
transcript, so rendering it for embedding appends the whole chunk a second time
interleaved with timestamps read as numbers.

---

### 7 · describe

| | |
|---|---|
| **reads** | `manifest.json`, `store/` |
| **produces** | `descriptions.json` · table `descriptions` |
| **cost** | one VLM call per `(chunk, sampler)`. The expensive stage |

Reads the frame store and nothing else — no seek-the-video fallback, because
that would quietly do the store's job while leaving it broken, silently, at
~40x the cost.

The question asked is the sampler's `prompt`, falling back to its name. Which
keys a call's schema may fill is narrowed by the other **questions** on the
chunk, never the sampler ids.

```json
{
  "description_version": 2,
  "video_id": "chernobyl",
  "manifest_fingerprint": "77c1de0a94b2",
  "timeline_fingerprint": "a3f9c21e8b40d7e5",
  "model": { "describer": "openai", "model": "gpt-5.4-mini", "prompts": "e41b0c2d" },
  "chunks": [
    { "chunk_id": 0,
      "samplers": {
        "yolo:overview": { "frame_count": 2, "frame_indexes": [0, 150],
          "description": "A 3D cutaway of an RBMK-1000 reactor…",
          "structured": {}, "elapsed_s": 6.4 } },
      "structured": { "setting": "…", "people": [] } }
  ]
}
```

---

### 8 · embed

| | |
|---|---|
| **reads** | `descriptions.json`, `transcript.json` |
| **produces** | vectors · table `embeddings` |
| **cost** | one embedding call per changed unit |

Both modalities land in one table. A transcript chunk is a unit with
`sampler = "transcript"`, so `--sampler transcript` narrows a search to what
was said exactly as `--sampler yolo` narrows it to who was seen.

Keyed by `text_hash`, so a re-run embeds only what changed. Silent chunks are
skipped — a vector of the empty string answers every query equally badly.

One vector space per **embedder**, never per sampler. The sampler is payload
and querying one is a filter.

---

### 9 · aggregate

| | |
|---|---|
| **reads** | `descriptions.json`, `transcript.json`, `timeline.json` |
| **produces** | `aggregates/<name>.json` · tables `aggregates`, `video_embeddings` |
| **cost** | by tier: `free` arithmetic, `local` GPU, `llm` paid |

Reads finished documents, never the video and never another component's
modules. Answers what similarity cannot: counts, coverage, who dominated,
chapters.

`inputs_fingerprint` is a hash of the chunk text actually read — a summary of
descriptions since rewritten reads perfectly, which is why it cannot be left to
a reader to notice.

Only the summary is embedded, into `video_embeddings`, because `chunk_id`
answers *which twenty seconds* and a summary answers *which video*.

---

### 10 · retrieve

| | |
|---|---|
| **reads** | `embeddings` (+ `chunks` for spans, `descriptions` for text) |
| **produces** | nothing on disk |

RRF twice: vector rank and text rank fused per description, then descriptions
fused into chunks as `1/(k+best) + 0.5/(k+second)` at k=10. Never a weighted
score — cosine distance and `ts_rank_cd` have no common scale.

---

## Files a complete run leaves behind

```
data/out/<video-id>/
  media.json             1  what the file is
  transcript.raw.json    2  words and speakers, ungridded
  cuts.json              3a scene cuts, raw          (scene policy only)
  timeline.json          4  THE GRID
  manifest.json          5  which frames, and why
  store/0000000.jpg      5  the frames themselves
  transcript.json        6  the transcript on the grid
  descriptions.json      7  one answer per (chunk, sampler)
  aggregates/*.json      9  video-level structure
```

---

## Schema

```sql
-- ===========================================================================
-- 1 · the file
-- ===========================================================================
create table if not exists videos (
  video_id      text primary key,
  uri           text not null,
  container     text not null,
  duration_s    numeric,
  has_video     boolean not null,
  has_audio     boolean not null,
  video_stream  jsonb,                    -- codec, rate, time_base, w, h, frames
  audio_stream  jsonb,                    -- codec, rate, channels
  seen_at       timestamptz not null default now()
);

-- ===========================================================================
-- 4 · the grid. One per video. Everything below joins on (video_id, chunk_id).
-- ===========================================================================
create table if not exists timelines (
  video_id      text primary key references videos on delete cascade,
  policy        text not null,            -- uniform | scene | vad | speaker
  derived_from  text not null,            -- video | audio | grid
  params        jsonb not null default '{}'::jsonb,
  fingerprint   text not null,
  duration_s    numeric not null,
  chunk_count   int not null,
  built_at      timestamptz not null default now()
);

create table if not exists chunks (
  video_id  text not null references timelines on delete cascade,
  chunk_id  int  not null,
  start_ts  numeric not null,
  end_ts    numeric not null,
  primary key (video_id, chunk_id),
  check (end_ts > start_ts)
);

create index if not exists chunks_start on chunks (video_id, start_ts);

-- ===========================================================================
-- 5 · ingest. Cheap to rebuild (~20 s), so it cascades from the grid.
-- ===========================================================================
create table if not exists manifests (
  video_id     text primary key references videos on delete cascade,
  timeline_fingerprint text not null,
  manifest_fingerprint text not null,
  source       jsonb not null,
  config       jsonb not null,            -- decimator, samplers[], frame_store
  stats        jsonb not null default '{}'::jsonb,
  ingested_at  timestamptz not null default now()
);

-- One row per (chunk, sampler) rather than a jsonb blob on the chunk: this is
-- what makes "which chunks did yolo pick frames in" a query rather than a scan,
-- and it is the level `--sampler` actually filters at.
create table if not exists chunk_samplers (
  video_id    text not null,
  chunk_id    int  not null,
  sampler_id  text not null,              -- "yolo" | "yolo:overview" | "uniform:text"
  sampler     text not null,              -- the strategy half
  question    text not null,              -- the prompt half, resolved
  frame_count int  not null,
  frames      jsonb not null,             -- [{index, media_ts, chunk_local_index, pts, score?}]
  primary key (video_id, chunk_id, sampler_id),
  foreign key (video_id, chunk_id) references chunks on delete cascade
);

-- ===========================================================================
-- 6 · transcript. Re-cutting is free, so it cascades too.
-- ===========================================================================
create table if not exists transcripts (
  video_id     text primary key references videos on delete cascade,
  timeline_fingerprint text,
  model        jsonb not null default '{}'::jsonb,
  track        jsonb not null default '{}'::jsonb,   -- rms, peak, silent, duration
  stats        jsonb not null default '{}'::jsonb,
  -- The ungridded record. Kept so a grid change never re-runs Whisper.
  segments     jsonb not null default '[]'::jsonb,
  words        jsonb not null default '[]'::jsonb,
  turns        jsonb not null default '[]'::jsonb,
  heard_at     timestamptz not null default now()
);

create table if not exists transcript_chunks (
  video_id    text not null,
  chunk_id    int  not null,
  text        text not null default '',
  word_count  int  not null default 0,
  structured  jsonb not null default '{}'::jsonb,    -- {speakers: [...]}
  turns       jsonb not null default '[]'::jsonb,
  primary key (video_id, chunk_id),
  foreign key (video_id, chunk_id) references chunks on delete cascade
);

-- ===========================================================================
-- 7 · descriptions. NO FOREIGN KEY, deliberately.
--
-- These cost inference. A cascade from `chunks` would mean retuning a scene
-- threshold destroys every VLM answer bought so far. Staleness is a
-- fingerprint comparison a reader makes, never a deletion the database makes.
-- ===========================================================================
create table if not exists descriptions (
  video_id      text not null,
  chunk_id      int  not null,
  sampler_id    text not null,
  question      text not null,
  frame_indexes int[] not null,
  frame_count   int  not null,
  description   text,
  structured    jsonb not null default '{}'::jsonb,
  model         jsonb not null default '{}'::jsonb,  -- describer, model, prompts hash
  elapsed_s     numeric,
  timeline_fingerprint text,
  manifest_fingerprint text,
  described_at  timestamptz not null default now(),
  primary key (video_id, chunk_id, sampler_id)
);

create index if not exists descriptions_chunk on descriptions (video_id, chunk_id);

-- ===========================================================================
-- 8 · embeddings. Also no foreign key, for the same reason: paid calls.
-- ===========================================================================
create table if not exists embeddings (
  video_id    text not null,
  chunk_id    int  not null,
  sampler_id  text not null,              -- + "transcript" for the audio side
  embedder    text not null,              -- name:model:dims
  text_hash   text not null,              -- embed only what changed
  content     text not null,
  structured  jsonb not null default '{}'::jsonb,
  vector      vector not null,
  fts         tsvector generated always as (
                to_tsvector('english',
                  content || ' ' || coalesce(jsonb_path_query_array(
                    structured, 'strict $.**.type()')::text, ''))
              ) stored,
  timeline_fingerprint text,
  embedded_at timestamptz not null default now(),
  primary key (video_id, chunk_id, sampler_id, embedder)
);

create index if not exists embeddings_fts on embeddings using gin (fts);
create index if not exists embeddings_structured on embeddings using gin (structured);

-- ===========================================================================
-- 9 · aggregates. Video-level, so not keyed by chunk at all.
-- ===========================================================================
create table if not exists aggregates (
  video_id     text not null references videos on delete cascade,
  aggregate_id text not null,             -- stats | speakers | summary | chapters | …
  tier         text not null,             -- free | local | llm
  payload      jsonb not null,
  inputs_fingerprint text not null,       -- hash of the chunk text actually read
  built_at     timestamptz not null default now(),
  primary key (video_id, aggregate_id)
);

create table if not exists video_embeddings (
  video_id   text not null references videos on delete cascade,
  kind       text not null,               -- "summary"
  embedder   text not null,
  text_hash  text not null,
  content    text not null,
  vector     vector not null,
  primary key (video_id, kind, embedder)
);
```

### What that changes from `falconvar`

| | `falconvar` | here |
|---|---|---|
| grid storage | `start_ts`/`end_ts` duplicated in `video_chunks` **and** `audio_chunks` | one `chunks` table, joined by both |
| grid provenance | in the manifest's `config.chunker` | its own `timelines` row |
| sampler picks | a `jsonb` blob on `video_chunks.samplers` | `chunk_samplers`, one row per pair, queryable |
| the split | not recorded | `videos` + `media.json` |
| raw transcript | in memory only, lost after `cut` | `transcript.raw.json` + `transcripts` |
| grid derivation | a byproduct of whichever pass ran first | component 4, always |
| ordering logic | four cases in `process()` | falls out of the policy's dependencies |
| chunk correction | ingest fixes `end_ts`, merges the tail, except on `fixed` | never; the grid is an input |

### Still to settle

- **RLS and access policies.** `falconvar`'s `db/schema.sql` carries them; none
  are written here yet.
- **The `fts` expression.** Copied in shape from `falconvar`, which learned
  that `add column if not exists` cannot repair a stale generated column — the
  statement has to be `drop column if exists` then an unconditional add.
- **Filterable structured values.** Still free text, so a filter for `cashier`
  still matches everything. Needs an `enum` on the fields meant to be filtered
  before `structured` is worth a GIN index.
- **Whether `videos.uri` should be stable.** `manifest_fingerprint` deliberately
  excludes it so a moved file is not a new video; `videos` currently stores the
  path it was last seen at.
