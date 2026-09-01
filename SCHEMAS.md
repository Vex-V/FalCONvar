# Schemas — every structure this project writes

One reference for what the pipeline produces, in all four places it lands:
local JSON, the frame store, Postgres, and the vector index. Runnable DDL is in
[`schema.sql`](schema.sql); this file explains what the fields mean and how the
copies correspond.

---

## Where everything lives

```
out/<video-id>/
  manifest.json        which frames were kept, and why      <- ingest
  store/               those frames, keyed by frame index   <- ingest
  descriptions.json    what a model said about them         <- describe
out/qdrant/            local vector index                   <- retrieve

Supabase   videos + chunks              = manifest.json, as rows
           descriptions                 = descriptions.json, as rows
           description_embeddings       = the vectors + what they were built from
```

Everything for one video sits under one directory, so a video's whole output is
one thing to inspect, copy or delete.

## What derives from what

```
video ──ingest──> manifest.json ──describe──> descriptions.json ──retrieve──> vectors
             └──> store/           (reads the store)              (embeds the text)
```

Only the manifest and the descriptions are records. The frame store is a cache
(regenerable byte-for-byte by `recovery.recreate`), and the vectors are derived
(regenerable by re-indexing). Deleting either loses nothing but time.

---

## 1. `manifest.json`

Written by `ver2.ingest`, rewritten atomically as each chunk closes, so a
reader always sees a whole document. `complete` says whether ingestion
finished — it distinguishes "no more chunks yet" from "no more chunks ever".

```jsonc
{
  "manifest_version": 1,
  "video_id": "test1",
  "complete": true,

  "source": {                    // what was probed, before anything was decided
    "uri": "media/test1.mp4",
    "fps": 13.093, "fps_trusted": true,
    "time_base": "1/15360",      // pts units; seconds are a lossy rendering
    "width": 1270, "height": 720,
    "frame_count": 1283,
    "timeline": "pts",           // "pts" or "index" — how frames are addressed
    "rotation": 0.0,
    "notes": []
  },

  "config": {                    // every setting, so a run is reproducible
    "decimator": { "per_second": 1.0 },
    "chunker":   { "name": "uniform", "duration_s": 20.0 },
    "samplers": [                // one entry per sampler, its own shape
      { "id": "clip", "name": "clip", "min_interval_s": 3.0,
        "max_per_chunk": null, "threshold": 0.96, "mode": "reference",
        "embedder": { "name": "clip", "model": "openai/clip-vit-base-patch32",
                      "dim": 512, "device": "cuda" } }
    ],
    "frame_store": {             // null if the run kept no pixels
      "dir": "out/test1/store", "format": "jpg",
      "max_width": 1920, "quality": 85,
      "key": "frame_index",
      "scope": "sampled"         // "sampled" | "decimated"
    }
  },

  "stats": {                     // run counters. Diagnostics, not content.
    "frames_read": 1283, "frames_decimated": 98, "frames_sampled": 105,
    "chunks": 5, "elapsed_s": 19.25,
    "stored_frames": 80, "stored_mb": 22.34
  },

  "chunks": [{
    "chunk_id": 0,
    "start_ts": 0.0, "end_ts": 20.0,     // media time; the last chunk's end
                                         // is corrected at finish()
    "decimated_frames": 20,              // how many frames were offered here
    "samplers": {
      "clip": {
        "frame_count": 8,
        "frames": [{
          "index": 66,                   // the frame store's key
          "media_ts": 5.041,             // the only clock a decision may use
          "chunk_local_index": 5,
          "pts": 66000,                  // the exact address for seeking back
          "score": 0.9356                // why it was kept; absent for the
        }]                               // first frame of a chunk
      },
      "yolo": { "frame_count": 7, "frames": [ /* ... */ ] }
    }
  }]
}
```

**The same frame appears under every sampler that kept it.** On test1, 105
frame records cover 80 distinct frames — the store holds one copy per index.

## 2. The frame store — `out/<video-id>/store/`

Files named `%07d.jpg` by **source frame index**, matching `frames[].index`.
Not keyed by sampler, because samplers overlap heavily and keying per sampler
would write a third of the store twice.

Written at `max_width` 1920 rather than smaller: at 1024 px a VLM misread a
burnt-in clock as 11:17:40 when it read 11:17:19.

## 3. `descriptions.json`

Written by `ver2.describe`, one entry per `(chunk, sampler)` pair — which is
the unit one describer call covers.

```jsonc
{
  "description_version": 1,
  "video_id": "test1",
  "complete": true,                      // every pair has an answer

  "manifest_fingerprint": "b343641077dbd835",
      // hash of `source` minus `uri` and `config` minus `frame_store`. Known
      // before the first chunk exists, survives the video and store moving,
      // changes when the sampling changes. Staleness is a comparison.

  "source": { "uri": "media/test1.mp4", "video_id": "test1",
              "manifest_version": 1 },

  "model": {                             // part of the resume key
    "name": "openai",
    "params": { "model": "gpt-5.4-mini", "max_output_tokens": 2000,
                "response": "json_schema/description_<sampler>",
                "prompts": "ebd10acd63a8" }
                // hash of every instruction and schema in vlm/prompts.py, so
                // editing a prompt invalidates resume the way a model change does
  },

  "stats": { "chunks": 5, "described": 10, "skipped": 0, "elapsed_s": 88.6,
             "frames_requested": 105, "cache_hits": 25, "frames_read": 80,
             "load_seconds": 0.3 },      // run counters; not stored in Postgres

  "chunks": [{
    "chunk_id": 2, "start_ts": 40.0, "end_ts": 60.0,
    "processed": true,                   // every sampler here has an answer

    "structured": { /* union of the samplers' fields — see below */ },

    "samplers": {
      "clip": {
        "frame_count": 15,
        "frame_indexes": [524, 537, 550],   // evidence, straight from the store
        "description": "A small retail shop ...",   // prose; gets embedded
        "structured": { "setting": "...", "objects": [], "visible_text": [],
                        "actions": [], "changes": [], "tags": [] },
        "elapsed_s": 9.6
      },
      "yolo": {
        "frame_count": 9,
        "frame_indexes": [524, 563, 603],
        "description": "A small toy and game shop ...",
        "structured": { "people": [{ "appearance": "...", "clothing": "...",
                                     "role": "...", "action": "..." }] },
        "elapsed_s": 6.4
      }
    }
  }]
}
```

**`chunks[].structured` is the union of its samplers' fields**, recomputed on
every write so it cannot drift. The union is unambiguous because no two
samplers on a chunk own the same key — see the next section.

## 4. What the model is asked to return

One strict `json_schema` per sampler. `summary` is always present and is the
prose that gets embedded; the rest is what a filter can use.

| sampler | fields beyond `summary` |
|---|---|
| `clip`, `uniform`, unregistered | `setting`, `people`, `objects`, `visible_text`, `actions`, `changes`, `tags` |
| `yolo` | `people[{appearance, clothing, role, action}]` |
| `objects` | `objects[{object, appearance, context}]` |
| `text` | `visible_text[{text, context}]` |

**The scene schema narrows.** A key a specialist owns is removed from the scene
schema when that specialist ran on the same chunk, so nothing is asked twice
and the union has nothing to reconcile:

```
clip alone                  -> setting, people, objects, visible_text, actions, changes, tags
clip + yolo                 -> setting,         objects, visible_text, actions, changes, tags
clip + yolo + text + objects-> setting, actions, changes, tags
```

Specialists return **objects, not parallel lists**: a list of people beside a
list of actions does not say who did what, and cannot be made to afterwards.

## 5. Supabase — four tables

DDL in [`schema.sql`](schema.sql), idempotent and safe to re-run.

### `videos` + `chunks` — the manifest, as rows
Identical content to `manifest.json`; `chunks.samplers` holds the frame records
verbatim as `jsonb`. `export_manifest(video_id)` reassembles the document
server-side, so there is no second implementation of the format to drift.
`chunks` cascades from `videos`, and ingest replaces both wholesale.

### `descriptions` — one row per `(video_id, chunk_id, sampler)`
```
video_id, chunk_id, sampler          primary key
frame_indexes int[], frame_count
description  text                    the summary
structured   jsonb                   that sampler's fields
model        jsonb                   name + params + prompts hash
elapsed_s, manifest_fingerprint, described_at
```
**No foreign key to `videos`, deliberately.** Ingest replaces a manifest
wholesale and re-ingesting costs 20 seconds where describing costs inference; a
cascade would let the cheap operation destroy the expensive one.
`manifest_fingerprint` replaces it — staleness becomes a comparison rather than
a deletion. Read back with `export_descriptions(video_id)`.

### `description_embeddings` — one row per `(description, embedder)`
```
video_id, chunk_id, sampler, embedder    primary key
dims int, embedding vector               unconstrained width, see below
content   text                           the summary, for display
structured jsonb                         for exact filtering (GIN)
fts       tsvector                       generated from content + structured strings
text_hash text                           hash of what was embedded
manifest_fingerprint, start_ts, end_ts, frame_indexes, indexed_at
```
`embedder` is `name:model:dims`, in the key, so several embedders coexist and
are never mixed in one ranking. `embedding` is declared `vector` with **no**
dimension, because nomic (768), bge-m3 (1024) and OpenAI (1536) differ — that
means exact cosine search and no ANN index, which is correct at this scale.
`search_descriptions()` does vector + full-text and fuses them with RRF.

## 6. Qdrant — `out/qdrant/`

Embedded, no server. **One collection per embedder**,
`descriptions__openai_text_embedding_3_small_1536`, because a collection has a
fixed width. Point id is `uuid5("video_id:chunk_id:sampler")` so re-indexing
overwrites rather than duplicating.

Payload mirrors the Postgres columns: `video_id`, `chunk_id`, `sampler`,
`content`, `structured`, `text_hash`, `manifest_fingerprint`, `start_ts`,
`end_ts`, `frame_indexes`. Dense only — the lexical half lives in Postgres.

## 7. What gets embedded

**Not** the summary alone. `Unit.embed_text` is the summary plus the structured
fields rendered to text, one line per field, with each entity's values joined so
`{appearance, clothing, role, action}` stays one clause. Measured on test1 over
22 disjoint query pairs (dense MRR, random 0.457):

| embedded from | literal | paraphrase |
|---|---|---|
| summary | 0.528 | 0.522 |
| structured | 0.636 | 0.586 |
| **both** | **0.705** | **0.608** |

`text_hash` covers `embed_text`, so a change to either half re-embeds.

---

## The keys that tie it together

| key | where | what it protects against |
|---|---|---|
| `video_id`, `chunk_id`, `sampler` | everywhere | the identity of a unit of work |
| `frames[].index` | manifest ↔ store | addressing a frame's pixels |
| `frames[].pts` | manifest ↔ video | seeking back exactly, on non-integer fps |
| `manifest_fingerprint` | descriptions, embeddings | descriptions of a sampling that has changed |
| `model` (incl. `prompts`) | descriptions | resuming across a model or prompt change |
| `text_hash` | embeddings | a vector built from text since rewritten |
| `embedder` / collection name | embeddings | ranking 768-wide vectors against 1536-wide |
