# Routes — the HTTP surface

Eleven endpoints in `api/main.py`. The live schema is always at `/docs`; this
file is the reasoning behind it, which the schema cannot carry.

Everything is same-origin: the browser client is served by the same app at
`/app`, so there is no CORS and no base URL to configure.

---

## The shape of it

The endpoints divide by **how long they take**, and that division is the whole
design:

| kind | endpoints | why |
|---|---|---|
| **immediate** | `/health`, `/capabilities`, `/videos` (GET), `/videos/{id}/{name}`, `/videos/{id}/frames/{n}`, `/search` | one query or one file read; tens of milliseconds |
| **queued** | `/videos` (POST), `/describe`, `/embed` | minutes of GPU or paid inference; returns `202` and a job id |
| **introspective** | `/capabilities` | reads the registries, so `ver2` gaining a sampler needs no edit here |

**Search is immediate on purpose.** It is one embedding call and one SQL query,
so making it a job would add a poll to something that answers faster than the
poll interval.

**Processing is queued on purpose, one at a time.** Every heavy stage contends
for the same GPU: CLIP and YOLO during ingest, Whisper and pyannote during the
audio pass. Two videos at once does not halve the wall clock, it doubles the
resident weights and invites an allocator failure halfway through the more
expensive one. A queue of one is the honest shape of the hardware.

---

## GET `/health`

```json
{ "ok": true, "queued": 0 }
```

`queued` is the depth of the work queue, not a count of running jobs — there is
only ever one of those.

## GET `/capabilities`

What this deployment can be asked for, read from the registries rather than a
list:

```json
{
  "samplers":     ["clip", "objects", "overview", "text", "uniform", "yolo"],
  "chunking":     ["uniform", "scene", "vad", "speaker"],
  "describers":   ["openai", "stub"],
  "transcribers": ["stub", "whisper"],
  "diarizers":    ["none", "pyannote"],
  "embedders":    ["local", "openai"],
  "indexes":      ["qdrant", "pgvector"],
  "sinks":        ["file", "supabase"],
  "defaults":     { "embedder": "openai", "index": "pgvector" }
}
```

Registering a sampler in `ver2` makes it appear here, and therefore in the web
form, with nothing else edited. That is the reason this endpoint exists rather
than the client hardcoding a list.

---

## POST `/videos` — upload and process

`multipart/form-data`. Returns **202** with a job id; the work happens on the
worker.

| field | default | notes |
|---|---|---|
| `file` | — | the media file |
| `samplers` | `clip` | comma-separated. `uniform:text` runs the *text* question on a clock |
| `chunking` | `uniform` | `uniform` \| `scene` \| `vad` \| `speaker` |
| `chunk_duration` | `20` | uniform: the length; everything else: the maximum |
| `every_seconds` | `3` | `uniform`/`overview` cadence, in media time |
| `vocabulary` | — | `objects` only. **No useful default**: a mismatched list found 2.4 detections/frame where a matched one found 5.1 |
| `threshold` | per-sampler | leave unset; the useful value differs by an order of magnitude between samplers |
| `per_second` | `1.0` | decimation rate |
| `frame_store` | `true` | keep the sampled pixels; describe reads only from here |
| `use_audio` | `true` | transcribe and diarize |
| `transcriber` / `diarizer` | `whisper` / `pyannote` | `stub` and `none` need no model |
| `language` | — | skip Whisper's language detection |
| `sinks` | `file` | `file,supabase` writes both; file is primary |
| `video_id` | from the filename | keys four tables and the output directory |

**Validation is synchronous even though the work is not.** A bad sampler name
or `chunking=speaker` with `diarizer=none` is a **422** the caller sees at once,
and the upload is deleted rather than left behind. What cannot be known without
opening the file still fails inside the job: `chunking=vad` on a silent track is
a **202**, because whether a track has speech is not a property of the request.

**The id comes from the filename, not the client.** It is also the key every
table, every output directory and every Qdrant payload uses, so it is stripped
to `[A-Za-z0-9_-]` before anything is written.

## POST `/describe`

```json
{ "video_id": "Chernobyl", "describer": "openai",
  "model": null, "sinks": ["file"], "limit": null }
```

202 + job id. `limit` stops after N describer calls, which is how to sample a
long video without paying for all of it. Resume is automatic and keyed on the
manifest fingerprint *and* the model block, so switching describers re-runs
rather than silently reporting success.

## POST `/embed`

```json
{ "video_id": "Chernobyl", "embedder": null, "model": null, "indexes": null }
```

202 + job id. Nulls fall through to `ver2/embed/defaults.py`. Embeds the
descriptions **and** the transcript if there is one — both land in
`chunk_embeddings`, distinguished only by `sampler`. Only units whose text has
changed are re-embedded; the rest are skipped on a hash comparison.

---

## GET `/jobs` and `/jobs/{job_id}`

```json
{ "id": "4e21db3a3b5d", "kind": "ingest", "video_id": "test",
  "state": "done", "stage": "done", "elapsed_s": 69.31,
  "result": { "chunks": 15, "frames_sampled": 51, ... },
  "error": null,
  "history": [ { "stage": "video", "at": 1.2, "chunks": 15 } ] }
```

`state` is `queued` \| `running` \| `done` \| `failed`. `history` is every
progress callback in order, so a poller that missed the middle of a run can
still see what happened rather than only where it ended.

On failure, `error` carries the message and `detail.traceback` the last 4 KB of
the stack — because a failure four stages into a pipeline is not diagnosable
from its last line, and there is no terminal here to have watched it happen.

**Job records live in memory and die with the process.** `GET /jobs` says so in
its own payload. What a job *produced* is on disk and in Postgres regardless, so
a restart loses the record of the run, never its output.

---

## GET `/videos`

```json
{ "videos": [ { "video_id": "Chernobyl", "chunks": 14, "policy": "vad",
                "frames": 47, "timeline_fingerprint": "ecc94042102dc605",
                "has": { "manifest": true, "timeline": true,
                         "descriptions": true, "transcript": true } } ] }
```

Read off the `out/` directory rather than remembered. That is why a restarted
server still knows everything it produced.

## GET `/videos/{video_id}/{name}`

`name` is one of `manifest`, `timeline`, `descriptions`, `transcript`. Returns
the document as written; see [SCHEMAS.md](SCHEMAS.md) for what each field means.
Anything else is a 404 rather than a path traversal.

## GET `/videos/{video_id}/frames/{index}`

One JPEG, exactly as ingest wrote it — no decode, no resize. The `index` is a
source frame index, which is what `frame_indexes` on a search result contains,
so a client can show the evidence for a moment without a second round trip to
work out what to ask for.

404 when the run kept no frame store, or when that frame was not one a sampler
selected. The message says which.

---

## POST `/search`

```json
{ "query": "the moment the reactor exploded",
  "video_id": null, "sampler": null, "moments": 5, "limit": 20 }
```

```json
{ "query": "...",
  "frames": "/videos/{video_id}/frames/{index}",
  "moments": [ {
    "video_id": "Chernobyl", "chunk_id": 7,
    "start_ts": 94.4, "end_ts": 110.9, "score": 0.1326,
    "samplers": ["overview", "transcript"],
    "frame_indexes": [2375, 2500, 2625, 2750],
    "descriptions": { "overview": "A 3D cutaway…", "transcript": "At 1.23 a.m.…" }
  } ] }
```

`limit` is how many **descriptions** to rank; `moments` how many **windows** to
return. The first should be comfortably larger than the second, or a chunk
cannot benefit from agreement between its own descriptions.

**`descriptions` is a map, and its size is the point.** Two entries means two
independent accounts of the same window both matched — what was seen and what
was said, or two samplers that never saw each other's output. That agreement is
what the per-sampler split exists to produce.

**`sampler` narrows to one question** (`yolo`, `text`, `transcript`, `overview`,
…) and gives up exactly that: with one sampler a chunk can contribute at most
one description, so a moment's score collapses to a single `1/(k+1)` and there
is nothing left to fuse.

A 502 here means the index refused the query — usually an embedder that does not
match the one the index was built with, which is caught rather than returning a
well-formed ranking that means nothing.
