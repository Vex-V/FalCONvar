# Routes

The HTTP surface is `api/`, over `falconvar`. Run it with:

    python -m uvicorn api.main:app --port 8000

[`/docs`](http://localhost:8000/docs) is the generated schema and is the
authority on request and response shapes -- with
[`/redoc`](http://localhost:8000/redoc) and
[`/openapi.json`](http://localhost:8000/openapi.json) the same thing rendered
differently. This file records the reasoning the schema cannot carry.
[`/`](http://localhost:8000/) redirects to [`/app/`](http://localhost:8000/app/)
when `web/` is present and to `/docs` when it is not -- decided from the
directory rather than assumed, since a redirect to a mount that does not exist
is a 404 that reads as breakage.

**Every link below assumes `--port 8000`.** Change the port in the command and
the links no longer point at your server.

## Three shapes of route

**Immediate** -- reading what exists, and searching. Milliseconds.

*Reading the rows is a fourth thing, added later and kept in `browse.py`:
`/videos/{id}/artifacts/{name}` hands over a whole document, which is the right
shape for a download and the wrong one for a question.*

**Queued** -- anything that decodes, transcribes or pays a model. A 202 with a
job id, and the caller polls. Slow work runs **one job at a time**: every heavy
stage contends for the same 8 GiB GPU, so two videos at once does not halve the
wall clock, it doubles the resident weights.

**Uniform** -- `POST /videos/{id}/run/{component}` runs *any* component,
because every one of them is `run(video_id, ...) -> Produced`. A route and a
handler per stage is what the uniform signature removes: adding a component
adds a row to `service.COMPONENTS` and this route already serves it.

## The endpoints

| method | path | |
|---|---|---|
| GET | [`/health`](http://localhost:8000/health) | liveness, and how many jobs are queued |
| GET | [`/capabilities`](http://localhost:8000/capabilities) | every registry, plus the defaults read off `workflow.Options` |
| POST | `/videos` | upload a file and queue the whole pipeline. **202** |
| POST | `/videos/{id}/run/{component}` | queue one component. **202** |
| GET | [`/jobs`](http://localhost:8000/jobs) | every job this process remembers |
| GET | `/jobs/{job_id}` | one job: state, stage, history, stats |
| GET | [`/videos`](http://localhost:8000/videos) | every video with an output directory |
| GET | `/videos/{id}` | what this video has, with URLs |
| GET | `/videos/{id}/artifacts/{name}` | one document. `?download=1` adds a filename |
| GET | `/videos/{id}/aggregates` | which aggregates exist |
| GET | `/videos/{id}/aggregates/{name}` | one aggregate |
| GET | `/videos/{id}/frames/{index}` | one stored frame as JPEG |
| GET | [`/prompts`](http://localhost:8000/prompts) | every question, and the shapes one may answer in |
| GET | `/prompts/{name}` | one question, with the schema a call would get |
| POST | `/prompts` | add a custom question. **201** |
| DELETE | `/prompts/{name}` | remove a custom one. **204** |
| POST | `/search` | scope is a set of videos; `level` picks moment or video |
| GET | [`/db/status`](http://localhost:8000/db/status) | can this deployment read Postgres, and what is in it |
| GET | [`/db/tables`](http://localhost:8000/db/tables) | every table, what it holds, its deployed columns |
| POST | `/db/query` | one page of one table: filters, order, offset, count |
| GET | [`/app/`](http://localhost:8000/app/) | the web client. `/` redirects here |

The linked rows are the GETs that take no parameters, so a browser can open
them as they stand. The rest cannot be followed: a path with `{id}` needs a
real video id substituted, and every `POST` and `DELETE` needs a body. Use
[`/docs`](http://localhost:8000/docs) -- each operation has a **Try it out**
button -- or `curl`:

```bash
curl -s http://localhost:8000/videos
curl -s http://localhost:8000/videos/Chernobyl/artifacts/media
curl -s http://localhost:8000/videos/Chernobyl/frames/125 -o frame.jpg

curl -s -X POST http://localhost:8000/db/query \
  -H 'Content-Type: application/json' \
  -d '{"table": "chunks", "limit": 5}'
```

## The web client

`web/` is three files a browser reads in order -- `index.html`, `app.js`,
`app.css`. No build step, no framework, one origin. `main.py` mounts it at
`/app` with `cache-control: no-cache`, so an edited file is picked up on reload
rather than served from a browser's guessed freshness lifetime.

**Nothing about the pipeline is written down in it.** Every parameter form is
generated from `/capabilities.parameters`, the legal values come from the
registries, and a blank field is *omitted* rather than sent as null so the
component's own default applies. The one widget written by hand is the
custom-shape builder, because a field is a name, a type, a description and
optionally a nested map, and a text box does not say so.

Four tabs: **Run** drives `POST /videos/{id}/run/{component}` one stage at a
time and polls the job; **Prompts** lists, builds and deletes questions;
**Search** posts `/search`; **Data** reads `/db/status` and `/db/query`.
Uploading is how `media` runs, since it is the one component the run route
cannot address.

## Reading the rows

**The read key, never the write key.** Nothing in `browse.py` writes, so
nothing in it holds a key that could.

**Columns are probed, not restated.** One `select * limit 1` per table per
process gives the deployed shape. A hard-coded list would be a second copy of
`install.sql` -- and that file is re-run against live databases precisely
because they drift. What is stated is which columns are too wide to send by
default (`embeddings.embedding` and `fts`, `transcripts.words/segments/turns`,
`cuts.scores`), which is a judgement about size rather than a claim about the
schema, so it cannot go stale the way a column list can.

**Every page carries the size of the whole result.** "20 rows" and "20 of 4,812
rows" are different answers, and a client showing the first as the second is
lying about coverage.

**`/db/status` answers rather than leaves it to be discovered.** RLS enabled
with no policy denies reads *silently* -- zero rows, no error -- so a client
that could not tell "not configured" from "nothing ingested" would show an
empty table for both.

## Why some things are the way they are

**Validation is synchronous even though the work is not.** `workflow.validate`
returns problems as a list, so a policy that needs a stream the run is not
reading is a 422 the caller sees at once rather than a job that fails a minute
later. What cannot be known without opening the file -- whether a track carries
speech -- still fails inside the job, because that is a property of the media
rather than of the request.

**The id comes from the filename, not the client.** It keys every table, every
output directory and every Qdrant payload, so it is derived and sanitised
rather than accepted.

**`/videos` is read from disk, not remembered.** A restarted server still knows
everything it produced. `GET /jobs` says the converse plainly: job records die
with the process, artifacts do not.

**The export list holds what exists, not what could exist.** An audio-only
video advertises no manifest rather than offering a link that 404s -- a broken
link reads as breakage rather than as a stage that never ran.

**`?download=1` only adds a `Content-Disposition`.** Content negotiation would
be tidier, but a browser cannot set an `Accept` header on a plain link.

**A sampler may carry several questions.** `sampler=clip:[text,scene]` is one
pass over the video answering two questions about the frames it kept, and
specs naming the same sampler merge into one pass. The manifest is keyed by
run, `descriptions` and `/search` by answer (`clip:text`), so `--sampler
clip:text` narrows exactly as it did.

**A custom prompt names a shape or brings its own.** The shape carries the
response schema. `POST /prompts` takes either `shape` (a name from
`/prompts.shapes`) or `fields` -- a builder, not JSON Schema: `{"type": "text"}`
or `{"type": "list"}`, plus `of` to make each list entry an object and `one_of`
to fix the vocabulary. The schema is generated from it, because the call goes
out with `strict: true` and a raw schema the API refuses would fail after the
frames are read. `/prompts.shapes` marks each shape `builtin`, and
`field_types` and `limits` publish what a builder may contain.

Built-ins use exactly this vocabulary -- `yolo` is not a special case, it is the
`people` shape. A custom shape is stored under its question's name, is deleted
with it, may not take a built-in shape's name, and never claims `fallback`.
Each entry lists the `fields` it answers; two questions may answer the same
field, and both answers are kept -- a pairing is independent of every other
pairing on the chunk.

**Built-ins cannot be edited or deleted, and that is a 409 rather than a 404.**
They ship in the package so that every deployment's `yolo` means the same
thing; a request that could shadow one would make a run unreproducible from the
repo. 409 because the name exists and the request is well-formed -- there is
nothing to correct except which name it asks for. Custom questions live in
`data/prompts.json`, which the API writes and git ignores.

**Deleting a question does not touch the descriptions it produced.** A
description cost a paid call and records the question it was asked, so removing
the question does not make the answer untrue; it only stops new runs asking it.

**Scope is a set, not an argument.** `video_ids` names the videos to search and
omitting it searches every one; `video_id` is the one-element shorthand.
Searching one video, three, or all of them is the same question asked over a
different set, so it is one endpoint and a set of one is not a special case.

**Moments are keyed by `(video_id, chunk_id)`.** A chunk id is an index into
*one* video's grid, so grouping on the id alone fused chunk 0 of one video with
chunk 0 of another into a single "moment" carrying two unrelated clips -- and
the agreement bonus then scored the collision *above* either real answer. It
was invisible while the scope was one video, which is exactly why it was
written that way.

**`level` picks granularity, and never mixes the two.** `moment` ranks chunks
and honours every filter; `video` ranks whole videos by their summary out of
`video_embeddings`. One endpoint, but never one ranking: a whole-video "moment"
beside real ones is a result nobody can play. The moment filters narrow *inside*
a video, so at `level=video` they are reported back under `ignored` rather than
silently dropped.

**`/search` narrows nine ways**, and `/capabilities.search` publishes them so
a client generates its form rather than restating the list:

| | |
|---|---|
| `sampler` | one **pairing**, `clip:text` |
| `question` | one question across every sampler that asked it, `text` |
| `strategy` | one sampler's whole output, `clip` |
| `chunk_ids` | a set of chunks -- the drill-down |
| `window` | widen `chunk_ids` by N neighbours each side |
| `after` / `before` | seconds |
| `structured` | exact values, `{"severity": "severe"}` |
| `candidates` | units ranked per half before fusion |

None is a prefix match on an id: a bare `sampler_id` like `clip` means the
question *is* the strategy name, so a unit carries both halves as their own
fields. Measured across both backends, all six filters select **identical chunk
sets**; rank order differs on two, which is the lexical halves diverging rather
than a filter disagreeing.

**Time is not a field on a vector.** `after`/`before` are resolved to chunk ids
through the grid before the query, so one mechanism reaches both stores and
Qdrant needs no payload change -- payload is written only on upsert, so adding a
span there would need a forced re-index.

**`structured` is only useful where a shape fixed the vocabulary with
`one_of`.** `/capabilities.search.structured_fields` lists exactly those
fields and their values, read off the shapes. On free text a filter for
`cashier` also matches `cashier or customer near checkout`.

**A moment carries the ranks each half gave it.** `ranks[sampler_id] =
{dense, text}`, and `text: null` means the lexical half was silent. Without
them a score is uninterpretable: measured on this corpus, a nonsense query
scores 0.1136 against a real one's 0.1294. There is no relevance floor, so the
ranks are the signal.

**`level=video` is Postgres only** -- that is where `embed` writes the
summaries, from the `summary` aggregate. 4/4 correct on the test corpus.

**`/search` names its embedder.** It must be the one that built the index: a
mismatch across widths fails loudly, but two models of the same width return a
well-formed ranking that means nothing. The embedder key is in the collection
name, so a wrong name searches a collection that does not exist.
