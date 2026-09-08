# Routes

The HTTP surface is `api/`, over `falconvar`. Run it with:

    python -m uvicorn api.main:app --port 8000

`/docs` is the generated schema and is the authority on request and response
shapes; this file records the reasoning the schema cannot carry.

## Three shapes of route

**Immediate** -- reading what exists, and searching. Milliseconds.

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
| GET | `/health` | liveness, and how many jobs are queued |
| GET | `/capabilities` | every registry, plus the defaults read off `workflow.Options` |
| POST | `/videos` | upload a file and queue the whole pipeline. **202** |
| POST | `/videos/{id}/run/{component}` | queue one component. **202** |
| GET | `/jobs` | every job this process remembers |
| GET | `/jobs/{job_id}` | one job: state, stage, history, stats |
| GET | `/videos` | every video with an output directory |
| GET | `/videos/{id}` | what this video has, with URLs |
| GET | `/videos/{id}/artifacts/{name}` | one document. `?download=1` adds a filename |
| GET | `/videos/{id}/aggregates` | which aggregates exist |
| GET | `/videos/{id}/aggregates/{name}` | one aggregate |
| GET | `/videos/{id}/frames/{index}` | one stored frame as JPEG |
| POST | `/search` | ranked moments |

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

**`/search` names its embedder.** It must be the one that built the index: a
mismatch across widths fails loudly, but two models of the same width return a
well-formed ranking that means nothing. The embedder key is in the collection
name, so a wrong name searches a collection that does not exist.
