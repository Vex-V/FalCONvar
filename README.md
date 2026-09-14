# FalCONvar

Video RAG ingestion. A video goes in; a searchable index of moments comes out.

Both the picture and the soundtrack are read onto **one chunk grid**. Everything
downstream is keyed by `(video_id, chunk_id)`.

## Install

```bash
pip install -r requirements.txt
pip install -e .              # then `python -m falconvar.…` from any directory
cp .env.example .env          # OpenAI key; Supabase and HF tokens if used
```

For the Postgres backend, run `db/supabase/install.sql` in the SQL editor and
add `falconvar` to **Settings → API → Exposed schemas**. The file is idempotent
and is also how a schema change is applied — re-run the whole thing.
`db/supabase/reset.sql` drops everything first, and is the only destructive one.

## Quickstart

```bash
python -m falconvar.workflow samples/video.mp4 --policy vad --sampler clip,yolo:overview
python -m falconvar.rag.retrieve "the moment the reactor exploded" video
python -m uvicorn api.main:app --port 8000     # the app at /, /docs for the schema
```

## The pipeline

Nine components. Each reads files and writes files, and none imports another.
Every one has the signature `run(video_id, ...) -> Produced`.

| # | component | reads | writes |
|---|---|---|---|
| 1 | `media` | the media file | `media.json` |
| 2 | `audio` | `media.json` | `transcript.raw.json` |
| 3 | `boundaries.evidence` | `media.json` *or* `transcript.raw.json` | `cuts.json` |
| 4 | `boundaries` | `media.json` + `cuts.json` | **`timeline.json`** |
| 5 | `video` | `media.json` + `timeline.json` | `manifest.json`, `store/` |
| 6 | `cut` | `transcript.raw.json` + `timeline.json` | `transcript.json` |
| 7 | `describe` | `manifest.json` + `store/` | `descriptions.json` |
| 8 | `embed` | `descriptions.json` + `transcript.json` | vectors, `embedded.json` |
| 9 | `aggregate` | everything above | `aggregates/*.json` |

Everything a run writes lands in `data/out/<video-id>/`:

```
media.json           the two streams and their addressing
transcript.raw.json  words, segments and speaker turns — no chunk ids
cuts.json            boundary evidence, and the score series behind it
timeline.json        THE GRID: every chunk's span
manifest.json        which frames each sampler kept
store/               those frames as JPEG, named by read index
transcript.json      what was said, cut to the grid
descriptions.json    one model answer per (chunk, sampler:question)
embedded.json        the text that went into the index, without the vectors
aggregates/*.json    one file per aggregator
```

Vectors go to `data/out/_qdrant/` or to Postgres, not into the video's
directory.

### The grid

`--policy` decides where boundaries come from, which decides what has to run
first:

| policy | boundaries come from | runs first |
|---|---|---|
| `uniform` | arithmetic over the duration | **nothing** |
| `scene` | frame-to-frame content change | a scene pass over the picture |
| `vad` | silences between speech | the audio pass |
| `speaker` | where the voice changes | the audio pass |

Three guards apply under every policy: `--min-chunk` merges spans below the
floor, `--max-chunk` splits those above the ceiling (defaulting to
`--chunk-duration`), and a final chunk shorter than a quarter of the chunk
length is merged into the one before it. A floor above the ceiling is refused.

### One component at a time

Per-stage tuning lives on these, not on `workflow`.

```bash
python -m falconvar.media samples/x.mp4
python -m falconvar.audio <id> --transcriber whisper --diarizer pyannote
python -m falconvar.boundaries <id> --policy scene --evidence --stride 5 --threshold 27
python -m falconvar.boundaries <id> --calibrate            # what each threshold costs
python -m falconvar.boundaries <id> --retune 45            # re-threshold, runs no model
python -m falconvar.boundaries <id> --policy scene --min-chunk 30 --max-chunk 60
python -m falconvar.video <id> --sampler "clip:[text,scene]" --per-second 1
python -m falconvar.video <id> --sampler uniform:text --every-frames 5
python -m falconvar.cut <id>
python -m falconvar.describe <id> --describer openai --limit 5     # costs money
python -m falconvar.rag.embed <id> --index qdrant,supabase
python -m falconvar.aggregate <id> --tier llm
```

`--sink file,supabase` writes the document to both, on every stage that writes
one; `embed` takes `--index qdrant,supabase` instead. `--calibrate` and
`--retune` read the cached score series and run no model.

## Samplers and questions

**Samplers** decide which frames get described:

| sampler | keeps a frame when |
|---|---|
| `clip` | the scene changes |
| `yolo` | the people change |
| `objects` | an open-vocabulary detection changes (`--vocabulary`) |
| `text` | the writing on screen changes |
| `uniform` | every Nth decimated frame (`--every-frames`) |

**Questions** decide what is asked about those frames, and the two are
independent — pair any sampler with any question as `name:question`:

```bash
--sampler uniform:text          # read the screen on a stride
--sampler yolo:overview         # frames where people changed, asked for prose
--sampler "clip:[text,scene]"   # ONE pass over the video, two questions about it
--sampler clip:text+scene       # the same, for shells that glob brackets
```

A sampler runs once however many questions it carries, and
`clip:text,clip:scene` merges into that same single pass. Unpaired, a sampler is
asked the question named after it. Two questions that answer the same field both
answer it, and both answers are stored and searched separately.

Rate limits are applied before a sampler runs: `--min-interval` in seconds and
`--max-per-chunk`. Every chunk keeps at least one frame.

### Adding a question

A question is an instruction plus a **shape**, and the shape carries the
response schema. The shipped shapes are `scene`, `people`, `objects`, `text` and
`prose`; `yolo` is the `people` shape, `overview` is `prose`. Built-in questions
live in `falconvar/describe/prompts.json`; anything you add lands in
`data/prompts.json` and may not shadow a built-in.

```bash
curl -X POST localhost:8000/prompts -H 'Content-Type: application/json' -d '{
  "name": "safety", "shape": "scene",
  "instruction": "These {n} frames span {span}. List every safety hazard visible."}'

python -m falconvar.workflow site.mp4 --sampler clip,uniform:safety
```

An instruction may use `{n}`, `{span}` and `{vocabulary}`; anything else is
refused at submission. Editing a question re-describes only the pairs that used
it, and adding one re-describes nothing.

## Retrieval

Descriptions and transcript chunks both become units in one index, keyed
`(video_id, chunk_id, sampler_id)`. A transcript chunk is
`sampler_id = transcript`, so it filters exactly like any visual pairing.

```bash
python -m falconvar.rag.retrieve "..." <id>
python -m falconvar.rag.retrieve "..." <id> --sampler clip:text   # one pairing
python -m falconvar.rag.retrieve "..." <id> --question text       # across samplers
python -m falconvar.rag.retrieve "..." <id> --strategy clip       # one sampler's output
python -m falconvar.rag.retrieve "..." <id> --chunks 4,6 --window 1   # drill-down
python -m falconvar.rag.retrieve "..." <id> --after 90 --before 130   # a time window
python -m falconvar.rag.retrieve "..." <id> --where severity=severe   # a fixed vocabulary
```

A time window is resolved to chunk ids through the grid, so a span is stored in
one place and both backends get one filter. `--where` only means something
where a shape fixed the values with `one_of`.

`POST /search` takes `video_ids` as its scope — omit it for every video, name
one, or name three; a set of one is not a special case. `level: "video"` ranks
whole videos by their summary out of `video_embeddings` instead of ranking
chunks.

Ranking is RRF twice: a dense and a lexical ranking fused per unit, then the
units of a chunk fused into a moment as `1/(k+best) + 0.5/(k+second)` at k=10.
The number a search returns is a rank fusion, not a similarity.

Two backends, both hybrid: **`qdrant`** (embedded at `data/out/_qdrant/`, or
served with a url) and **`supabase`** (pgvector + `ts_rank_cd`).

## Aggregates

Video-level answers, one file per aggregator. A tier is a ceiling and they run
cheapest first, so `--tier llm` runs all three tiers.

| tier | aggregators | |
|---|---|---|
| `free` | `stats`, `speakers`, `coverage` | arithmetic |
| `local` | `ner`, `sentiment` | GPU models |
| `llm` | `summary`, `chapters`, `events` | paid calls |

`speakers` needs a transcript and `chapters` needs `summary`; an aggregator
whose input is missing is skipped with the reason rather than failing.

## The API

21 routes. `python -m uvicorn api.main:app --port 8000`, then `/` for the
client or `/docs` for the schema.

The client is three files under `web/` with no build step. Every parameter
form is generated from `/capabilities`, so nothing about the pipeline is
written down twice.

Two ways to run a video, and the choice is about how much you want to tune:

```bash
# the whole pipeline, on workflow defaults                             202
curl -X POST localhost:8000/videos -F file=@video.mp4 \
     -F policy=vad -F sampler=clip,uniform:text -F tier=llm

# or register it and drive the stages yourself, with per-stage settings
curl -X POST localhost:8000/videos -F file=@video.mp4 -F run=false   # 201
curl -X POST localhost:8000/videos/video/run/boundaries \
     -H 'Content-Type: application/json' \
     -d '{"params": {"policy": "scene", "min_s": 30, "max_s": 60}}'
curl -X POST localhost:8000/videos/video/run/video \
     -H 'Content-Type: application/json' \
     -d '{"params": {"sampler": "uniform:overview,clip:[mood,motion]",
                     "per_second": 1, "every_n": 5}}'
```

`params` is passed to the component as keyword arguments, so every component
setting is reachable over HTTP. Order on the second path is yours:
`audio · boundaries.evidence · boundaries · video · cut · describe · embed ·
aggregate`.

Anything that decodes, transcribes or pays a model is queued one job at a time
and answers **202** with a job id; poll `GET /jobs/{id}` for `stage` (running),
`history` (finished) and `detail`. Job records die with the process; artifacts
do not, and `GET /videos` reads them from disk.

- `GET /capabilities` — every registry, the defaults, and each component's
  parameters with their types and defaults
- `GET /videos/{id}` · `/artifacts/{name}` · `/aggregates/{name}` ·
  `/frames/{index}` — only the artifacts that exist are listed
- `POST /search` — narrows by pairing (`clip:text`) or by question (`text`,
  across every sampler that asked it)
- `GET|POST|DELETE /prompts` — built-ins refuse edits with 409
- `GET /db/tables` · `POST /db/query` — the rows a run wrote, filtered, ordered
  and paged under the publishable key, with the size of the whole result

`docs/ROUTES.md` is the full surface; `/docs` is the authority on shapes.

## Layout

```
falconvar/
  workflow.py      the whole run, as a list of component calls
  shared/          paths · documents · sinks · env · llm · db · rows · schemas
  media/           1  split
  audio/           2  source · reader · models · backends/
  boundaries/      3+4 scenes · speech · grid
  video/           5  reader · decimate · store · pipeline · samplers/
  cut/             6
  describe/        7  prompts · library · prompts.json · frames · backends/
  rag/embed/       8  units · embedders · indexes/ · readable
  rag/retrieve/   10
  aggregate/       9  statistics/ · model/ · llm/
api/               HTTP: routes, dispatch, one background worker, db reads
web/               the client at /app: one page, no build step
recovery/          STANDALONE: rebuild a store from a manifest + the video
db/
  supabase/        install.sql · reset.sql
  json/            document schemas, generated from the dataclasses
data/              everything a run writes; gitignored
docs/ROUTES.md     the HTTP surface
```

102 Python files, ~11.0k lines.

## State

Runs end to end on real models — Whisper, pyannote, CLIP, YOLO, GPT, GLiNER —
writing documents to files and Postgres and vectors to Qdrant and pgvector.

Last verified through the API from wiped local, Qdrant and Postgres state,
driving every stage with its own settings: a `scene` grid with a 30 s floor gave
4 chunks of 39.8–55.6 s; `uniform:overview` at a 5 s stride and
`clip:[mood,motion]` (two custom prompts added over HTTP) gave 12 descriptions
and 16 searchable units; every Postgres table exact under both keys;
`recovery.recreate` 76/76 byte-identical.

Not built: a test suite, an import checker, a corpus big enough for
`eval/harness.py` to give a result rather than a direction, and sampler
threshold calibration. `CLAUDE.md` is the reasoning
behind every decision here, and records what is measured and what is not.
