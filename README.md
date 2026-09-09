# FalCONvar

Video RAG ingestion. A video goes in; a searchable index of moments comes out.

Both the picture and the soundtrack are read onto **one chunk grid**, so a
search returns a span you can play rather than two answers that disagree about
where it starts.

```bash
pip install -e .                                # then python -m from anywhere
python -m falconvar.workflow media/video.mp4 --policy vad --sampler clip,yolo:overview
python -m falconvar.rag.retrieve "the moment the reactor exploded" video
python -m falconvar.rag.retrieve "..." video --question text   # across samplers
python -m uvicorn api.main:app --port 8000      # /docs for the schema
```

The HTTP surface drives the same thing: upload and run the whole pipeline, or
drive one component at a time with its own settings, then read back the grid,
the transcript, the frames each sampler kept, every description, the
aggregates, the raw rows, and hybrid search. See `docs/ROUTES.md`.

## How it works

Nine components, each reading files and writing files. Nothing imports another.

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

**The grid is a component, not a side effect.** Which modality decides it is a
policy, and the run order falls out of what that policy depends on rather than
from a rule:

| policy | boundaries come from | so what runs first |
|---|---|---|
| `uniform` | arithmetic over the duration | **nothing** |
| `scene` | frame-to-frame content change | a scene pass over the picture |
| `vad` | silences between speech | the audio pass |
| `speaker` | where the voice changes | the audio pass |

Everything downstream is keyed by `(video_id, chunk_id)`, and the grid is
stored exactly once.

## Which frames, and what to ask

**Samplers** decide which frames are worth describing — `clip` when the scene
changes · `yolo` when the people change · `objects` for an open vocabulary ·
`text` when the writing changes · `uniform` on a stride.

**Questions** are independent of them, and any pair is legal as `name:prompt`:

```bash
--sampler uniform:text        # read the screen on a stride, no OCR at ingest
--sampler yolo:overview       # frames where people changed, asked for prose
--sampler "clip:[text,scene]" # ONE pass over the video, two questions about it
--sampler clip:text+scene     # the same, for shells that glob brackets
```

Selecting frames is the expensive half, so a sampler runs once however many
questions it carries — and `clip:text,clip:scene` merges into that same single
pass rather than doing the work twice.

Unpaired, a sampler is asked the question named after it. Every pairing is
independent: two questions that answer the same field both answer it, and both
answers are kept — overlap is your choice, and the answers differ.

**Questions are data, and you can add your own.** A question is an instruction
plus a *shape*, and the shape carries the response schema — the built-ins are
written in the same terms, so `yolo` is just the `people` shape. Built-ins ship
in `falconvar/describe/prompts.json`; anything you add lands in
`data/prompts.json` and cannot shadow one.

```bash
curl -X POST localhost:8000/prompts -H 'Content-Type: application/json' -d '{
  "name": "safety", "shape": "scene",
  "instruction": "These {n} frames span {span}. List every safety hazard visible."}'

python -m falconvar.workflow site.mp4 --sampler clip,uniform:safety
```

Editing one question re-describes only the pairs that used it; adding one costs
nothing.

## Retrieval

Both modalities land in one index: a description and a transcript chunk are
both text with a span, so `--sampler transcript` narrows a search to what was
said exactly as `--sampler yolo` narrows it to who was seen.

Ranking is RRF twice — a dense and a lexical ranking fused per unit, then the
units of a chunk fused into a moment. Two backends, both hybrid: **`qdrant`**
(embedded or served) and **`supabase`** (pgvector + `ts_rank_cd`).

## Aggregates

What retrieval cannot answer, because embeddings cannot count. Three tiers, run
cheapest first: `free` is arithmetic (`stats`, `speakers`, `coverage`), `local`
adds GPU models (`ner`, `sentiment`), `llm` adds paid calls (`summary`,
`chapters`, `events`).

## The API

20 routes. Two ways to run a video, and the choice is about how much you want
to tune:

```bash
# the whole pipeline, on workflow defaults                            202
curl -X POST localhost:8000/videos -F file=@video.mp4      -F policy=vad -F sampler=clip,uniform:text -F tier=llm

# or register it and drive the stages yourself, with per-stage settings
curl -X POST localhost:8000/videos -F file=@video.mp4 -F run=false      # 201
curl -X POST localhost:8000/videos/video/run/boundaries      -H 'Content-Type: application/json'      -d '{"params": {"policy": "scene", "min_s": 30, "max_s": 60}}'
curl -X POST localhost:8000/videos/video/run/video      -d '{"params": {"sampler": "uniform:overview,clip:[mood,motion]",
                     "per_second": 1, "every_n": 5}}'
```

`GET /capabilities` publishes every registry, the defaults, **and each
component's parameters with their types and defaults** — so a UI is generated
from it rather than kept in step with it. `POST /search` narrows by pairing
(`clip:text`) or by question (`text`, across every sampler that asked it).
`docs/ROUTES.md` has the reasoning; `/docs` is the authority on shapes.

`GET /db/tables` and `POST /db/query` read the rows a run wrote -- filtered,
ordered and paged, with the size of the whole result beside the page -- under
the publishable key, so they see what a reader with the read grants sees.

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
recovery/          STANDALONE: rebuild a store from a manifest + the video
db/
  supabase/        install.sql · reset.sql
  json/            document schemas, generated from the dataclasses
data/              everything a run writes; gitignored
docs/ROUTES.md     the HTTP surface
```

102 Python files, ~11.0k lines.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
cp .env.example .env        # OpenAI key; Supabase and HF tokens if used
```

Run `db/supabase/install.sql` in the SQL editor and add `falconvar` to
**Settings → API → Exposed schemas** if you want the Postgres backend. The file
is idempotent and is also how a schema change is applied — re-run the whole
thing. `db/supabase/reset.sql` drops everything first, and is the only
destructive one.

## State

Runs end to end on real models — Whisper, pyannote, CLIP, YOLO, GPT, GLiNER —
writing documents to files and Postgres and vectors to Qdrant and pgvector.

Last verified through the API from wiped local, Qdrant and Postgres state,
driving every stage with its own settings: a `scene` grid with a 30 s floor
gave 4 chunks of 39.8–55.6 s; `uniform:overview` at a 5 s stride and
`clip:[mood,motion]` (two custom prompts added over HTTP) gave 12 descriptions
and 16 searchable units; every Postgres table exact under both keys;
`recovery.recreate` 76/76 byte-identical.

Not built: a test suite, an import checker, an eval harness, video-level
embedding, and sampler threshold calibration. See `CLAUDE.md` for what is
measured and what is not.
