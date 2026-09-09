# FalCONvar

Video RAG ingestion. A video goes in; a searchable index of moments comes out.

Both the picture and the soundtrack are read onto **one chunk grid**, so a
search returns a span you can play rather than two answers that disagree about
where it starts.

```bash
pip install -e .                                # then python -m from anywhere
python -m falconvar.workflow media/video.mp4 --policy vad --sampler clip,yolo:overview
python -m falconvar.rag.retrieve "the moment the reactor exploded" video
python -m uvicorn api.main:app --port 8000      # /docs for the schema
```

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
--sampler uniform:text      # read the screen on a stride, no OCR at ingest
--sampler yolo:overview     # frames where people changed, asked for prose
```

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
api/               HTTP: routes, dispatch, one background worker
recovery/          STANDALONE: rebuild a store from a manifest + the video
db/
  supabase/        install.sql · reset.sql
  json/            document schemas, generated from the dataclasses
data/              everything a run writes; gitignored
docs/ROUTES.md     the HTTP surface
```

101 files, ~10.4k lines.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
cp .env.example .env        # OpenAI key; Supabase and HF tokens if used
```

Run `db/supabase/install.sql` in the SQL editor and add `falconvar` to
**Settings → API → Exposed schemas** if you want the Postgres backend.

## State

The pipeline runs end to end on real models — Whisper, pyannote, CLIP, YOLO,
GPT, GLiNER — writing to files and Postgres, with vectors in Qdrant and
pgvector. `recovery.recreate` byte-compares a rebuilt frame store against the
original and is the end-to-end oracle.

Not built: a test suite, an import checker, an eval harness, and sampler
threshold calibration. See `CLAUDE.md` for what is measured and what is not.
