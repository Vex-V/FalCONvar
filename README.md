# FalCONvar

Media goes in; a searchable index of moments comes out — plus the video-level
answers that searching cannot give you.

```
                         ┌──────────────────────────────────────┐
   video ─── ingest ───► │ manifest + frames │ ──── describe ───┤
                         └──────────────────────────────────────┤
                                                                ├─ embed ─► vectors ─► retrieve ─► moments
   audio ─── listen ───► │ transcript                           │
                         └──────────────────────────────────────┘
                                          └──── aggregate ─────► summary, chapters, events, stats…
```

Both streams land on **one chunk grid**, so `chunk_id 7` means the same twenty
seconds whether you ask what was seen or what was said. A search can then be
answered by two independent accounts of the same moment.

---

## What each stage does

| stage | in | out | what it decides |
|---|---|---|---|
| **ingest** | a video | `manifest.json`, `store/` | which frames are worth describing, **and why** |
| **describe** | manifest + frames | `descriptions.json` | one VLM answer per (chunk, sampler) — a different question per sampler |
| **listen** | a soundtrack | `transcript.json` | words with timestamps, who spoke, cut to the grid |
| **embed** | descriptions + transcript | vectors | what text to embed, and what has actually changed |
| **retrieve** | a question | ranked moments | dense + lexical, fused twice |
| **aggregate** | everything above | `aggregates/*.json` | the whole-video questions retrieval is bad at |

Nothing looks at the video except `ingest`. Every later stage reads only what
the one before it wrote, which is why any of them can be re-run alone.

## Capabilities

**Samplers** — why a frame was kept, which decides which question gets asked
about it:

`clip` has the scene changed · `yolo` have the people changed · `objects` have
things moved or appeared (open-vocabulary, you supply the class list) · `text`
has the writing changed · `uniform` every Nth decimated frame.

**Questions** are separate from samplers, and any pair is allowed as
`name:prompt`: `uniform:text` reads the screen on a stride without running OCR
at ingest at all, `yolo:overview` keeps frames where the people changed and
asks for prose rather than the structured people call. Unpaired, a sampler is
asked its own question. A stride counts the frames the sampler
was offered, so the cadence in seconds is `--every-frames` over
`--per-second`; a cadence in seconds regardless is `--min-interval`.

**Chunk grids** — one per run, and either modality may decide it:

`uniform` arithmetic, nothing propagates · `scene` the video pass, from frame
content · `vad` the audio pass, in the gaps between speech · `speaker` the
audio pass, where the voice changes.

**Audio** — Whisper transcription and pyannote diarization as separate passes,
joined by word midpoints. Scanned whole, then cut: word-level timestamps make
re-segmenting free, which is what lets either modality own the boundaries.

**Retrieval** — dense vectors and BM25, fused by RRF inside Postgres, then
descriptions folded into moments by `best + ½·second`. Neither half knows which
kind of question it was handed, which is the argument for fusing rather than
choosing. Qdrant is available and dense-only; it says so when you pick it.

**Aggregates** — `stats`, `speakers`, `novelty` (free: arithmetic, no model),
`ner`, `sentiment` (local GPU), `summary`, `chapters`, `events` (paid LLM
calls). Everything at or below the chosen tier runs, cheapest first; anything
whose inputs have not changed is left alone. A long video's summary is reduced
hierarchically, and **every layer is kept** with the span it covers — a leaf
summary is the only account of the video between one chunk and the whole file.

**Audio-only or video-only.** Either stream can be switched off. Audio alone
writes a transcript and no manifest, skips describe entirely — a transcript is
already the text that stage would produce — and still embeds, searches and
aggregates.

## Quick start

```bash
pip install -r requirements.txt
python -m ver2.imports                 # after ANY install: what actually loaded
cp .env.example .env                   # keys for OpenAI and Supabase

python -m uvicorn api.main:app --port 8000
```

Then <http://localhost:8000/app> for the browser client, or
<http://localhost:8000/docs> for the API schema.

From the command line, one file end to end:

```bash
python -m ver2.driver media/x.mp4 --sampler clip --chunking uniform
python -m ver2.driver media/x.mp4 --no-video --chunking vad    # sound alone
python -m ver2.video.describe.driver out/x/manifest.json --describer openai
python -m ver2.embed.driver out/x/descriptions.json
python -m ver2.aggregate.driver x --tier llm
python -m ver2.retrieve.driver "people at the checkout" --moments 3
```

Everything one file produces lives under `out/<video-id>/`. Grouped by video
rather than by artifact type, so one video's whole output is one thing to
inspect, copy or delete.

## Layout

```
web/            the browser client: one page, no build step
api/            HTTP in front of it all; slow stages queued, search immediate
ver2/
  driver.py     the CLI for both streams
  orchestrate.py both streams, one grid — what the CLI and the API both call
  timeline.py   the shared chunk grid: spans + the policy that produced them
  video/
    ingest/     source, chunker, samplers, output, pipeline, calibrate
    describe/   input, describers, vlm/prompts.py, output, reader
  audio/        source, transcribe, diarize, align, segment, output, reader
  embed/        SHARED: descriptions and transcripts both land here
  retrieve/     SHARED: query -> ranked descriptions -> ranked moments
  aggregate/    video-level structure over what the chunk stages wrote
  recovery/     STANDALONE: rebuild a store from a manifest + the video
eval/           the measurements behind the choices, reproducible
schema.sql      runnable DDL: every table, function and RLS policy
docs/           SCHEMAS.md · ROUTES.md · RUN.md
```

`recovery/` imports nothing from `ver2`, and `imports.py` enforces it by
parsing the files. Hand someone those three files, a video id and the video,
and they rebuild the frame store byte for byte with `av`, `opencv` and `numpy`.

## Where to read next

- **[docs/RUN.md](docs/RUN.md)** — every CLI, the API, the web app, and the
  order to run them in.
- **[docs/ROUTES.md](docs/ROUTES.md)** — the HTTP surface and the reasoning
  behind each route.
- **[docs/SCHEMAS.md](docs/SCHEMAS.md)** — what every field means, in the local
  JSON and in Postgres alike.
- **[CLAUDE.md](CLAUDE.md)** — the invariants, and the measurements that
  produced them. Every design decision here has a number behind it; this is
  where the numbers are.

## Storage

Local JSON is always written and is the primary sink. Postgres (Supabase) is
optional and additive — `--sink file,supabase` writes both, file first. Vectors
go to pgvector by default because it is the half with a lexical index; Qdrant
runs embedded with no server and is dense-only.

Run `schema.sql` once against a fresh database. It is idempotent and repairs
its own generated columns.

## Not built

- A local embedder has been written and guarded but never run; every
  measurement here is OpenAI.
- Structured fields reach both indexes and both can filter on them, but the
  values are free text, so a filter for `cashier` matches everything. They need
  an `enum` first.
- Live sources. `Frame.gap_before` and `Frame.discontinuity` are the seams.
- Tests. Verification is `imports.py`, recreate's byte-comparison, and the
  measurements in `eval/`.
