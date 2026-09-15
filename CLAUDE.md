# FalCONvar — working notes

Video RAG ingestion. A video goes in; both the picture and the soundtrack are
read onto **one chunk grid**, and a searchable index of moments comes out.

This file is the *why*. File headers say what a module does; the reasoning, the
measurements and the traps live here.

---

## Layout

```
falconvar/
  workflow.py      the whole run, as a list of component calls
  shared/          paths · env         everything below imports these
    contracts/     documents · schemas   what components hand each other
    storage/       sinks · db · rows     where a document goes
    models/        providers · llm       who answers a model call
                   paths and documents import nothing
  media/         1 split: what streams the file carries
  audio/         2 source · reader · models
    backends/      whisper · pyannote · cuda
  boundaries/   3+4 scenes (picture) · speech (soundtrack) · grid
  video/         5 reader · decimate · store · pipeline
    samplers/      base · uniform · scene · people · objects · ocr
      perception/  detectors · descriptors · embedders. Every model weight
                   lives below this line and none above it.
  cut/           6 the transcript, onto the grid
  describe/      7 prompts (logic) · library (the vocabulary) · frames · reader
    prompts.json   BUILT-IN questions and shapes; shipped, read-only
    backends/      stub · model (every provider, through shared/models/llm)
  rag/embed/     8 units · embedders · remote · local · readable
    indexes/       qdrant · supabase
  rag/retrieve/ 10 search
  aggregate/     9 base · rendering · linking (who is who, no model)
    statistics/    stats · speakers · coverage      free: arithmetic
    model/         ner · sentiment                  local: GPU models
    llm/           summary · chapters · events · entities   llm: paid calls
api/               main (routes) · service (dispatch) · jobs (one worker)
                   browse (read-only queries over the rows a run wrote)
web/               the client at /app. No build step: index.html · app.js ·
                   app.css. Every form generated from /capabilities
recovery/          STANDALONE: recreate.py, imports nothing from the pipeline
db/
  supabase/        install.sql · reset.sql
  json/            document schemas, generated from the dataclasses
data/              everything a run writes; gitignored
  out/<id>/        media, transcript.raw, cuts, timeline, manifest, store,
                   transcript, descriptions, embedded, aggregates/
  out/_qdrant/     the embedded vector store, beside the videos not inside one
  uploads/         what the API parked until a run read it
  prompts.json     custom questions, added through the API
  providers.json   model endpoints added or overridden; names key variables, never keys
weights/           detector and embedder checkpoints; a cache, not output
docs/ROUTES.md     the HTTP surface
```

111 Python files, ~13.7k lines.

## Commands

```bash
python -m falconvar.workflow samples/x.mp4 --policy vad --sampler clip,yolo:overview
python -m falconvar.workflow samples/x.mp4 --no-audio --sampler uniform:text
python -m falconvar.workflow samples/x.mp4 --tier llm --sink file,supabase \
       --index qdrant,supabase

# one component at a time; per-stage tuning lives on these, not on workflow
python -m falconvar.media samples/x.mp4
python -m falconvar.audio <id> --transcriber whisper --diarizer pyannote
python -m falconvar.boundaries <id> --policy scene --evidence --stride 5 --threshold 27
python -m falconvar.boundaries <id> --calibrate        # sweep, no decode
python -m falconvar.boundaries <id> --retune 45        # rethreshold cached scores
python -m falconvar.boundaries <id> --policy scene --chunk-duration 30
python -m falconvar.video <id> --sampler "clip:[text,scene]"   # one pass, two questions
python -m falconvar.video <id> --sampler clip:text+scene       # same, no brackets
python -m falconvar.video <id> --sampler yolo --per-second 4 --min-interval 3
python -m falconvar.video <id> --sampler objects --vocabulary "crate,pallet"
python -m falconvar.video <id> --prune-store           # irreversible, opt-in
python -m falconvar.cut <id>
python -m falconvar.describe <id> --describer openai --limit 5   # costs money
python -m falconvar.describe <id> --describer ollama/gemma3:4b   # any provider, provider/model
python -m falconvar.aggregate <id> --tier llm --llm anthropic
python -m falconvar.rag.embed <id> --embedder local --index qdrant   # in-process, no key
python -m falconvar.video <id> --sampler uniform:safety   # a custom question
python -m falconvar.rag.embed <id> --index qdrant,supabase
python -m falconvar.rag.retrieve "..." <id> --sampler clip:text   # one pairing
python -m falconvar.rag.retrieve "..." <id> --question text       # across samplers
python -m falconvar.aggregate <id> --tier llm
python -m falconvar.aggregate <id> --tier llm --only entities   # who is who, across chunks
python -m eval.entities                         # grade linking against hand labels

python -m falconvar.shared.contracts.schemas --check     # CI: are the schemas stale
python -m recovery.recreate data/out/<id>/manifest.json --verify data/out/<id>/store
#      ^ from the checkout root: recovery/ is not an installed package, on purpose
python -m uvicorn api.main:app --port 8000     # the app at /, /docs for the schema
```

**Everything a run writes lives under `data/out/<video-id>/`.** Grouped by
video rather than by artifact type, so one video's whole output is one thing to
inspect, copy or delete, and a later component adds to it without a new
top-level directory.

---

## Architecture — the load-bearing decisions

**The grid is a component, not a side effect.** Everything that needs
boundaries reads `timeline.json`; nothing derives them as a byproduct. There is
no ordering rule anywhere in `workflow.py`, because the answer falls out of
what the policy depends on, and `boundaries.POLICIES` is that table as data:

    uniform    nothing.  arithmetic over a duration `media.json` already has
    scene      the picture.  a scene pass decodes and scores it
    vad        a transcript. audio must finish first
    speaker    a transcript. audio must finish first

`uniform` is the case worth noticing: it needs neither modality to have run.

**Each modality has a precursor and a consumer**, and the symmetry is the
point:

    audio  ──┐              ┌── cut
             ├──► boundaries ┤
    scenes ──┘              └── video

`speech.py` derives vad/speaker cuts and lives in `boundaries/`, not `audio/`,
so that package never learns chunking exists. It reads `transcript.raw.json` as
a **file**, never by importing `audio` — which is what keeps the import graph
acyclic while the run order flips between policies.

**Ingest never edits a boundary.** It asks `timeline.nearest(ts)` and nothing
else. No streaming chunker, no `observe()` at native rate, no open-ended
`bounds_of`, no final-`end_ts` correction, no tail merge — all of which exist
only when a pass produces the grid it is simultaneously consuming, so decisions
get made before the information to make them exists and have to be patched.

**Components exchange files, never objects.** Every one is
`run(video_id, ...) -> Produced`, addressed by video id and a backend rather
than by assembled paths. That is what makes each independently runnable,
retryable and testable, and what lets one API route serve all of them.

**`documents.py` imports nothing, and that is load-bearing.** If a document's
dataclass lived in the component that produces it, `cut` would import
`boundaries` to read a timeline — growing exactly the edges the file handoff
removes.

**The raw transcript is stored before it is cut.** `listen` writes words,
segments and turns with no chunk ids; `cut` applies a grid. Re-cutting is free
and transcription is not, so a grid change never re-runs Whisper. That
asymmetry — audio conforms cheaply, a sampler's decisions cannot be revisited —
is what lets either modality own the grid.

**Cascade follows cost.** Rebuildable in seconds (`chunk_samplers`,
`transcript_chunks`) gets a foreign key with `on delete cascade`; anything that
cost inference or a paid call (`descriptions`, `embeddings`) gets **none** and
carries a fingerprint instead. Re-ingesting costs seconds where describing
costs money, so a cascade from the grid into `descriptions` would mean retuning
a scene threshold silently destroying everything a VLM was paid to produce.

**The frames are written before the row that claims them.** There is no
transaction across two REST writes, so ordering is the only guard. `_manifest`
wrote `manifests` and then `chunk_samplers`; when the second was refused the
first had landed, leaving a manifest claiming a run with no sampled frames —
indistinguishable from a run whose samplers kept nothing, which happens. Written
the other way round, the same failure leaves rows nobody points at and no
manifest claiming them, so a reader is told the truth: not ingested here yet.

**Recompute and write are different questions.** The fingerprint governs
whether to recompute; every requested backend is written regardless. Conflating
them means a run that adds a backend has nothing to recompute, writes nothing,
and reports success — found twice, in `embed` across two indexes and in
`aggregate` across two sinks.

**`recovery/` imports nothing from the pipeline.** Hand someone those files, a
manifest and the video, and they rebuild the store byte for byte with only
`av`, `opencv-python`, `numpy`. If recovery imported the pipeline it could lean
on a default living in code rather than in the manifest, and the manifest's
claim to be authoritative would go untested. Nothing enforces this
automatically any more — the old AST check went with `imports.py`.

**Recreate is the end-to-end oracle.** Any change to encoding, addressing or
the manifest is verified by rebuilding a store and byte-comparing. Verified at
13/13, 60/60, 83/83, 206/206, 76/76. If a change makes recreate non-identical,
the change is wrong.

**`--verify` compares the frames the manifest names, not the output
directory.** An output directory accumulates across runs for exactly the reason
a frame store does — which the orphan rule already tolerates in the other
direction. `rebuilt/` holding 206 frames from an earlier manifest made a
correct 76-frame run report **FAIL, 130 absent from the store**, when every
named frame was present and byte-identical. A false FAIL is the worst answer
the oracle can give, because a real one means the change is wrong and is
supposed to stop everything.

**Paths are anchored to the checkout, found by marker.** `shared/paths.py`
searches upward for `pyproject.toml` rather than counting parents — a parent
count is a fact about a file's depth in the tree, which is exactly what a
reorganisation changes. It broke twice this way: `WEIGHTS_DIR` at
`parents[3]/"weights"` resolved to a directory that never existed, so weights
downloaded wherever ultralytics decided and nothing reported it; and moving
`paths.py` one level down silently redirected every artifact.

`FALCONVAR_DATA` moves the data root. It is a **process** variable, not a
`.env` key: `.env` is read at the top of an entry point, later than these
constants resolve, and a path that moved depending on how early it was read
would be worse than one that cannot go in `.env` at all.

**A leading underscore under `data/out/` marks a directory that is not a
video.** The embedded Qdrant store lives at `_qdrant`, beside the videos rather
than inside one; without the rule it was listed as a video with no artifacts,
by `paths.videos()` and so by `GET /videos`.

**A requirement is required.** Nothing falls back when a package in
`requirements.txt` is missing or older, or when `install.sql` was never run:
no hand-parsed `.env` without `python-dotenv`, no plain Supabase client for an
old SDK, no Python ranking when the search RPC is absent, no `transformers`
stand-in for `sentence-transformers`. Each of those was a second code path that
answered *differently* while looking the same -- the dense-only Supabase
ranking had never ranked anything, and nobody could tell. Heavy imports stay
function-local so `import falconvar` is light; a missing one fails with the
interpreter's own `ModuleNotFoundError`.

---

## Sampling and questions

**Which frames, and what to ask, are independent.** Every sampler takes a
`prompt` — it lives on the base class — so any strategy pairs with any question
as `name:prompt`. `uniform:text` reads the screen on a stride; `yolo:overview`
keeps frames where the people changed and asks for prose instead of the
structured people call. Unpaired, the question is the sampler's own name.

The point is cost. `text` fires *when the writing changes*, and finding that
out costs EasyOCR on every decimated frame — 98.1% of that sampler's total.
"Read the screen every so often" wants none of that: measured on Chernobyl,
`uniform:text` ran **no model at ingest** and the VLM still transcribed
`RadioFreeEurope RadioLiberty`, `BYELORUSSIAN S.S.R.`, `REACTOR 1`, `1977`
correctly. Two different questions, not two settings of one.

**A sampler runs once and answers a list of questions.** `clip:[text,scene]`
is one pass over the video answering two questions about the frames it kept;
`clip:text+scene` is the same thing without brackets, for shells that glob
them. Selecting frames is the expensive half -- CLIP or YOLO on every decimated
frame, EasyOCR at 98% of the text sampler's cost -- and a second question about
frames already chosen costs one more describe call.

**Specs naming the same sampler merge into one run.** `clip:text,clip:scene`
means exactly `clip:[text,scene]`, so brackets are the explicit spelling of
something that happens anyway rather than the only way to avoid paying twice.
Measured before this existed: `uniform:text` and `uniform:reactor` produced
**identical frame lists on all 14 chunks**, each having walked the video
separately. Merging is by name because one CLI has one `--threshold` and one
`--vocabulary`, so every spec shares a configuration.

**The manifest is keyed by run; everything downstream is keyed by answer.**
`chunks[].samplers` holds `clip` once, with its frames and `prompts:
[text, scene]` in the config. `descriptions` and `embeddings` hold `clip:text`
and `clip:scene` separately, because they are different units to a search even
though one pass produced both. `answer_id` is the bare name when the question
is the strategy's own, so an unpaired sampler keeps the id it always had and
nothing already indexed becomes unreachable.

`chunk_samplers.questions` is a `text[]` for the same reason: one row per run,
carrying the list.

**Every (sampler, question) pairing is independent.** `schema_for` is a
function of the question alone: two questions that share a field both answer
it, and both answers are kept under their own sampler id. Overlap is a choice
the user made at the `--sampler` line, and the answers are genuinely different
— measured on one chunk, `clip` gave `visible_text` as
`['RadioFreeEurope', 'RadioLiberty']`, `uniform:text` gave six objects each
with a `context` and an explicit `[unreadable]`, and a custom `reactor`
question gave five plain strings including `RADIATION` and `115,000`. All three
are stored, embedded and separately searchable.

**Sibling narrowing was tried and removed.** A call's schema used to depend on
which other questions were asked about the same chunk: the fallback shape gave
up any key a specialist owned. It produced three silent faults — sampler ids
passed where questions were meant; `yolo:overview` taking a key from a call
whose schema answered none; and two fallback questions overlapping on all seven
keys with no rule for which won, so a chunk rollup kept whichever sorted first.

Each fault left a well-formed document. That is the signature of a seam in the
wrong place rather than three unrelated mistakes, and the payoff did not
justify it: narrowing only ever fired for **one of four** pairings (fallback +
specialist, for three specific keys) and saved 2.3% of output on the case
measured, because the summary is half the output and was never narrowed.
Extending it instead would have meant answering "when two questions both want
`people`, who wins" — and there is no non-arbitrary answer, because the user
asked both.

Gone with it: `OWNER`, `owned_by`, `questions_on`, the `siblings` argument and
the `chunk_questions` context key, across five files.

**There is no chunk-level rollup.** `descriptions.chunks[].structured` used to
flatten every sampler's answer into one record, which is what forced a winner
for a shared key. Nothing read it — units, both index writers, `rows.py` and
every aggregator work from the per-sampler blocks — so it was deleted rather
than given a tie-break rule.

**An unknown question is rejected, not fallen through.** `question_for` falls
back to the scene question by design, which makes `yolo:overvew` a run that
completes, costs money and answers something nobody asked. `prompts.QUESTIONS`
is the vocabulary, and `workflow.validate` checks **both halves** of every
`name:question` pair against it and the sampler registry.

It has to be there rather than only in `describe`, which is where it used to
be: that check reads the finished manifest, so `yolo:overvew` was a 202 that
ran media, audio, boundaries and a whole video decode before failing on a
typo -- the late failure `validate` exists to prevent. `validate` imports both
vocabularies function-locally, as a composition root; `video` still never
imports `describe`.

**Ingest does not depend on describe.** A sampler records the question as an
opaque string and `base.py` never reads it. Only the drivers import `prompts`,
function-locally: they are composition roots, and validating a typo is the one
thing they want the vocabulary for.

**A question is an instruction and a shape, and the shape carries the schema.**
`prompts.json` holds both; `prompts.py` is logic over it. The built-ins are
expressed in the same terms a custom question uses -- `yolo` is not a special
case in the code, it is the `people` shape -- so there is one mechanism rather
than a shipped set and an extension point beside it. A shape is a set of
fields; `prose` is the degenerate one with none, which is what `overview`
answers in. The shape marked `fallback` is what an unrecognised question
resolves to, and that is its only privilege.

**A custom question names a shape or brings its own, and a custom shape is
built rather than accepted.** `fields` is a builder -- `text` or `list`, plus
`of` for a list of objects and `one_of` for a fixed vocabulary -- and the JSON
Schema is generated from it. Raw JSON Schema over HTTP is refused because the
call goes out with `strict: true`, whose subset is narrow: a schema the API
rejects would fail *after* the frames are read, with the request about to be
paid for, which is the same late failure `check()` exists to prevent for a
`{typo}` placeholder. The builder spans exactly what the shipped shapes use --
verified by rebuilding all five of them from their own field lists, identical.

Shipped shapes stay shipped. A custom shape is stored under its question's
name, dies with it, may not take a built-in shape's name (`people` and `prose`
are shape names that are *not* question names, so the question-level guard
misses them), and can never claim `fallback`. So `yolo` still means the same
thing in every deployment; what is deployment-local is a custom question, which
it already was. Caps -- 12 fields, 8 nested keys, 24 enum values -- are refused
at write time, because structured answers already run ~3x longer than prose and
the `people` schema truncated into unparseable JSON at 700 output tokens.

The earlier shipped-shapes-only rule was verified against the previous
hard-coded module over **448 (question, siblings) combinations** at the time it
was extracted -- every schema, every instruction, the owner map and `merge`
identical.

**Built-ins ship in the package; custom questions live in `data/prompts.json`.**
A custom entry may not shadow a built-in, refused at write time as a 409 and
dropped at load time as well, because the file is hand-editable and a shadowed
built-in is the one failure that would change a shipped question's meaning
silently. The alternative is a deployment whose `yolo` means something other
than every other deployment's, with nothing in the repo saying so.

**A single shared schema was tried and is wrong.** Every field being present
means the model may fill any of them, and it does — asked about people it
returned a paragraph about the room, paid for twice, leaving two `setting`
values with no rule for which wins. A prompt saying "focus on X" is a request;
a schema with no Y field is a guarantee.

**A specialist returns objects, never parallel lists.** One entry per person
carrying `appearance`, `clothing`, `role`, `action` — not a list of people
beside a list of actions, which does not say who did what and cannot be made to
afterwards.

**`overview` is a question with no fields at all**, and only a question. Summary
only, 4–5 sentences, overriding the ">= 150 words" instruction every other
prompt carries. Its shape has no fields at all, so it costs only a summary.
Measured: 84 and 86 words, 4 sentences, zero structured keys.

**Structured answers are ~3x longer than prose.** At `max_output_tokens=700`
the people schema truncated mid-string and came back as unparseable JSON. The
default is 2000, and truncation is detected from the response's own
`status`/`incomplete_details` rather than inferred from a JSON error further
down.

**`uniform` counts decimated frames, not seconds.** `every_n` is a stride over
the stream the sampler was actually offered, read off `chunk_local_index` — the
one thing about position a sampler is handed. The cadence in seconds is a
consequence of decimation: `every_n=3` is one frame every 3 s at
`per_second=1` and one every 0.75 s at 4. A cadence in seconds regardless is
still expressible through `min_interval_s`, enforced in the base class before
the strategy runs. Default 1: every decimated frame.

**Resume is keyed on the manifest, the describer *and* the prompts.** Without
the model check, describing with the stub and then switching to a real one
skips every pair and reports success having done nothing — the most expensive
kind of silent no-op, since the output looks complete. Editing a prompt changes
the output but not the model id, so the `model` block carries prompt hashes
too.

**Those hashes are per question, not one over the vocabulary.** A single hash
meant that *adding* a question — which cannot change what any existing answer
should say — invalidated every description of every video, and the next run
silently paid to rebuild them all. `model.prompts` is `{question: hash}`, and a
stored pair is current when its own question still hashes the same. Measured on
Chernobyl with `clip,uniform:safety`: editing only `safety` gave **14
described, 14 skipped**, and adding an unrelated third question gave **0
described, 28 skipped in 0.0 s**. Under the single hash both would have been 28.

A stored `prompts` that is a bare string is the pre-map format; it cannot be
reduced to a per-question map, so every pair is re-described once rather than
kept on a provenance nothing can check.

**Describing reads the frame store and nothing else.** No seek-the-video
fallback: the store exists so this stage has its frames in hand, and a fallback
would quietly do its job while leaving it broken — silently, since the output
is identical and only ~40x slower. A short frame list is never returned either:
a description covering 8 of the 9 frames it claims is indistinguishable from a
correct one once written down.

---

## Cost — where the time actually goes

**Convert lazily. This is the largest single factor.** Decoding a frame costs
**0.40 ms**; converting it to a full-resolution BGR array costs **6.2 ms**.
Ingest needs pixels only for frames that survive decimation — 4% of them at 1/s
from 25 fps — and the decimator answers from `media_ts` alone, so its verdict
is asked *before* the conversion.

| 3 h of 720p25 | |
|---|---|
| convert every frame | 3.55 ms/frame → **16.0 min** |
| convert only decimated | 0.50 ms/frame → **2.2 min** |

**7.2x for identical output.** This is only possible because scene detection
left the ingest pass; while it was there, `observe()` needed pixels at native
rate and the reader was the only thing holding them.

**A separate scene pass costs ~3 minutes per three hours, not more.** Measured
optimally on both sides: fused 11.97 min against separate 14.86, the difference
being one bare decode at 1.82 min. So the split is architectural, not a
performance trade — and the earlier "10–15 min" estimate was measuring a
lazy-conversion bug, not the architecture.

**Stride the scene detector, and recalibrate when you do.** It scores the
difference between *consecutive frames it was given*, so at stride 5 it
compares moments 200 ms apart rather than 40 ms.

| stride | ms/frame | 3 h | cuts found | real cuts kept |
|---|---|---|---|---|
| 1 | 12.89 | 58.0 min | 2 | — |
| 5 | 2.97 | **13.4 min** | 6 | 2/2 |
| 25 | 0.94 | 4.2 min | 11 | 2/2 |

Every real cut survives every stride; what rises is false positives, because
threshold 27 is calibrated for adjacent frames. `min_s` absorbs some — measured,
a strided pass double-fired at 27.0 s and 27.2 s on one scene change and the
guard merged both away.

**Cache the scores, not the cuts.** `cuts.json` records the per-frame series,
so re-thresholding is arithmetic over a cached array. Verified **identical**
cut lists to a full re-run at thresholds 45 and 20, in 0.12 ms against 5.9 s —
40,000x. Fused, every threshold change costs a full re-ingest with CLIP and
YOLO loaded.

**A frame kept by two samplers is one file, and must be counted once.**
`uniform:text` at stride 1 offers every decimated frame and `clip` picks from
that same set, so the second pick names a file the first already wrote. The
store is addressed by read index, so the write was harmless -- the accounting
was not: 266 frames and 52.75 MB reported against 206 files and 42.18 MB on
disk, a 25% overstatement of the figure someone would use to plan capacity, on
a two-sampler run. `frames_sampled` is picks and `stored_frames` is files, and
they are different numbers. Deduplicated per run, not per directory, so a file
an earlier run left behind is still rewritten.

**A store accumulates across runs and is never pruned automatically.**
Ingesting with `uniform` then with `clip` left 206 files where the manifest
named 83. `--prune-store` is opt-in because deleting frames is the one
irreversible thing ingest can do.

---

## Retrieval — measured

These figures predate the current tree and were taken on the previous
pipeline's corpus. The **orderings** are the finding; treat the decimals as
provenance rather than current fact, since nothing here has been re-measured
against a corpus larger than one video.

**Embed the summary *and* the structured fields.** 22 disjoint query pairs,
dense MRR (random 0.457):

| embedded from | literal | paraphrase |
|---|---|---|
| summary | 0.528 | 0.522 |
| structured | 0.636 | 0.586 |
| both | **0.705** | **0.608** |

Every summary repeats the same setting; the fields do not, so they carry far
more distinctive content per token while a bound object per entity keeps
who-did-what intact.

**Each half of the hybrid is strong exactly where the other fails.** 22 query
pairs with zero shared content words:

| | literal top-1 | literal MRR | paraphrase top-1 | paraphrase MRR |
|---|---|---|---|---|
| dense | 23% | 0.528 | 23% | 0.522 |
| BM25 | **59%** | **0.752** | 18% | 0.468 |

Neither half knows which kind of query it was handed, and a search box gives no
signal — which is the whole argument for fusing rather than choosing. Withheld
lexically on the shipped path: literal **+0.140, CI [+0.020, +0.260]**;
paraphrase **+0.000, CI [0, 0]** — not degraded, *identical*, because BM25
matched nothing at all.

**RRF twice, never a weighted score.** Cosine distance and `ts_rank_cd` have no
common scale, and any weight between them would be invented; RRF reads only the
orderings, so it needs no calibration.

**A chunk scores as its best unit plus a discounted second, never a sum.**
`score = 1/(k+best) + 0.5/(k+second)` at k=10. Summing over every unit a chunk
contributed applies RRF to the wrong problem: it fuses several rankings of the
*same* items, where the term count is constant, while a chunk contributes one
term per sampler that described it. At k=60 over ~20 candidates `1/(k+rank)`
spans only 1.31x, so count overwhelmed rank — a chunk whose best description
ranked 13th beat one whose best ranked 1st, and the shipped ranking got the
*video* right 57.7% of the time.

| aggregation | video ok | literal MRR | paraphrase MRR |
|---|---|---|---|
| sum, k=60 | 0.577 | 0.421 | 0.341 |
| max, k=60 | 1.000 | 0.668 | 0.442 |
| **max + 0.5·second, k=10** | **1.000** | **0.682** | 0.446 |

Invisible on a single video with a uniform sampler set, where every chunk
contributes the same number of terms and the bias cancels. It appears the
moment an index holds more than one video.

**One vector space per embedder, never per sampler.** A space is defined by the
model, not by which prompt produced the text. The sampler is payload and
querying one is a *filter* — which gives up the agreement signal: a chunk
contributes fewer terms, so scores fall.

**A unit carries both halves of its id as fields, not just the id.** `sampler`
and `question` are separate columns in `embeddings` and separate keys in the
Qdrant payload, so three filters are each one equality: this **pairing**
(`sampler_id = clip:text`), this **question** wherever asked
(`question = text`), and this sampler's whole output (`sampler = clip`).

Filtering by question is the query a person actually makes — "the text on
screen", not "what the CLIP sampler said" — and it is **not** a suffix match on
the id, because a bare id like `clip` means the question *is* the strategy
name. Measured with `clip:[text,scene],uniform:text`: `question=text` returns a
chunk carrying `clip:text` **and** `uniform:text` at 0.1222 where
`sampler=clip:text` returns one unit at 0.0909, and unfiltered returns all four
units at 0.1294. The three answers are different, which is the point.

Neither column cost a re-embedding: `text_hash` is over the content, so
existing rows were backfilled from `sampler_id` with an `update` (since removed
from `install.sql`: 0 of 49 rows still needed it). Qdrant is the exception --
payload is written only on upsert, so points predating the fields need a forced
re-index rather than a backfill.

**What gets embedded must not depend on which copy it was read from.** `jsonb`
preserves array order but not object key order, so a description read back from
Postgres hands its keys back in a different order from the file. Joining
`item.values()` in iteration order made the same person into different text, a
different `text_hash` and a different vector depending on its source — cosine
0.995 between them, close enough that no ranking ever looked wrong. Keys are
sorted at **every** level; determinism had been handled one level deep and not
two.

**A structured field is only filterable if its values are a vocabulary.** `role`
is free text, so one video produced `cashier`, `customer`, `cashier or customer
near checkout`, `child customer` — and a filter for "cashier" matches all of
them. The mechanism works; what is missing is an `enum`.

**Generating paraphrase test queries needs verification.** Asked to "share no
content words", the model kept a median 50% of them, and BM25 appeared to win
on paraphrases as a result.

**Cross-modal agreement is the point, and it shows.** Chernobyl, one `vad`
grid: `"the moment the reactor exploded"` returns the chunk where the visual
pass describes "a severe explosion and fire in one section of the reactor
building" and the audio pass "At 1.23 a.m., reactor 4 exploded" — two
independent accounts of the same 16 seconds, neither pass having seen the
other's output.

### The two lexical halves are not equivalent

Measured on 41 identical units with identical OpenAI vectors. **Dense halves
identical on 6/6 queries**; both silent on a query sharing no content word with
the corpus. The lexical halves differ, and `supabase` is the better one:

- **No stemming in `indexes.tokenize`.** Postgres stems via
  `to_tsvector('english')`, so "lived" matches "live". On one query Postgres
  matched 21 of 41 units and Qdrant 13.
- **No length normalisation or TF saturation in the sparse vector.**
  `Modifier.IDF` supplies the IDF half of BM25 and expects the client to send
  weighted values; the client sends raw counts. `ts_rank_cd` normalises by
  cover density, so the shortest unit in the corpus tops Qdrant's half where
  Postgres ranks it third.

Neither is fixed: stemming is a dependency, proper BM25 weighting needs an
average document length the client does not have, and 41 units of one video
cannot show either change helps.

**`websearch_to_tsquery` ANDs its terms**, so one word absent from the corpus
silences the whole lexical half — `"reactor exploded"` ranked 2 rows and
`"the moment the reactor exploded"` ranked **0**, because "moment" appears
nowhere. The RPC replaces `&` with `|` in the rendered tsquery, keeping the
parser's stemming and stopword removal and only loosening the conjunction;
`ts_rank_cd` then does the work AND was doing badly. After the fix those
queries rank 7 rows each, and a query sharing no content word still ranks 0.

**ANY-term over-corrects, so it has a floor.** It fires on a single stem
collision: measured, a query sharing no content word with the corpus still
matched, promoted an unrelated chunk to second and pushed the right answer to
third, where Qdrant's half stayed silent and ranked better for it. With two or
more query lexemes a row must share at least two; a one-word query needs one.
"the moment the reactor exploded" shares `reactor` and `explod`, so it still
ranks.

**RRF ties are broken by the vector rank.** Ties are exact and common -- dense
1 / text 2 and dense 2 / text 1 are both 1/61 + 1/62 -- and `order by score`
alone left the winner to the planner. The dense half has an opinion on every
query where the lexical one may not, and the two backends now agree rather
than flipping a coin opposite ways.

**Qdrant does sparse and hybrid.** Sparse vectors since 1.7, native
`Fusion.RRF` since 1.10. An earlier note calling it dense-only was a claim
about the implementation dressed as one about the database. Prefetch depth must
match the RPC's `greatest(limit*4, 40)`: prefetching only `limit` truncates
each ranking before fusion, measured as the lexical half firing on 12/20 rows
where Postgres, ranking 80, fired on 20/20.

---

## Audio

**Audio is scanned whole; it cannot be chunked first.** Whisper carries context
across an utterance and detects language from the opening seconds. Diarization
is worse: speaker labels come from clustering embeddings over the *entire*
recording, so `SPEAKER_00` in one window bears no relation to `SPEAKER_00` in
the next — chunk first and the speakers are not misaligned, they are
unnameable.

**Transcription and diarization stay separate passes, joined by `align`.**
Whisper does not know who spoke and pyannote does not know what was said. A
word is attributed by its **midpoint**, because the two models estimate edges
independently and word spans routinely straddle a turn boundary; a segment
takes the speaker who spoke most of it by duration. With no diarizer, every
segment keeps `speaker=None` — truthful, where labelling everything
`SPEAKER_00` is not.

**Silence is answered without a model.** CCTV with a live but empty microphone
sits at RMS 0.000221, peak 0.0291, against 0.1796 for narration — three orders
of magnitude, so the 1e-3 threshold sits in a wide gap rather than on a cliff.

**Chunks with no speech are kept, with empty text.** The grid is shared, so
`chunk_id` must mean the same thing in the manifest and the transcript;
dropping the quiet ones renumbers everything after them.

**Only `speakers` goes into `structured`, never `turns`.** `turns[].text` *is*
the transcript, so rendering it appended the whole chunk a second time
interleaved with timestamps read as numbers — 337 characters where 151 was
right.

**Measured, RTX 4060:** decode 205 s of AAC in 0.30 s (~700x realtime), Whisper
`small` float16 37.6x, pyannote 3.1 30.1x. Chernobyl: 34 segments, 428 word
timestamps, 1 speaker over 18 turns covering 89% of duration.

---

## The grid

**One chunk grid per run, and either modality may decide it.** `chunk_id` is
simultaneously the unit a describer summarises, a transcript is cut into, and
retrieval returns — so if the two halves disagreed about a boundary, one query
would return two different clips with no honest way to say which is the answer.

**Content-derived boundaries need both guards or they are unusable.** Voice
activity cuts on every pause — every second or two on conversational audio,
which would shred the video into chunks too short to describe. A monologue
gives the opposite failure: zero cuts, one chunk covering the file. Measured on
Chernobyl, `speaker` on single-narrator audio found no speaker changes and
`enforce` divided the file into 7 even chunks of 29.3 s, which is the honest
outcome rather than an invented one.

**A floor above the ceiling is refused, not resolved.** `enforce` merges up to
`min_s` and *then* splits at `max_s`, so the split runs last and wins — and
`--max-chunk` defaults to `--chunk-duration`, which defaults to 20. So
`--min-chunk 30` on its own asked for "at least 30, at most 20" and produced a
grid whose shortest span was **18.07 s**, reported as success, because a scene
grid of 18-second chunks is an entirely ordinary thing to see. There is no
reading of that request to honour. With a coherent pair the same video gives 4
chunks of 39.8-55.6 s.

**`max_s` splits evenly, not into fixed bites.** Taking 30 s bites off a 62.5 s
span leaves a 2.5 s remainder, so the guard against short chunks would create
one. It becomes three of 20.8 s.

**A short final chunk is merged into the one before it, under every policy.** A
grid divides the media wherever it happens to end, so the tail is uniformly
distributed over the chunk length: 97.99 s at 20 s leaves a usable 17.99 s, but
100.4 s leaves 0.40 s. That stub costs a describer call *per sampler*, keeps a
frame because every chunk keeps one, and is a moment retrieval can return that
nobody can play. `MIN_TAIL_FRACTION = 0.25` is a fraction rather than a number
of seconds so it holds at any chunk length.

**The grid spans the longer stream.** A file is as long as its longest one, and
the streams differ — 205.264 s of audio against 205.280 s of video on Chernobyl
— so a grid built from the audio alone leaves the last video frames outside
every chunk. `Timeline.nearest` clamps anything past the end to the last chunk
rather than dropping the frame.

**`Timeline.fingerprint` is recorded by everything cut on it.** If two
documents disagree, they were cut on different grids and `chunk_id` means two
different things — a comparison a reader can make, rather than drift nothing
reports.

**`media_ts` is the only clock a decision may use.** Never wall time, never
frame counts. Decimation buckets on media time, never "every Nth frame":
identical on a clean file, self-correcting on a lossy one.

**Samplers reset at every chunk boundary** and every chunk keeps at least one
frame. Rate limits are enforced in the base class *before* the strategy runs,
so a rate-limited frame costs no inference.

**Pixels are borrowed.** `frame.release()` runs every iteration; anything that
outlives the loop must copy.

---

## Aggregates

**Aggregates answer what retrieval cannot.** Embeddings cannot count, so "the
busiest moment", "who dominated", "how much of this is speech" are exact
questions similarity answers approximately. Measured on Chernobyl: 89.4% speech
ratio, 125.1 words per minute, `monologue: true` with 0 handovers, and chapters
that tile the whole video with the explosion at 94.4 s.

**A tier is a cost ceiling, and asking for a dear one still runs the cheap
ones.** Cheapest first, so a run that dies partway has produced the free results
rather than none.

**`depends_on` drops rather than fails.** `speakers` on silent CCTV is not an
error, it is a question that does not apply, and it is reported as skipped
*with the reason* — because "speakers did not run" is only useful beside why.

**Every account of a chunk is rendered, not the first one found.** Taking only
the best-ranked source throws the other modality away. Measured the hard way: a
run picked stub `clip` text over 428 words of real narration, and the model
correctly reported that it had been given nothing to summarise.

**Ask for a word count, not "several sentences".** Once structured fields
arrived the model sized the summary as one field among many: 105 median words
against 363 in the prose-only era. Saying "at least 150 words" took it back to
246 median. The summary is the only text that gets embedded, so its length is a
retrieval parameter.

**Every summary layer is recorded; none of them is embedded.** A leaf summary
covers a real span and is the only description at that granularity, between one
chunk and the whole file, so it is kept. Indexing them would return the same
moment two or three times over under different wordings — the count-bias
failure the moment aggregation guards against, one level up.

The final summary goes somewhere else: `units.from_summary` turns it into one
unit per video, which `embed` writes to `video_embeddings` rather than
`embeddings`. `embeddings` answers *which twenty seconds*, a summary answers
*which video*, and a video is not a moment you can play -- so the two never
share a ranking, and `/search` reaches the second only as `level=video`.
Postgres only, and best-effort: a missing `summary.json` writes nothing.

**Spans are resolved through the timeline, never trusted from the model.** It
is asked for chunk ids, which it can copy; times it would invent.

**`inputs_fingerprint` is a hash of the chunk text actually read.** A summary of
descriptions since rewritten reads perfectly, which is precisely why staleness
cannot be left to a reader to notice.

**Entities: who is who is decided by rules; the model only narrates.**
`aggregate/linking.py` embeds each entry's identity fields and merges under
constraints; `entities` then asks the model for one narrative per linked entity
(concurrently, capped at 12). Tried the other way first: gpt-5.4-mini, given
test1's 39 actor entries, linked 31, broke the same-answer rule twice after
being told it, and merged an older woman with an older man.

**Identity is declared, never inferred.** A shape names the keys that identify
an entry -- `people`: `appearance`, `clothing` -- in `prompts.json`'s top-level
`identity` map, or as `identity` on a custom field spec. Beside the shape, not
inside it: `version_of` hashes the shape, so a key there would re-describe
every answer given in it (verified: 18/18 prompt hashes unchanged). A shape
declaring none is never linked. The driver fills `Context.identity`, so the
aggregator never imports `describe`, and `entities` folds the declarations
into its fingerprint through `inputs_of` -- the only aggregator with one, so
every other stored fingerprint is untouched.

**Cannot-link is per answer, and the threshold is read off it.** Two entries in
one answer are different by the question's own wording ("one entry per
distinct person"), so their similarities are a calibration set for this video
and this embedder: link only above the most similar provably-different pair,
and only mutual best matches. Per answer rather than v0's per chunk, because
two questions about one chunk may describe the same person. v0's fixed 0.88 was
a fact about one embedder:

| F1, test1 / test2 | text-embedding-3-small | bge-small |
|---|---|---|
| v0: 0.88 fixed, cannot-link per chunk | 0.64 / 0.29 | 0.77 / 0.80 |
| **`max` + mutual (default)** | **0.94 / 0.91** | 0.88 / 0.91 |
| `q95` + mutual | 0.90 / 0.91 | 0.97 / 0.91 |

The default holds precision 1.00 on both videos under both embedders. `q95` is
better on bge and makes wrong merges on OpenAI, and the default has to hold
under whatever embedder a deployment runs. v0's genericness filter, tried as an
outlier test on mean similarity, changed no result anywhere and was dropped.

**A score that skips doubtful labels hides exactly the wrong merges.** The
first real run scored precision 1.00 while merging a dark puffy coat and a
cream coat into the woman in the gray top -- every one of those mentions was
labelled unsure, so no pair of them was scored. `eval/entities.py` now counts
links touching unsure mentions as `unchecked`, and `different` labels rule a
mention out of a group. Still in the stored output, unscored: the dark-coat and
cream-coat women linked to each other, and two women linked on "entering".
`activity`'s `actor` field carries position and behaviour; the `people` shape's
`clothing` would not.

**The labels are tiny and were written by the builder**: 20 scored mentions on
test1 and 9 on test2, read from the descriptions after seeing one run. test2 is
the check test1 was not tuned on. A direction, not a result.

---

## Models and providers

**A stage names a provider and a model; `shared/models/providers.py` knows the rest.**
Three roles -- `describe`, `llm`, `embed` -- and four protocols, because the
wire format is the only thing that really differs between vendors:

    openai     Responses + /embeddings
    chat       Chat Completions + /embeddings: Ollama, LM Studio, llama.cpp,
               vLLM, Gemini, Mistral, Groq, OpenRouter, Together, DeepSeek,
               xAI, Voyage
    anthropic  Messages, with the schema as a forced tool's input
    local      a Hugging Face model in this process. Vectors only

One `ModelDescriber` and one `llm.Model` serve every provider, so adding one is
a row in `_BUILTIN` or an entry in `data/providers.json`, never a class. A
describer and an embedder per vendor would be the per-stage copies of
connect-and-complain that `db.py` and `llm.py` each exist to prevent.

**OpenAI stayed on Responses rather than joining Chat Completions.** A
describer's `config()` is half of describe's resume key, and every description
in the corpus was paid for on the Responses path; folding OpenAI into the
generic path would have been one branch fewer and a reason to re-describe
everything. Verified: `ModelDescriber().config()` is dict-equal to the old
describer's, and **40/40** stored pairs across four videos still resolve as
current. The generic path was then run against OpenAI itself, as a custom
`chat` provider: the same pair in the same shape, and embeddings identical to
the native path's (cosine 1.000000).

**A default is resolved when a call is made, never captured.** `Options`'
model fields are `None`; `providers.choose` reads the call, then
`FALCONVAR_DESCRIBER` / `_LLM` / `_EMBEDDER`, then `openai`. A constant would be
read at import, before `.env` -- the trap that keeps `FALCONVAR_DATA` a process
variable. `/capabilities.defaults` calls `providers.defaults()` for the same
reason: a form defaulting to the dataclass's `None` shows nothing where the
answer is `openai`.

**A model choice is one string, and only the stages that call a model take
it.** `describe`, `embed`/`retrieve` and `aggregate` each take one
`provider/model` argument; media, audio, boundaries, video and cut never see
one. There was a separate `model` field beside every provider field -- six on
`Options` and the upload form, a `--model` on four CLIs, two env variables per
role -- and it bought nothing `ollama/gemma3:4b` does not already say, while
needing a rule for which of the pair wins (an env model applying only to the
env's provider). Collapsed: three fields, three variables, no precedence rule.

**`provider/model` splits on the first slash, and only after a known
provider.** Model ids carry slashes (`BAAI/bge-small-en-v1.5`), so a provider
name may not. An unknown head leaves the spec whole, so the error names what
was typed rather than half of it.

**A missing key is a 422, not a failed job.** `workflow.validate` checks every
role the run will use -- describe only when reading the picture, llm only at
`--tier llm` -- because `describe` finding no `ANTHROPIC_API_KEY` happens after
the whole video has been decoded. It does not ping local servers: validation
stays synchronous and offline.

**Keys never enter `providers.json`.** A field with `key` in its name is
refused; the file names variables in `key_vars`. A bad entry is dropped and
listed under `/capabilities.models.problems` rather than raised, because one
typo in a hand-edited file taking down `/capabilities` takes every generated
form with it.

**Switching `--llm` rebuilds the llm aggregates.** `inputs_fingerprint` said
nothing about who wrote an answer, so a switch reused the stored summary and
reported success -- describe's silent no-op, one stage later. Aggregates now
record `stats.model` and reuse needs it to match. A stored llm aggregate with
no `model` reads as `openai:gpt-5.4-mini`, which is a fact rather than a guess:
before this, `aggregate.run` had no way to name another. That reading is also
what stopped the change rebuilding every summary -- verified, `--tier llm` on
defaults computed **0**.

**An embedder key carries its width, known before the index opens.** OpenAI's
are tabled; any other remote model is probed with one short string, once per
process, and every later batch is checked against it -- a server swapping the
model behind a name would otherwise write a new width into the old space.

**Query and document are embedded differently where the model says so.** e5,
nomic, bge and mxbai were trained with prefixes, and a search embedded as a
passage loses recall with no error anywhere. `embedders.PREFIXES` looks them up
by model id and `retrieve` goes through `query_vector`. A hand-set prefix adds
`:p<hash>` to the key, since its vectors are not comparable to the defaults';
nothing is added otherwise, so every key already written is unchanged.

**`local` is `sentence-transformers`, and nothing else.** It applies the
model's whole module list -- the pooling it declares (bge is CLS, not mean) and
any Dense layer after it. A hand-rolled `transformers` path for when it was not
installed was removed: it was a second function that had to agree with the
first under one key, and a model with a Dense layer could not agree at all.
Verified before removing it: on bge-small the two gave **identical** vectors
(max abs diff 0.0), so the index it built stands. Weights land in
`weights/embedders/`; a model is loaded once per process, because `/search`
runs in the server and a reload is seconds.

**Measured: a 33M-parameter local embedder is level with OpenAI here.**
`BAAI/bge-small-en-v1.5` (384-d, CUDA) against `text-embedding-3-small`, same
Qdrant hybrid, `eval/harness.py`:

| embedder | MRR | top-1 | recall@5 | median query |
|---|---|---|---|---|
| openai | 0.7315 | 0.611 | 0.833 | 0.619 s |
| local bge-small | 0.7241 | 0.611 | **0.917** | **0.051 s** |

By band: low overlap 0.222 vs 0.333 (n=3), mixed 0.694 vs 0.806 (n=6), high
overlap 0.911 vs 0.815 (n=9). Eighteen cases is a direction, and the direction
is "not obviously worse, and free". The 12x latency is the network round trip.
Indexing all 49 units took 2.0 s after a 34.5 s first load, download included.

**Supabase's vector columns were `vector(1536)`** -- OpenAI's width written into
the schema, refusing any other embedder at the first upsert. `install.sql`
alters both to unconstrained `vector` and drops the HNSW index, which needs a
fixed width. Every query filters on `embedder`, whose key carries the width,
before a distance is taken, so two widths never meet. On a corpus this size the
exact scan is milliseconds; one space with millions of rows would want a
partial expression index back. `SupabaseIndex` turns "expected 1536
dimensions" into "re-run install.sql".

**Verified as far as this machine reaches.** Real: OpenAI on both protocols
(including gpt-5.4-mini refusing `max_tokens` on Chat Completions and the retry
answering), and the local embedder end to end. A mock server that records
requests covered every other shape: images as data URIs, json_schema /
json_object / prompt modes, fenced JSON, truncation, an unreachable local
server, and Anthropic's base64 image blocks, forced tool, 401 and `max_tokens`
stop. **Not run against** the real Anthropic, Gemini, Mistral, Groq,
OpenRouter, Together, DeepSeek, xAI or Voyage APIs, nor a real Ollama or LM
Studio. Their default model ids are reasonable picks, not measured ones, and a
server's schema support is exactly what `structured` exists to downgrade.

**Model calls are async, and the provider says how many at once.**
`llm.Model.generate` is a coroutine behind every provider, so concurrency was
added once. `describe` plans every call in manifest order and reserves its slot
in the document, then gathers one task per (chunk, sampler run) under
`describer.concurrency`; a run's frames are read inside that gate, so memory is
bounded by the cap rather than the video, and still read once per run. The
summary's folds within a layer are gathered too. `Provider.concurrency` is 8
for cloud APIs and 1 for Ollama, LM Studio and llama.cpp, which answer one at a
time; `providers.json` can set it. Each stage runs its own `asyncio.run`, so a
client is opened per call -- an async client belongs to the loop that opened it
and fails when a second loop reuses it. The Anthropic path retries 429 and 5xx
twice, as the OpenAI SDK already does inside a call. Embeddings stay
synchronous: one request carries 64 texts, which is a whole test video.

Measured on Chernobyl with `gpt-5.4-mini`, nothing written:

| | concurrency 1 | concurrency 8 | |
|---|---|---|---|
| describe, 18 calls / 108 images | 76.0 s | **12.8 s** | 5.9x |
| summary, `batch=2`: 10 folds over 3 levels + final | 28.9 s | **16.7 s** | 1.7x |

Describe's wall is now its slowest single call (12.7 s). The summary gains
less because its layers, and the final call, are sequential by construction.
**A failed call still discards the run's answers** -- as it did sequentially,
but now with more paid calls already in flight when it happens.

---

## API

**Every component has the same signature, so one route runs any of them.**
`POST /videos/{id}/run/{component}` — `service.COMPONENTS` is a dispatch table,
and adding a component adds a row. A route and a handler per stage is what the
uniform signature removes. `params` is passed through as keyword arguments, so
**every component setting is reachable over HTTP** without the route knowing
any of them.

**`POST /videos` runs the whole pipeline; `run=false` registers and stops.**
The workflow deliberately carries no per-stage tuning — a scene grid with a
30 s floor, a uniform stride of 5, an `objects` vocabulary — so a caller
wanting those drives the components itself. Without `run=false` it had to run
the pipeline once on defaults first, paying for a describe it was about to
redo. `run=false` runs `media` and nothing else, answered rather than queued
because it is a container probe, and returns **201** with the id. `media` is
the one component the run route cannot reach: until it has run there is no id
to address.

**`/capabilities` publishes each component's parameters**, read off the
signature — name, type, default, required. Introspected for the reason
`defaults` is read off `workflow.Options`: a restated list drifts, and a
drifted one offers a parameter the component does not take or hides one it
does. That is the contract a configuration UI builds against.

**The API calls components, never drivers.** A driver is argparse; importing
one to reach the work behind it would make a server depend on a CLI.

**Slow work is queued, one job at a time.** Every heavy stage contends for the
same 8 GiB GPU: CLIP and YOLO in the video pass, Whisper and pyannote in the
audio one. Two videos at once doubles the resident weights and invites an
allocator failure halfway through the more expensive one.

**Validation is synchronous even though the work is not.** `workflow.validate`
returns problems as a list, so a contradiction is a 422 the caller sees at once
rather than a job that fails a minute later. What cannot be known without
opening the file — whether a track carries speech — still fails inside the job,
because that is a property of the media rather than of the request.

**Job records die with the process; artifacts do not.** `GET /jobs` says so.
`GET /videos` reads the directory rather than remembering.

**`stage` is what is running; `history` is what has finished.** The workflow
announces each component twice -- once by name before it runs, once with its
`Produced` after -- because a `stage` set only on completion names the
*previous* component throughout the longest stage of the run. Observed: a
poller read `stage: "cut"` for the whole of describe, four minutes of OpenAI
calls, which is indistinguishable from being stuck on cut. The before-name is
the caller's word for the step, since `boundaries.evidence` only reveals
itself as `boundaries.audio` or `boundaries.scenes` once it has run.

**A single-component job has no progress callback**, because components do not
take one — they are one step, and a step emitting its own progress would be
reporting to itself. The runner records the returned `Produced` instead, so a
one-component job ends with the same shape a workflow job builds up.

**Adding a prompt runs nothing, so it is not queued.** `POST /prompts` writes
one file and returns 201; the queue exists for work that contends for the GPU.
Deleting a question does not touch the descriptions it produced — a description
cost a paid call and records the question it was asked, so removing the question
does not make the answer untrue, it only stops new runs asking it.

**A placeholder is checked at write time, not at call time.** `{frames}` in an
instruction would raise inside `str.format` — after the frames are read, with
the request about to be paid for. `check()` parses the instruction against
`{n}`, `{span}`, `{vocabulary}` when it is submitted.

**The export surface lists what exists, not what could exist.** An audio-only
video advertises no manifest rather than offering a link that 404s — a broken
link reads as breakage, not as a stage that never ran.

**A form built from a registry defaults to whatever sorts first**, and that was
`stub`: an audio-only run completed in 10.6 s, reported 42 segments and 205
words, and wrote a transcript of `[stub0.0][stub0.1]`. Nothing was wrong enough
to report. `/capabilities` publishes `defaults` read off `workflow.Options`, so
the default lives in the dataclass the pipeline actually uses -- and, for the
three model roles, `providers.defaults()`, since those fields stay `None` until
a run resolves them.

---

## Building a client

Nothing about the pipeline needs to be hardcoded in a UI. `GET /capabilities`
publishes every registry, the defaults, and each component's parameters, so a
form is generated from it rather than kept in step with it.

**Two ways to run a video, and the choice is about tuning.**

    POST /videos                          upload + the whole pipeline.  202
      policy sampler use_video use_audio describer embedder tier sink index
      -> workflow defaults for everything per-stage

    POST /videos  run=false               upload, probe, stop.          201
      -> then POST /videos/{id}/run/{component} with `params`, in order:
         audio · boundaries.evidence · boundaries · video · cut ·
         describe · embed · aggregate

The second is the one a configuration UI wants: `min_s`, `every_n`,
`threshold`, `vocabulary`, `per_second` and the rest live on the component that
owns them, and `/capabilities.parameters` names them with their types and
defaults. `boundaries.evidence` returns a `Produced` with `skipped: [evidence]`
when the policy needs none, rather than nothing.

**Order is the caller's responsibility on that path.** `workflow.py` is the
reference for it; the dependencies are real — `boundaries` under `vad` or
`speaker` needs `audio` to have run, under `scene` needs
`boundaries.evidence`, and `cut`/`video` need the grid.

**Progress.** `stage` is what is running, `history` what has finished, and a
job's `detail` is the last `Produced`. Records die with the process; artifacts
do not, so a restarted server still lists every video from disk.

**Displaying results.** A search moment carries `descriptions` and `questions`
keyed by answer id, so grouping by sampler or by question needs no id parsing.
`GET /videos/{id}` lists only the artifacts that exist. Frames are
`GET /videos/{id}/frames/{index}` by the read index a manifest names.

**A score is a rank fusion, not a similarity.** `1/(k+best) + 0.5/(k+second)`
at k=10, so 0.1326 is the ceiling for a chunk contributing two units and means
"best ranked first, second ranked second" — never "this matched well". Measured:
a nonsense query scores identically to the best real one, because dense always
returns nearest neighbours and there is no relevance floor. If a UI shows a
number, show the ranks beside it — the CLI prints `clip(v7,t1)
uniform:text(v9,tNone)`, and `tNone` is how a reader sees the lexical half was
silent. The API carries them as `moments[].ranks`.

**Prompts are editable at runtime.** `GET/POST/DELETE /prompts`; a custom
question names a shape or brings its own, built-ins refuse edits with 409, and
adding a question invalidates nothing already described.

**`/search` narrows nine ways, and `/capabilities.search` publishes them** --
`sampler` (a pairing), `question`, `strategy` (one sampler's whole output),
`chunk_ids`, `window`, `after`/`before`, `structured`, `candidates`. None costs
a re-embedding: `text_hash` is over content alone, so every one of them is a
query-layer change over payload the unit already carries. Measured across both
backends, all six filters tried select **identical chunk sets**; rank order
differs on two, which is the lexical halves diverging.

**Time is resolved to chunk ids through the grid, never stored beside a
vector.** One mechanism for both stores, and no Qdrant payload change -- payload
is written only on upsert, so a span there would need a forced re-index.

**`structured` is only useful with `one_of`.**
`/capabilities.search.structured_fields` reads the shapes and lists exactly the
fields whose values are a vocabulary, so a form offers `severity: severe` and
not a free-text box that would match three different things.

**A moment carries the ranks.** `ranks[sampler_id] = {dense, text}`, `text:
null` meaning the lexical half was silent. Measured on the four-video corpus: a
nonsense query scores **0.1136** against a real one's **0.1294**, so there is no
relevance floor and the number alone says nothing. The ranks are the signal, and
the CLI printed them long before the API did.

**Scope is a set of videos, and that is one endpoint.** `video_ids` names them,
omitting it searches every one, and `video_id` is the one-element shorthand.
Searching one video, three, or all is the same question over a different set, so
a second route for "all" was a distinction the data never had.

**A search's notes are top-level, not only on its moments.** They rode on each
moment, so an empty result -- the case "nothing matched those filters" exists
for -- had nowhere to put them and reached the caller as a bare `[]`. Found by
searching with an embedder that had never indexed the video: a silent 200. The
note now names the embedder key, because that is exactly how such a search
comes back empty.

**Moments are keyed by `(video_id, chunk_id)`.** A chunk id indexes *one*
video's grid. Grouping on the id alone fused chunk 0 of two videos into one
moment with two unrelated accounts, and the agreement bonus scored that
collision above either real answer -- invisible while the scope was one video,
which is why it was written that way.

**`level` picks granularity and never mixes the two.** `moment` ranks chunks;
`video` ranks whole videos by their summary out of `video_embeddings`, which
`embed` writes from the `summary` aggregate (Postgres only; 4/4 on the test
corpus, winner 0.40-0.50 against runners-up of 0.05-0.32). One endpoint, but
never one ranking: a whole-video "moment" beside real ones is not something you
can play. At `level=video` the moment filters are reported under `ignored`
rather than silently dropped, because they narrow *inside* a video.

---

## Measured facts — do not re-derive

**Thresholds do not transfer between videos.** Same sampler, same domain,
`clip 0.96`:

| video | keep rate | median 1-second similarity |
|---|---|---|
| test.mp4 | 13.4% | 0.9868 |
| test2.mp4 | 18.0% | 0.9895 |
| test1.mp4 | 59.2% | 0.9605 |

That spread is the sampler working. **If cost is too high use
`min_interval_s` / `max_per_chunk`, not a lower threshold**: they keep the
most-changed frames and even out the per-chunk yield, where lowering the
threshold just keeps fewer and leaves the distribution lopsided.

**Do not solve for a fixed keep rate.** Forcing 15% turns a change sampler into
a worse uniform one. On a frozen video, solving for 15% produced threshold
1.000 which sampled *100%* of frames — encoder noise, since one PNG looped for
60 s H.264-encodes to 60 *different* frames.

**Detection samplers shifted 12–15% when the reader moved OpenCV → PyAV**
(yolo 40→45, objects 48→55; clip unchanged). Sub-LSB colour differences flip
detections near the confidence boundary.

**EasyOCR is 98.1% of the text sampler's cost** (129.6 ms of 132.0). The
descriptor is 2.5 ms. `canvas_size` is the only real lever and it is **not**
free: at 736 it is 3x faster but covers only 70% of the ink 1280 finds.

**PaddleOCR and craft-text-detector were evaluated and rejected.** EasyOCR's
detector *is* CRAFT, so the standalone package is the same model in an
unmaintained wrapper. PaddleOCR will not import here: `WinError 127` on
`cudnn_cnn64_9.dll` despite exactly-matching pinned versions.

**A difference hash is not usable for text change detection.** On a static
slide, dHash scored 0.86 agreement against *itself* frame to frame while two
*different* slides scored 0.88 — pure noise. dHash compares adjacent pixels and
most of a text region is flat background, so the sign it records comes from
sensor noise. Averaging down with `INTER_AREA`, centring and normalising gives
1.000 for a static slide and at most 0.80 between different ones.

**`objects` vocabulary is the highest-value setting**, and has no useful
default. A mismatched list found 2.4 detections/frame and labelled wire baskets
"shopping bag"; a matched one found 5.1.

---

## Environment traps

**`cublas64_12.dll` is not found, on a machine where it is present.**
CTranslate2 asks Windows for it *by name* at the first encode, not at import.
`nvidia-cublas-cu12` installs it to `site-packages/nvidia/cublas/bin`, which is
on no search path: since Python 3.8 an extension's dependencies do not resolve
from `PATH`, and `os.add_dll_directory` does not help because the load happens
lazily inside an already-initialised C++ extension. `audio/backends/cuda.py`
loads each by absolute path with `ctypes.WinDLL` first. The version is a
contract — this project's torch is cu130 and ships `cublas64_13.dll`, which is
not a substitute.

**pyannote 4.x returns `DiarizeOutput`, not `Annotation`.** The 3.x recipe
`pipeline(audio).itertracks(yield_label=True)` raises `AttributeError`. The
annotation is `.speaker_diarization`; `.exclusive_speaker_diarization` has
overlaps resolved, which is what this uses — a word cannot belong to two
speakers.

**`create table if not exists` never changes a table.** `install.sql` is re-run
against live databases, so a column added, dropped or retyped only inside the
`create` is simply not applied on every deployment that already had the table
-- and the first statement to reference it fails, or worse, a writer sends a
column PostgREST does not know. This bit three times: `structured`, then
`chunk_samplers.questions`, then `embeddings.sampler`/`question`.

So the `create`s hold the current shape and **section 10 holds the change**
for a live database -- an `alter`, a `drop ... if exists` -- each a no-op once
applied, and removed once every deployment has run it. Those migrations had
accumulated to about a third of the file before being cleared out: dropped
columns long gone, a backfill 0 rows needed, three superseded RPC signatures.

An audit is cheap and worth running after editing the file: parse the `create
table` bodies, diff them against PostgREST's deployed column list, and check
that every difference is covered in section 10.

**A generated column is not repaired by `add column if not exists`.** It is a
no-op when the column *exists*, so an `fts` built by an earlier expression
survived a re-run and the lexical half quietly stopped indexing the terms it is
best at. Changing `fts` means `drop column if exists fts` then the add, in
section 10; the column is generated, so nothing is lost.

**A `vector` column comes back from PostgREST as a *string*.** `vector(1536)`
arrives as the text `"[-0.0342,0.0450,...]"`, not a list. A cosine written
against a list of floats returned the not-comparable sentinel for every row and
`sorted` fell through to whatever order the rows arrived in, so it had never
ranked anything. Found by `video_embeddings` returning -1.0000 for four rows at
once; on the moment path (a dense-only fallback, since removed) the wrongness
was invisible, because table order is roughly chunk order and scores a
plausible MRR. `as_vector()` parses either form.

**PostgREST's cached schema is the fastest way to see what is really
deployed.** `GET /rest/v1/` with `Accept: application/openapi+json` lists every
column and RPC parameter it knows. That is how a table-name collision with the
previous pipeline was found before it silently ate writes.

**Exposing a Postgres schema is a dashboard setting, not SQL.** Until the
schema is in Settings → API → Exposed schemas, every request returns
`PGRST106`, which reads as a missing table rather than a missing setting.

**Embedded Qdrant takes an exclusive lock** on its storage folder -- and it is
not only a *two-process* problem, which is how this was filed until a server
reproduced it alone. `QdrantIndex` had no `close`, no `__del__` and no context
manager, so a client opened for one search never gave the lock back. A CLI run
never notices: the process exits and the lock goes with it. A long-lived API
leaks it on the first search and every later one fails with `Storage folder ...
is already accessed by another instance`, on a route that worked a minute
earlier -- and the traceback frame of the first failure pins the client alive,
so it never recovers.

`close()` is now called from a `finally` in both `retrieve.search` and
`embed.run` (`indexes.release`), because the not-indexed raise in the middle of
search is exactly the exit that leaked it. Verified: ten sequential opens
through one server process. Served mode (`url=`) holds no lock, so `close` is a
no-op there, and `supabase` is REST and has nothing to release.

**`weights/clip/ViT-B-32.pt` (338 MB) is NOT stale.** YOLO-World embeds its
vocabulary with OpenAI CLIP. That is a *different* CLIP from the one the scene
sampler loads through HuggingFace — different library, format and job.

**opencv variants shadow each other.** `opencv-python`,
`opencv-contrib-python` and `opencv-python-headless` all install `cv2`;
whichever wins depends on install order and nothing warns you. Uninstalling one
**breaks the others** — repair with `pip install --force-reinstall --no-deps
opencv-python==5.0.0.93`.

**Installing `gliner` downgraded transformers 5.15.1 → 5.13.1.** An import
check would not have caught it: it proves the import works, not that CLIP still
embeds. Verify by actually embedding — 512 dims, L2 norm 1.0.

**`PYTHONIOENCODING=utf-8`** is needed for third-party libraries that print
non-ASCII on Windows' cp1252 console.

**`stream.thread_type = "AUTO"`** is mandatory in PyAV, not an optimisation:
7.15 ms/frame without it against 3.97 with.

**Background processes started with `&` survive `pkill -f` on Windows.** Six
uvicorn servers from an earlier session held the Qdrant lock and blocked a
directory delete. `Get-CimInstance Win32_Process` and `Stop-Process` is what
actually finds and kills them.

---

## Current state

The pipeline runs end to end on real models — Whisper `small`, pyannote 3.1,
CLIP, YOLO, `gpt-5.4-mini`, `text-embedding-3-small`, GLiNER, DistilBERT —
writing documents to files and Postgres and vectors to Qdrant and pgvector.

Verified end to end **through the API**, from wiped local, Qdrant and Postgres
state, driving every stage with its own settings rather than the workflow's
defaults: `run=false` to register (201, no job), then `audio`,
`boundaries.evidence`, `boundaries` (scene, `min_s=30`, `max_s=60`), `video`
(`uniform:overview` at `every_n=5`, `clip:[mood,motion]` — two custom prompts
added over HTTP), `cut`, `describe`, `embed`, `aggregate --tier llm`.

Produced 4 chunks of 39.8-55.6 s on real scene cuts; `uniform` kept 11 frames
of a 54 s chunk, which is exactly every 5 s at 1/sec decimation; `clip` kept 51
on content change. 12 descriptions, 16 units under `clip:mood`, `clip:motion`,
`transcript` and `uniform:overview`, 8 aggregates. Every Postgres table exact
under both the secret and the publishable key — 4 chunks, 8 `chunk_samplers`
(one row per run, `questions` an array), 12 descriptions, 16 embeddings, 8
aggregates. `recovery.recreate` rebuilt the store **76/76 byte-identical**.

Both search filters verified against the RPC: `p_question` alone, `p_sampler`
alone, both intersecting, and a contradictory pair returning 0 rows. Lexical
firing by query type on a 55-unit corpus: literal 14/20 rows, paraphrase 20/20,
narration 20/20, nonsense **0/20**.

`shared.contracts.schemas --check` proves the dataclasses and the generated JSON
Schema still agree. The API serves **21 routes**; `docs/ROUTES.md` is the
reasoning and `/docs` the authority on shapes. The web client at `/app` drives
every route a run or a question needs, generating each form from
`/capabilities` rather than restating it.

**The live deployment, as of 2026-09-15** -- facts about one database, not the
code, and worth re-checking before trusting:

- `install.sql` has not been re-run since it was trimmed and widened. Both
  vector columns are still `vector(1536)` and `embeddings.timeline_fingerprint`
  still exists; section 10 fixes both.
- The publishable key in `.env` is refused with a 401 while the secret key
  works, so every read that goes through `db.client(write=False)` -- the
  `/db/*` routes, `video_embeddings`, the grid fallback in `retrieve` -- fails
  there. The code has not changed; the key needs re-copying from Settings >
  API. The "exact under both keys" result above predates it.

## Not built

- **Tests.** No suite. Verification is recreate's byte-comparison, the schema
  check, and ad-hoc scripts that are not kept.
- **An import checker.** The previous tree had one that AST-enforced the
  recovery invariant and proved every module loads after an install. Nothing
  enforces the recovery rule automatically now.
- **A bigger corpus.** `eval/harness.py` exists and runs, but 18 cases over 4
  videos is a direction, not a result -- the low-overlap band is n=3.
- **Sampler threshold calibration.** Scene thresholds sweep from cached scores;
  sampler thresholds (`clip 0.96`, `yolo 0.83`) do not.
- **The other providers, run.** Anthropic, Gemini, Mistral, Groq, OpenRouter,
  Together, DeepSeek, xAI, Voyage, Ollama and LM Studio are verified against a
  mock server's recording of the request, not against the real thing.
- **An answering model in-process.** Local answers go through a server --
  Ollama, LM Studio, llama.cpp, vLLM. Nothing loads a VLM into this process.
- **Supabase at any width, run.** `install.sql` drops the fixed width, but the
  live database has not had it re-run: both vector columns are still
  `vector(1536)`, so nothing but OpenAI's embedder can write there yet.
- **Keeping answers when a describe call fails.** One failed call raises and
  the run's other answers -- already paid for, and with concurrency more of
  them in flight -- are discarded. Writing the successes before raising would
  let a re-run pay only for the failures.
- **A stemmed, properly weighted sparse half.** `indexes.tokenize` does not
  stem and the sparse vector sends raw counts where `Modifier.IDF` expects BM25
  weights. `eval/harness.py` can now grade the change; nothing has been
  measured yet.
- **Reranking, query expansion, fusion tuning.** All plausible; none measured.
- **Live sources.** `Frame` carries no `gap_before`/`discontinuity` seams, so
  that is a retrofit through every stage rather than a field already there.
- **Entity linking from pixels, and labels worth the name.** `entities` reads
  descriptions only, so wording decides identity. `yolo` computes a CLIP
  embedding per person crop and discards it; keeping crops, and asking the
  describer to cite numbered boxes, would let appearance decide instead. The
  labelled set is 29 scored mentions.
- **Qdrant served mode, tested.** The `url=` path is written, never run against
  a server.
