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
  shared/          paths · documents · sinks · env · llm · db · rows · schemas
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
  describe/      7 prompts · frames · reader
    backends/      stub · openai_client
  rag/embed/     8 units · embedders · readable
    indexes/       qdrant · supabase
  rag/retrieve/ 10 search
  aggregate/     9 base · rendering
    statistics/    stats · speakers · coverage      free: arithmetic
    model/         ner · sentiment                  local: GPU models
    llm/           summary · chapters · events      llm: paid calls
api/               main (routes) · service (dispatch) · jobs (one worker)
recovery/          STANDALONE: recreate.py, imports nothing from the pipeline
db/
  supabase/        install.sql · reset.sql
  json/            document schemas, generated from the dataclasses
data/              everything a run writes; gitignored
  out/<id>/        media, transcript.raw, cuts, timeline, manifest, store,
                   transcript, descriptions, embedded, aggregates/
  out/_qdrant/     the embedded vector store, beside the videos not inside one
  uploads/         what the API parked until a run read it
weights/           detector and embedder checkpoints; a cache, not output
docs/ROUTES.md     the HTTP surface
```

100 files, ~10k lines.

## Commands

```bash
python -m falconvar.workflow media/x.mp4 --policy vad --sampler clip,yolo:overview
python -m falconvar.workflow media/x.mp4 --no-audio --sampler uniform:text
python -m falconvar.workflow media/x.mp4 --tier llm --sink file,supabase \
       --index qdrant,supabase

# one component at a time; per-stage tuning lives on these, not on workflow
python -m falconvar.media media/x.mp4
python -m falconvar.audio <id> --transcriber whisper --diarizer pyannote
python -m falconvar.boundaries <id> --policy scene --evidence --stride 5 --threshold 27
python -m falconvar.boundaries <id> --calibrate        # sweep, no decode
python -m falconvar.boundaries <id> --retune 45        # rethreshold cached scores
python -m falconvar.boundaries <id> --policy scene --chunk-duration 30
python -m falconvar.video <id> --sampler yolo --per-second 4 --min-interval 3
python -m falconvar.video <id> --sampler objects --vocabulary "crate,pallet"
python -m falconvar.video <id> --prune-store           # irreversible, opt-in
python -m falconvar.cut <id>
python -m falconvar.describe <id> --describer openai --limit 5   # costs money
python -m falconvar.rag.embed <id> --index qdrant,supabase
python -m falconvar.rag.retrieve "..." <id> --sampler transcript
python -m falconvar.aggregate <id> --tier llm

python -m falconvar.shared.schemas --check     # CI: are the schemas stale
python -m recovery.recreate data/out/<id>/manifest.json --verify data/out/<id>/store
#      ^ from the checkout root: recovery/ is not an installed package, on purpose
python -m uvicorn api.main:app --port 8000     # /docs for the schema
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
13/13, 60/60, 83/83. If a change makes recreate non-identical, the change is
wrong.

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

**A pairing is keyed by both halves.** `sampler_id` is `name:prompt` when a
question is paired and the bare name otherwise. Keying by the question alone
breaks the moment any sampler can ask anything: `yolo:overview` and
`clip:overview` hold different frames and would collide on one manifest key.
Keying by the strategy alone loses the question, which is the more useful half
when reading a search result. The id is what the manifest, `descriptions.json`,
`embeddings.sampler_id` and `--sampler` all inherit.

**Ownership is over the questions on a chunk, never the sampler ids.** `OWNER`
is keyed by question, and the two are only equal while no sampler is paired
with someone else's question. With `yolo:overview` present, passing ids would
have `clip` surrender `people` to a call whose schema owns no keys at all, and
the field would leave the document with everything still well-formed. Verified
end to end: `clip,yolo:overview,uniform:text` on one chunk — `clip` keeps
`people` and gives up `visible_text`, nothing is answered twice, the merge is a
plain union of all seven keys.

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
prompt carries. It owns no keys, so it takes none from the scene question and
costs a `clip` running beside it nothing. Measured: 84 and 86 words, 4
sentences, zero structured keys.

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
kind of silent no-op, since the output looks complete. The `model` block
carries a hash of every instruction and schema in `prompts.py`: editing a
prompt changes the output but not the model id.

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
querying one is a *filter* — which gives up the agreement signal: a chunk can
then contribute at most one term, so scores roughly halve.

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

Nor is the final summary. `db/supabase/install.sql` creates
`video_embeddings` and `units.py` has only `from_descriptions` and
`from_transcript`, so the table is empty after a complete `--tier llm` run into
Postgres — verified, 0 rows where every other table is exactly full. The
argument for it stands (`embeddings` answers *which twenty seconds*, a summary
answers *which video*, and a video is not a moment you can play); the code does
not exist. See "Not built".

**Spans are resolved through the timeline, never trusted from the model.** It
is asked for chunk ids, which it can copy; times it would invent.

**`inputs_fingerprint` is a hash of the chunk text actually read.** A summary of
descriptions since rewritten reads perfectly, which is precisely why staleness
cannot be left to a reader to notice.

---

## API

**Every component has the same signature, so one route runs any of them.**
`POST /videos/{id}/run/{component}` — `service.COMPONENTS` is a dispatch table,
and adding a component adds a row. A route and a handler per stage is what the
uniform signature removes.

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

**The export surface lists what exists, not what could exist.** An audio-only
video advertises no manifest rather than offering a link that 404s — a broken
link reads as breakage, not as a stage that never ran.

**A form built from a registry defaults to whatever sorts first**, and that was
`stub`: an audio-only run completed in 10.6 s, reported 42 segments and 205
words, and wrote a transcript of `[stub0.0][stub0.1]`. Nothing was wrong enough
to report. `/capabilities` publishes `defaults` read off `workflow.Options`, so
the default lives in the dataclass the pipeline actually uses.

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

**`add column if not exists fts` cannot repair a stale generated column.** It
is a no-op when the column exists, so an `fts` built by an earlier version of
the file survived a re-run untouched and the lexical half quietly stopped
indexing the terms it is best at. Nothing reported it. The statement is `drop
column if exists` followed by an unconditional add; the column is generated, so
nothing is lost.

**PostgREST's cached schema is the fastest way to see what is really
deployed.** `GET /rest/v1/` with `Accept: application/openapi+json` lists every
column and RPC parameter it knows. That is how a table-name collision with the
previous pipeline was found before it silently ate writes.

**Exposing a Postgres schema is a dashboard setting, not SQL.** Until the
schema is in Settings → API → Exposed schemas, every request returns
`PGRST106`, which reads as a missing table rather than a missing setting.

**Embedded Qdrant takes an exclusive lock** on its storage folder, so two
processes cannot open it at once. Sequential CLI use never hits this; an API
serving search while a run ingests would. Served mode (`url=`) has no such
limit.

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

Verified through the API on Chernobyl (205 s, `vad`, `clip,uniform:text`,
`--tier llm`): all nine components and all eight aggregators in 287.6 s, none
skipped. 34 segments, 428 words placed with none outside the grid, 14 chunks,
28 descriptions, 41 embedded units, 125.1 wpm, the explosion at 94.439 s and
chapters tiling 0-205.28 contiguously. `recovery.recreate` rebuilt the store
206/206 byte-identical. A second identical run took **32.1 s**: describe
skipped 28, embed found 41 unchanged, aggregate found 8 current -- all three
resume fingerprints holding at once.

Both backends verified against a schema installed from scratch. A run with
`--sink file,supabase --index qdrant,supabase` filled every table to exactly
the expected count -- 14 chunks, 28 chunk_samplers, 28 descriptions, 41
embeddings, 8 aggregates -- and identical counts under the publishable key, so
RLS reads what it should rather than silently denying. A second run left every
count unchanged: upsert, not append. 28 descriptions re-rendered from the
Postgres copy produced text byte-identical to the file although 14 came back
with different `jsonb` key order, so the sort-at-every-level fix holds where it
was designed to. Both indexes returned the same top moment on three queries and
agreed on `0.1326` for one of them; they diverge at rank 3, which is the
lexical-half difference recorded above.

`shared.schemas --check` proves the dataclasses, the generated JSON Schema and
the SQL still agree. The API serves 13 routes.

## Not built

- **Tests.** No suite. Verification is recreate's byte-comparison, the schema
  check, and ad-hoc scripts that are not kept.
- **An import checker.** The previous tree had one that AST-enforced the
  recovery invariant and proved every module loads after an install. Nothing
  enforces the recovery rule automatically now.
- **An eval harness.** Every ranking claim above predates the current tree and
  was measured on the previous pipeline's corpus. Nothing here can currently
  show a retrieval change helps.
- **Sampler threshold calibration.** Scene thresholds sweep from cached scores;
  sampler thresholds (`clip 0.96`, `yolo 0.83`) do not.
- **A local embedder, run.** Written and guarded, but every measurement is
  OpenAI.
- **Video-level embedding.** `falconvar.video_embeddings` exists in the DDL and
  nothing writes it: `units.py` builds units from descriptions and transcripts
  only. Searching "which video is this about" is therefore not possible, only
  "which moment".
- **Filterable structured values.** The mechanism works; the values are free
  text, so a filter for `cashier` matches everything. Needs an `enum`.
- **Live sources.** `Frame` carries no `gap_before`/`discontinuity` seams, so
  that is a retrofit through every stage rather than a field already there.
- **Entity narratives.** Linking the same person or object across chunks, then
  asking for one narrative per entity. The specialists already return one bound
  object per entity, which is the input it needs; every aggregator reads chunks
  as independent documents.
- **Qdrant served mode, tested.** The `url=` path is written, never run against
  a server.
