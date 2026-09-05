# FalCONvar — working notes

Video RAG ingestion. A video goes in; a **manifest** comes out saying which
frames are worth describing, grouped into retrievable chunks, with enough
addressing information to fetch those frames back later.

Everything under `ver2/`. Version 1 has been deleted.

---

## Layout

```
web/           the browser client: one page, no build step
  index.html     the form, the search box, the moment cards
  app.js         fetch + render; everything offered comes from /capabilities
  style.css
api/           HTTP in front of it all
  main.py        routes; queued for the slow stages, immediate for search
  service.py     the pipeline in terms a request can supply
  jobs.py        one background worker, in-memory job records
ver2/
  driver.py      the CLI for both streams (argparse only)
  orchestrate.py BOTH streams, one grid: open, split, decide the policy, run
  timeline.py    the shared chunk grid: spans + the policy that produced them
  video/
    ingest/
      source/    probe, sequential read, decimation, random access (PyAV)
      chunker/   media time -> chunk id    (uniform | scene | fixed)
      samplers/  which decimated frames to keep
                 (uniform|clip|yolo|objects|text); any may be
                 paired with any question as `name:prompt`
                 policy: base.py, uniform.py, scene.py, detection.py (the base
                 for detector-driven ones) + people.py, objects.py, ocr.py
        components/  perception: detectors, descriptors, embedders. Every model
                     weight lives below this line and none above it.
      output/    manifest sinks (file | supabase | both) + frame store
      pipeline.py  ingest() -- one decode pass, feeding every stage
      driver.py  the CLI (argparse only; no pipeline logic)
      calibrate.py what a threshold would cost here, and where it must not go
    describe/
      input/     manifest (file|db), live chunk stream, pixels from the store
      describers/  the Describer protocol + a registry (stub | openai)
      vlm/       the OpenAI call + prompts.py, one prompt per sampler
      output/    description sinks (file | supabase | both)
      reader.py  describe() -- one call per (chunk, sampler)
      driver.py  the CLI
  audio/         no describe stage: a transcript is already the text
    source.py    decode whole -> 16k mono float32 (PyAV) + silence guard
    cuda.py      preload the vendored CUDA DLLs before CTranslate2 asks
    transcribe/  Transcriber protocol + registry (stub | whisper)
    diarize/     Diarizer protocol + registry (none | pyannote)
    align.py     words + speaker turns -> attributed segments
    segment/     transcript -> boundaries (vad | speaker) + cut.py
    output/      transcript sinks (file; one call, not three)
    reader.py    listen() -- decode, transcribe, diarize, attribute
    driver.py    the CLI, audio alone
  embed/         SHARED: descriptions and transcripts both land here
    defaults.py  which embedder + which index, shared by both CLIs
    embedders/   Embedder protocol + registry (openai | local)
    index/       VectorIndex: qdrant (embedded) | pgvector | both, + build()
    units.py     description -> embeddable unit, keyed by a hash of its text
    indexer.py   embed only what changed
    driver.py    the CLI
  retrieve/      SHARED
    search.py    query -> ranked descriptions -> ranked moments
    driver.py    the CLI
  aggregate/     SHARED: video-level structure over what the chunks said
    base.py      the Aggregator protocol + Context (joins the documents)
    stats.py speakers.py novelty.py       free: arithmetic and existing vectors
    ner.py sentiment.py                   local: GPU models
    summary.py chapters.py events.py      llm: paid calls
    llm.py       one rendering of the video, shared by the text aggregators
    output/      aggregate sinks (file | supabase | both)
    reader.py    aggregate() -- resolve order, run, skip what is current
    driver.py    the CLI
  llm.py         the text model call + the key, shared by describe and aggregate
  fanout.py      primary + best-effort secondaries, shared by every stage
  db.py          the Supabase client + the reads more than one stage needs
  recovery/
    recreate.py  STANDALONE: rebuild a store from a manifest + the video
    supabase_manifest.py  STANDALONE: video id -> manifest file (urllib only)
    supabase_description.py  STANDALONE: video id -> description document
  imports.py     import everything, exercise it, report what loaded
eval/
  queries.py     literal/paraphrase query pairs, disjointness verified
  render_ab.py   A/B a change to what gets embedded, paired bootstrap
  retrieval.py   the shipped path, with and without its lexical half
  aggregation.py how a chunk scores from its descriptions' ranks
```

15650 lines, 132 files.

## Commands

```bash
python -m ver2.video.ingest.driver media/test2.mp4 --sampler clip --frame-store
python -m ver2.video.ingest.driver video.mp4 --sampler uniform,clip,yolo,objects,text \
       --min-interval 3 --chunking scene --scene-threshold 15
python -m ver2.video.ingest.driver video.mp4 --sampler objects --vocabulary "crate,pallet"
python -m ver2.video.ingest.driver video.mp4 --sink file,supabase   # both; file is primary
python -m ver2.video.ingest.calibrate video.mp4 --sampler clip
python -m ver2.video.describe.driver out/<id>/manifest.json     # describe -> json
python -m ver2.video.describe.driver --video-id <id> --sink file,supabase
python -m ver2.video.describe.driver --video-id <id> --follow    # tail a live ingest
python -m ver2.video.describe.driver out/<id>/manifest.json --describer openai
       --model gpt-5.4-mini --sink file,supabase          # costs money
python -m ver2.embed.driver out/<id>/descriptions.json     # -> pgvector
python -m ver2.embed.driver out/<id>/transcript.json      # audio-only video
python -m ver2.embed.driver --video-id <id> --index pgvector,qdrant
python -m ver2.retrieve.driver "people at the checkout" --moments 3
python -m ver2.retrieve.driver "..." --sampler yolo        # one question only
python -m ver2.retrieve.driver "..." --index qdrant        # dense only, says so
python -m ver2.recovery.supabase_manifest --list          # what is published
python -m ver2.recovery.supabase_manifest <id>            # -> <id>.json
python -m ver2.recovery.supabase_description <id>         # -> descriptions json
python -m ver2.recovery.recreate out/<id>/manifest.json --out rebuilt/
python -m ver2.driver media/x.mp4 --sampler clip --chunking uniform
python -m ver2.driver media/x.mp4 --no-video --chunking vad    # sound alone
python -m ver2.driver media/x.mp4 --no-audio --chunking scene  # picture alone
python -m ver2.driver media/x.mp4 --sampler clip --chunking vad   # audio grid
python -m ver2.driver media/x.mp4 --sampler clip --chunking scene # video grid
python -m ver2.audio.driver media/x.mp4 --chunking speaker        # audio alone
python -m ver2.video.ingest.driver v.mp4 --sampler uniform:overview   # prose only
python -m ver2.video.ingest.driver v.mp4 --sampler yolo:overview      # their frames, prose
python -m ver2.video.ingest.driver v.mp4 --sampler uniform:text        --every-frames 10                    # read the screen on a stride
python -m uvicorn api.main:app --port 8000   # / for the site, /docs for the schema
python -m ver2.aggregate.driver <id> --tier free   # no model, no network
python -m ver2.aggregate.driver <id> --tier llm    # + summary, chapters, events
python -m ver2.imports                      # after ANY install
python -m eval.queries out/<id>/descriptions.json   # -> eval/query_pairs.json
python -m eval.render_ab                    # compare renderings, paired CI
python -m eval.retrieval                    # hybrid vs dense on the shipped path
python -m eval.aggregation                  # chunk scoring, count-bias check
# schema.sql       runnable DDL: every table, function and RLS policy
# docs/SCHEMAS.md  what every field means, local JSON and Postgres alike
# docs/ROUTES.md   the HTTP surface and the reasoning behind it
# docs/RUN.md      how to run the web app, the API and every CLI
```

**Everything a video produces lives under `out/<video-id>/`** — `manifest.json`,
`store/`, `descriptions.json`. Grouped by video rather than by artifact type, so
one video's whole output is one thing to inspect, copy or delete, and later
stages add to it without a new top-level directory. Bare `--frame-store` uses
`out/<video-id>/store/`; an explicit path is used exactly as given.

---

## Invariants — do not break these

**`recovery/` imports nothing from `ver2`.** Hand someone those files, a
manifest (or just a video id) and the video, and they rebuild the store byte
for byte with only `av`, `opencv-python`, `numpy`. If recovery imported the
pipeline it could lean on a default living in code rather than in the
manifest, and the manifest's claim to be authoritative would go untested.
`imports.py` enforces this by AST-parsing every file in `recovery/`; a
`from ver2...` in any of them fails the check.

**The recovery kit is split by question.** `supabase_manifest.py` answers
*where is the manifest* and `supabase_description.py` *what was said about it*
(both `urllib`, no `supabase-py` — that would break the
hand-someone-the-files property); `recreate.py` answers *what frames does it
name*, and never touches the network. Someone who already has a manifest runs
recreate directly.

**Ask for a word count, not "several sentences".** Once structured fields
arrived the model sized the summary as one field among many: 105 median words
against 363 in the prose-only era. Saying "at least 150 words" and that the
fields *index* the summary rather than replace it took it to **246 median
words** (min 177). The summary is the only text that gets embedded, so its
length is a retrieval parameter, not a style preference.

**Embed the summary *and* the structured fields, not the summary alone.**
Measured on test1, 22 disjoint query pairs, dense MRR (random 0.457):

| embedded from | literal | paraphrase | mean pairwise similarity |
|---|---|---|---|
| summary | 0.528 | 0.522 | 0.854 |
| structured | 0.636 | 0.586 | **0.802** |
| both | **0.705** | **0.608** | 0.838 |

The earlier argument for summary-only — that relations live in the grammar and
flat lists destroy them — stopped being true when the specialists began
returning one bound object per entity. Rendering `{appearance, clothing, role,
action}` as a line keeps who-did-what *and* carries far more distinctive
content per token: every summary repeats the same setting, the fields do not.
Caveat: the queries were derived from structured phrases, which favours them;
"both" wins on either reading. Through the shipped path this lands at literal
MRR 0.731, paraphrase 0.607. These figures predate the rendering change below
and a re-described corpus, so treat the ordering as the finding, not the
decimals.

**What gets embedded must not depend on which copy it was read from.**
`jsonb` preserves array order but not object key order -- it stores keys by
(length, bytewise) -- so a description read back from Postgres hands back
`{role, action, clothing, appearance}` where the file had `{appearance,
clothing, role, action}`. `render` joined `item.values()` in iteration order,
so the same person became different text, a different `text_hash` and a
different vector depending on its source: cosine 0.995 between the two, close
enough that no ranking looked wrong. Indexing from the file and from
`--video-id` alternately re-embedded the same five specialist units forever,
each run reporting "description changed" when nothing had. Only specialists
were hit -- `clip`'s fields are strings and string lists, and `render` already
sorted the top level; determinism had been handled one level deep and not two.
Keys are now sorted, so the text is a function of content alone. Verified: all
ten units hash identically from both sources, and alternating runs skip all
ten.

**The rendering names each attribute, and that was not a retrieval decision.**
Measured on test1, 29 query pairs with zero shared content words, comparing
three renderings of the same fields against the same summaries and queries:

| rendering | literal MRR | paraphrase MRR |
|---|---|---|
| values, iteration order (the bug) | 0.516 | 0.511 |
| values, sorted | 0.536 | 0.499 |
| labelled, sorted (shipped) | 0.509 | 0.528 |

Every paired bootstrap CI on the per-query difference spans zero, so **the
corpus cannot tell these apart** -- five chunks of one shop, where dense
retrieval scores 0.51 against 0.457 random. Names are kept for structure, not
score: the separator between two fields used to be `", "`, which also occurs
inside them ("dark green top, dark pants, black sneakers"), so field
boundaries were invisible; and a field destined to be a filterable enum should
be a named term rather than a bare word among clothing. It costs ~6% more
characters. Three separators, one per level: `. ` between fields, `|` between
entities, `;` between one entity's attributes -- though `;` can still occur
inside a value, so the names, not the punctuation, are what make it
recoverable.

**A structured field is only filterable if its values are a vocabulary.**
`role` is free text, so one video produced `cashier`, `customer`,
`cashier or customer near checkout`, `child customer`, `customer at checkout` —
and a filter for "cashier" matches all five chunks, which is no filter at all.
The mechanism works (payload in Qdrant, `jsonb` + GIN in Postgres); what is
missing is an `enum` on the fields meant to be filtered.

**The hybrid is deployed, and the lexical half earns its place.** Measured on
test1 through the shipped path — 7 literal queries, each a string that appears
in exactly one chunk's `visible_text`, ground truth being that chunk:

| | top-1 | MRR |
|---|---|---|
| qdrant (dense only) | 0.429 | 0.600 |
| pgvector (dense + BM25, RRF) | **0.714** | **0.857** |

Better on 4 of 7, tied on 3, **never worse** — which is the property RRF is
chosen for. `"9p"` went from dense rank 5 to 2, `"£1.85"` from 4 to 1. Every
query produced exactly one `t1` on the right chunk, so the lexical index is
precise, not merely present.

Two caveats, both structural. The queries are verbatim strings lifted from the
corpus, which favours BM25 the way the structured-derived queries favoured the
structured fields — it measures the ceiling, not the average. And the dense
half is not embarrassed here only because `embed_text` already contains the
structured fields, so the timestamps are *in* the vectors; against
summary-only embeddings the gap would be far wider.

**The shipped retrieval path, measured by withholding only the lexical half.**
Same vectors, same table, same ranking code -- `p_query_text` withheld so
`search_embeddings` skips its `by_text` CTE and fuses nothing. 29 query
pairs, ground truth the chunk a query was written from, 5 chunks (random:
top-1 0.200, MRR 0.457):

| | | top-1 | MRR | queries with a text rank |
|---|---|---|---|---|
| literal | dense only | 0.276 | 0.509 | -- |
| literal | dense + BM25 | **0.379** | **0.649** | **29/29** |
| paraphrase | dense only | 0.241 | 0.528 | -- |
| paraphrase | dense + BM25 | 0.241 | 0.528 | **0/29** |

Paired bootstrap on per-query reciprocal rank: literal **+0.140, CI [+0.020,
+0.260]** -- the lexical half helps, significantly. Paraphrase **+0.000, CI
[0, 0]** -- not degraded, *identical*, because BM25 matched nothing at all on
queries built to share no content word with the corpus.

That is the argument for RRF in one table. Fusing costs nothing when one half
has no opinion, and the half with no opinion is silent rather than wrong.
`eval/retrieval.py` reproduces it.

**The fallback is confirmed in both directions.** `"youngster examining
merchandise racks"` — no content word shared with any description — returns
`tNone` on every hit: BM25 contributes nothing and dense answers alone. Swap
in `"a child near the shelves"` and four descriptions carry a text rank. Neither
half knows which kind of query it was handed, and a search box gives no signal,
which is the whole argument for fusing both rankings rather than choosing.

**Each half of the hybrid is strong exactly where the other fails.** Measured
on test1, 5 chunks, 22 query pairs with **zero shared content words** between
the literal and paraphrase forms (random: top-1 20%, MRR 0.457):

| | literal top-1 | literal MRR | paraphrase top-1 | paraphrase MRR |
|---|---|---|---|---|
| dense | 23% | 0.528 | 23% | 0.522 |
| BM25 | **59%** | **0.752** | 18% | 0.468 |

BM25 collapses to chance when the words change; dense is unmoved by
rephrasing. Neither knows which kind of query it was handed, and a search box
gives no signal, which is the whole argument for fusing both rankings.

Dense being only weakly informative here (0.522 against 0.457 random) is a
corpus property, not a verdict: ten descriptions of one shop over 98 seconds
are near-identical semantically, mean pairwise cosine 0.85.

**Generating paraphrase test queries needs verification.** Asked to "share no
content words", the model kept a median 50% of them, and BM25 appeared to win
on paraphrases as a result. Forbidding each query's own content words by name
and re-checking disjointness — retrying, then dropping what cannot be rewritten
— is what made the test mean anything.

**Resume is keyed on the manifest, the describer *and* the prompts.** A stored description
counts as done only if its `manifest_fingerprint` matches and its `model`
block is identical. Without the model check, describing with the stub and then
switching to a real one skips all ten pairs and reports success having done
nothing — the most expensive kind of silent no-op, since the output looks
complete. The `model` block therefore also carries `prompts`, a hash of every
instruction and schema in `vlm/prompts.py`: editing a prompt changes the output
but not the model id, and a re-run reported "10 skipped" while measuring the
summary-length fix.

**Both copies must be able to catch up.** A describe run's resume set is the
**intersection** of what every sink already holds, not the primary's answer.
Asking only the primary was tried and is wrong: a secondary dropped mid-run
reports as done forever and nothing repairs it. Refilling a gap costs one
describer call, which is the right price for a second copy that is actually
complete.

**Recreate is the end-to-end oracle.** Any change to encoding, addressing or
the manifest is verified by rebuilding a store and byte-comparing. Verified
repeatedly at 24/24, 25/25, 61/61, 93/93, 350/350. If a change makes recreate
non-identical, the change is wrong.

**Every sink hears the same thing.** `begin` / `chunk_closed` / `finish`, with
identical arguments to all of them. A sink is constructed with what its own
destination needs and learns nothing else that way; interpreting those facts
into a file rewrite or an INSERT is the sink's business. Writing to two places
is then a fan-out, not a branch in the pipeline.

**The per-sampler prompt is the point, and it shows.** Measured on test1
chunk 2, same 20 s window: the `clip` prompt returned the room — layout,
lighting, fixtures, what changed between frames; the `yolo` prompt went
straight to people, their positions, clothing and actions, with no scene
preamble at all. Ten calls carried 105 images for 80 distinct frames (a frame
two samplers chose is sent twice, once per question), 5.3-9.0 s each, 1185-2577
characters back.

**A specialist returns objects, never parallel lists.** One entry per person
carrying `appearance`, `clothing`, `role` and `action` — not a list of people
beside a list of actions, which does not say who did what and cannot be made to
say it afterwards. Same for objects (`object`/`appearance`/`context`) and text
(`text`/`context`). This was v1's mistake and v0 had it right.

**The scene question is the fallback, and it narrows.** `clip`, `uniform` and
any unregistered sampler ask it: setting, people, objects, visible_text,
actions, changes, tags. But a key a specialist owns is *removed from the scene
schema when that specialist ran on the same chunk*, so ownership stays
exclusive per chunk without anyone being asked twice, and a uniform-only run
still records everything rather than a stub.

**Ownership is over the questions on a chunk, never the sampler ids.**
`OWNER` is keyed by question, and `schema_for` was already passed a resolved
question as its first argument -- but its `siblings` arrived as raw ids from
the manifest. That worked only while every id equalled its question, which is
no longer true. `reader._questions_on` resolves each sampler on the chunk
through the same `question_for` the call itself uses, so what a schema gives up
and what is actually asked cannot disagree.

Measured on test2, `uniform,uniform:overview,uniform:yolo`: with ids as
siblings the scene call keeps `people` and `uniform:yolo` answers it too --
two answers to one key, resolved arbitrarily by `merge`. With questions, the
scene call gives `people` up and exactly one call answers it. The mirror case
is worse and was the reason to look: a `yolo` sampler paired with `overview`
would have had `clip` surrender `people` to a call whose schema owns no keys at
all, and the field would leave the document with everything still well-formed.
Verified end to end -- `yolo:overview,clip` on test2 keeps all seven scene
keys, `yolo:overview` owns none, nothing is answered twice.

A single shared schema was tried and is wrong: every field being present means
the model may fill any of them, and it does — asked about people it returned a
paragraph about the room, paid for twice, leaving two `setting` values with no
rule for which wins. A prompt saying "focus on X" is a request; a schema with
no Y field is a guarantee.

**Structured answers are ~3x longer than prose.** At `max_output_tokens=700`
the people schema truncated mid-string on a busy frame and came back as
unparseable JSON. The default is 2000, and truncation is detected from the
response's own `status`/`incomplete_details` rather than inferred from a JSON
error further down.

**Which frames, and what to ask, are independent.** Every sampler takes a
`prompt` -- it lives on the base class -- so any strategy pairs with any
question as `name:prompt`. `uniform:text` reads the screen on a stride;
`yolo:overview` keeps frames where the people changed and asks for prose
instead of the structured people call. Unpaired, the question is the sampler's
own name.

It was `uniform` alone that took a `prompt`, enforced by a hardcoded
`if name not in ("uniform", "overview")` in two drivers, and there was never a
reason for the restriction: `question_for`, `for_sampler` and `schema_for` all
already took a *question* rather than a sampler. Generalising deleted a special
case rather than adding one. Passing a sampler rather than a prompt was the
first design and is worse -- the wrapper never runs the inner sampler's logic,
so it would construct an EasyOCR or CLIP object purely to borrow its name, and
it could never extend to a question no sampler corresponds to.

**A pairing is keyed by both halves.** `sampler_id` is `name:prompt` when a
question is paired and the bare name otherwise. Keying by the question alone
was the earlier rule and it stops working the moment any sampler can ask
anything: `yolo:overview` and `clip:overview` hold different frames and would
collide on one manifest key. The id is what the manifest, `descriptions.json`,
`chunk_embeddings.sampler` and `--sampler` all inherit, so `uniform:text` runs
are keyed `uniform:text` where they used to be keyed `text` -- old vectors need
re-indexing to be reachable by the new name.

**An unknown question is rejected, not fallen through.** `question_for` falls
back to the scene question by design, which makes `yolo:overvew` a run that
completes, costs money and answers something nobody asked. `prompts.QUESTIONS`
is the vocabulary and it is checked in all four places a run can start: both
CLIs, `orchestrate.validate`, and `describe()` itself before a single call --
because the manifest is written by a stage that cannot see the vocabulary.

**Ingest still does not depend on describe.** A sampler records the question as
an opaque string and `base.py` never reads it; the DAG edge runs describe ->
ingest and `imports.py` enforces it. Only the *drivers* import `prompts`, and
function-locally: they are composition roots, and validating a typo is the one
thing they want the vocabulary for. That is also why `vlm/__init__.py` resolves
`OpenAIDescriber` through a module `__getattr__` -- it re-exported the client
eagerly, so reading `vlm.prompts` pulled in `openai`, which its own docstring
already claimed it did not.

The point is cost. `text` fires *when the writing changes*, and finding that
out costs EasyOCR on every decimated frame -- 98.1% of that sampler's total.
"Read the screen every so often" wants none of that: measured on Chernobyl,
`uniform:text` at a stride of 6 (6 s at `per_second=1`) ran **no model at
ingest** and the VLM still transcribed
`RadioFreeEurope RadioLiberty`, `BYELORUSSIAN S.S.R.`, `REACTOR 1`, `1977`
correctly. The two are different questions, not two settings of one.

`prompt` is recorded in the sampler's config, so it is in the manifest and
therefore inside `manifest_fingerprint` -- editing it invalidates describe's
resume exactly as editing `vlm/prompts.py` does. `question_for` reads it back
and falls through to the sampler id, so nothing that does not set it changes.

**`overview` is a question with no fields at all**, and only a question --
there is no `OverviewSampler` any more. Its whole content was
`prompt="overview"`, which every sampler now expresses, so `--sampler overview`
is spelled `--sampler uniform:overview`. Summary only, 4 to 5
sentences, and it overrides the ">= 150 words" instruction that every other
prompt carries -- the length rule exists because the summary is what gets
embedded and its length is a retrieval parameter, but an overview is chosen
precisely when a paragraph is more than the moment deserves. It owns no keys,
so it takes none from the scene question and costs a `clip` running beside it
nothing. Measured: 84 and 86 words, 4 sentences, zero structured keys.

**`uniform` counts decimated frames, not seconds.** `every_n` is a stride over
the stream the sampler was actually offered, read off `chunk_local_index` --
the one thing about position a sampler is handed. Counting seconds instead
meant dividing a wall of media time the sampler had no other business knowing,
which is the one place a sampler decided on something other than the frames
flowing past it.

The cadence in seconds is therefore a consequence of decimation rather than a
second setting beside it: `every_n=3` is one frame every 3 s at
`per_second=1` and one every 0.75 s at 4. That is the intent -- `per_second`
already decides how much of the video anything downstream may look at, so a
positional sampler is a stride over what survived it, and the two numbers stop
being two ways to say the same thing. Verified on test2 with `--every-frames 3`:
identical `chunk_local_index` sequences (0, 3, 6, ...) at `per_second` 1 and 4,
36.1% and 33.9% of decimated frames kept, at 3 s and 0.767 s spacing.

A cadence in *seconds* regardless of decimation is still expressible, and by
the mechanism every sampler shares: `min_interval_s`, enforced in the base
class before the strategy runs. Verified: `--every-frames 2 --per-second 4
--min-interval 3` keeps frames exactly 3 s apart, at indices 0, 12, 24 --
both constraints holding at once. The cadence is no longer folded *into*
`min_interval_s`, which is what let a stride and an interval be set
independently; nothing is lost by deciding in `propose` here because this
sampler runs no model, so a frame it turns down costs what one the base class
never offered it costs.

The default is frame-native and shared: 1, every decimated frame. Unlike
`--threshold`, which is left unset so each change sampler keeps the value
calibrated for what *it* compares, one stride for every positional sampler is
correct -- because **`overview` is a prompt, not a cadence**. It briefly had
its own `every_s=5.0` as `OverviewSampler`; that was dead code (every caller
passed `every_seconds` explicitly), and making the flag optional resurrected it
as a divergence where the same question kept different frames depending on
which spelling was typed. The class is gone entirely now.

**A prompt is chosen by sampler, never shared.** The manifest records *why*
each frame was kept and that is the most useful thing it knows: a person
sampler fired because the people changed, a text sampler because the writing
did. One prompt for all of them returns near-identical captions for the same
moment — several near-identical embeddings in a retrieval index, which is worse
than one. `vlm/prompts.py` is data: a new sampler adds an entry, an unknown one
falls back to `GENERIC`.

**The module graph is a DAG, and every edge names a produced artifact.** `db`
and `fanout` import nothing; `ingest` uses both; `describe` adds `FrameStore`;
`embed` uses only `db` and `fanout`; `retrieve` uses `embed` -- an `Embedder`
and a `VectorIndex` -- plus `db.load_env`. `retrieve` importing `describe` for
a Supabase helper was the one edge with no domain reason behind it, and
extracting `db.py` removed it along with four copies of the same
connect-and-complain logic.

**Each stage owns the store it writes.** `ingest` owns the manifest sinks and
the frame store, `describe` the description sinks, `embed` the vector index --
so `retrieve` reads an index the way `describe` reads a `FrameStore`, from the
stage that defined it. This is why the embed/retrieve split puts `index/` on
the writing side even though searching is the more visible use of it: the
alternative is a store owned by neither, defined twice.

**Postgres is the default index, and the reason is the lexical half.** Qdrant
is dense-only, so a search against it silently drops a component worth 0.429
against 0.714 top-1 on the same corpus. Both are real options and the local
stack is the one to develop next, but the weaker one should be chosen out
loud: `--index qdrant` prints a note saying the ranking has no lexical half,
because a dense-only result and a hybrid one are indistinguishable on sight --
every hit just lacks a `t` marker, which reads as "no lexical match for this
query" rather than "this index cannot have one".

**Where the stack is configured is `embed/defaults.py`, read by both CLIs.**
`embed` writes vectors and `retrieve` reads them, and they must name the same
embedder or the ranking is well-formed and meaningless. Leaving that agreement
to whoever types the second command is the one configuration mistake here that
does not announce itself. Flag beats `FALCONVAR_INDEX` / `FALCONVAR_EMBEDDER` /
`FALCONVAR_EMBED_MODEL` beats the constants, and `db.load_env()` runs *before*
the parser is built so a `.env` is in effect when the defaults resolve.
Switching the whole stack to local is three lines in `.env`.

**`retrieve` embeds the query with the embedder that built the index.** Not a
default, an argument -- `--embedder` and `--model` on both CLIs have to agree.
A mismatch across *widths* fails loudly; a mismatch between two models of the
same width returns a well-formed ranking that means nothing, which is why the
embedder key is in the Qdrant collection name and in the pgvector primary key.
A wrong name then searches a collection that does not exist.

**`describe/` touches the manifest and the frame store, nothing else.** The
manifest arrives as parsed JSON and the chunk stream as rows, so neither needs
an import; the store is the single class it imports from `ingest`. Anything
more would mean depending on how ingest works rather than on what it produced.
`imports.py` enforces this by AST-parsing every file under `describe/` and
rejecting any `ver2.video.ingest` import other than `FrameStore`.

**Describing reads the frame store and nothing else.** No seek-the-video
fallback: the store exists so this stage has its frames in hand, and a
fallback would quietly do the store's job while leaving it broken — silently,
since the output is identical and only ~40x slower (0.12 s against 4.97 s on
test1). A missing store, or a frame the store lacks, raises `StoreUnavailable`
and names the fix (`recovery.recreate`). A short frame list is never returned
either: a description covering 8 of the 9 frames it claims is
indistinguishable from a correct one once written down.

**A following describer holds the newest chunk back by one.** A chunk's
`end_ts` is provisional while it is the last one: the pipeline corrects the
final chunk to where the video actually ends (60.333 s, not the grid's 80.0)
and only restates the rows at `finish`. Measured — a follower without the
lookahead described chunk 3 as covering 60-80 s, handing the model a window
twenty seconds longer than the footage. A chunk is released only once a later
one proves it is not the last, and the true last is re-read after ingest
completes. Descriptions lag ingestion by one chunk; the alternative is telling
the model something false about every video.

**A transcript chunk embeds through the same stage as a description, and that
was the test of the split.** Both are text with a time span and some bound
structure, so audio needed no index of its own, no table of its own and no
code past `units.from_transcript` -- it lands in `chunk_embeddings` with
`sampler = "transcript"`, and `--sampler transcript` narrows a search to what
was said exactly as `--sampler yolo` narrows it to who was seen. Had it needed
more, the seam between `embed` and the modalities would have been in the wrong
place.

Two things it does *not* carry over. Silent chunks are skipped: they stay in
the transcript document because `chunk_id` is shared with the video side, but
a vector for the empty string answers every query equally badly. And only
`speakers` goes into `structured`, never `turns` -- for a description the
structured half is worth embedding because it holds what the summary does not
(0.705 against 0.528 MRR), but `turns[].text` *is* the transcript, so rendering
it appended the whole chunk a second time interleaved with timestamps read as
numbers. 337 characters became 151. The turns are the record and live in
`audio_chunks`.

**Cross-modal agreement is the point, and it shows.** Measured on Chernobyl
with `uniform:overview` on the picture and Whisper on the sound, one `vad`
grid: `"the moment the reactor exploded"` returns chunk 7 at 0.1326 on
`overview(v1) + transcript(v2)` -- the visual pass describing "a severe
explosion and fire in one section of the reactor building" and the audio pass
"At 1.23 a.m., reactor 4 exploded", two independent accounts of the same 16
seconds. Asking the audio for `"the control rods were removed during the
test"` gives chunk 6, and that chunk's overview independently reports "the
control rods appear to move upward out of the core ... highlighted in a
stronger red tone, suggesting increased heat". Neither pass saw the other's
output.

**A sink that appends has to clean up after a shrink.** The pipeline streams
each chunk as it closes, then folds a too-short final chunk into the one
before it once it knows where the media really ends -- so the last row written
can name a chunk that no longer exists. Observed on test2: 60.325 s at 20 s
closed four chunks, the last 0.325 s long, and the merge left `chunk_id = 3`
orphaned in Postgres while the file held three. The file writer rewrites its
whole document and never had the problem. `finish()` now deletes
`chunk_id >= len(chunks)` after the upserts -- after, so a failure leaves the
previous copy whole rather than a hole. The transcript sink already did this;
the manifest sink predated the tail merge.

**One vector space per embedder, never per sampler.** A space is defined by
the model, not by which prompt produced the text, so everything one embedder
writes is directly comparable. The sampler is payload, and querying one is a
*filter*: `--sampler yolo`. Separate collections per sampler would partition
the same space while making cross-sampler queries impossible. Different
embedders do get separate collections — the name carries `name:model:dims`,
which is what stops 768-wide vectors being ranked against 1536-wide ones.

**Filtering to one sampler gives up the agreement signal.** Measured on
"a child near the shelves": unfiltered, chunk 2 wins at 0.0313 with clip and
yolo both ranking; with `--sampler yolo` a chunk can contribute at most one
description, scores halve to ~0.0164 (a single `1/(k+1)`), and the winner
changes to chunk 1. Neither is wrong — but only unfiltered search lets two
independent accounts of the same window vote together.

**A chunk scores as its best description plus a discounted second, never a
sum.** Summing `1/(k+rank)` over every description a chunk contributed is RRF
applied to the wrong problem: RRF fuses several rankings of the *same* items,
where the term count is constant, while a chunk contributes one term per
sampler that described it. At k=60 over ~20 candidates `1/(k+rank)` spans only
1.31x, so count overwhelmed rank. Found by indexing two videos with different
sampler counts -- test1 on clip+yolo+objects, Chernobyl on text alone -- and
asking for `"RBMK-1000 nuclear reactor diagram"`: all eleven Chernobyl
descriptions ranked 1-11 and every test1 description 12-20, yet a test1 chunk
whose best description ranked *13th* won on three mediocre terms against one
excellent one. The shipped ranking got the **video** right 57.7% of the time.

Now `score = 1/(k+best) + 0.5/(k+second)` at k=10 -- two terms at most,
whatever the sampler count. Measured over 52 query pairs on both videos:

| aggregation | video ok | literal MRR | paraphrase MRR |
|---|---|---|---|
| sum, k=60 (was shipped) | 0.577 | 0.421 | 0.341 |
| sum, k=5 | 1.000 | 0.671 | 0.451 |
| max, k=60 | 1.000 | 0.668 | 0.442 |
| **max + 0.5*second, k=10** | **1.000** | **0.682** | 0.446 |

`sum k=5` scores as well but is not chosen: it still sums, so it only survives
because chunks here have at most three descriptions. The bounded form cannot
regress however many samplers run. It keeps the agreement property on purpose
-- two accounts at ranks 2 and 3 still beat one at rank 1, which is why the
per-sampler split exists -- but a third and fourth account add nothing.

**This is invisible on a single video with a uniform sampler set**, where
every chunk contributes the same number of terms and the bias cancels. It
appears the moment an index holds more than one video, which is the normal
case. `eval/aggregation.py` reproduces it.

**Retrieval is RRF twice, never a weighted score.** Vector rank and text rank
are fused per description, then descriptions are fused into chunks. Cosine
distance and `ts_rank_cd` have no common scale, and any weight between them
would be invented; RRF reads only the orderings, so it needs no calibration.

**Descriptions never cascade from a manifest.** The `video_descriptions` table has
no foreign key to `video_manifests` on purpose: ingest replaces a manifest wholesale
(`begin()` deletes the row, `video_chunks` cascades), and re-ingesting costs 20
seconds where describing costs inference. Instead each row carries
`manifest_fingerprint` — a hash of `source` minus `uri` and `config` minus
`frame_store`, so it is known before the first chunk exists, survives the video
and store being moved, and changes when the sampling changes. Staleness is a
comparison, not a deletion.

**The site is served by the API, at `/app` rather than `/`.** Same origin, so
the browser client needs no CORS and no base URL -- but not the root, because
`GET /videos` is an API route and a site mounted there would shadow it. The
form is built from `GET /capabilities`, so a sampler registered in `ver2`
appears in the browser without anyone editing JavaScript.

**Two bugs in that page were only findable in a real browser**, and both were
invisible to reading the file. `#lightbox` carried the `hidden` attribute but
also `display: grid`, and any author `display` beats the UA stylesheet's
`[hidden] { display: none }` -- so a semi-transparent overlay covered the
viewport, dimmed the whole page and swallowed every click. And `queue()`
switched tabs by *clicking* the tab button, which fires an unawaited
`loadJobs()` whose `innerHTML = ""` detached the live progress box that
`follow()` had just inserted; the box kept polling and updating a node no
longer in the document, so a finished job read "queued" forever. Both are the
same shape as the pipeline bugs recorded above: the output looked plausible
and nothing reported the fault.

**Aggregates answer what retrieval cannot.** Embeddings cannot count, so "the
busiest moment", "who dominated", "how much of this is speech" are exact
questions similarity answers approximately. `aggregate/` reads the finished
documents -- never the video, never another stage's modules -- and produces
video-level structure. Measured on Chernobyl: 90.3% speech ratio, 125.1 words
per minute, `monologue: true` with 0 handovers, and 4 chapters that tile the
whole video with the explosion correctly at 94.4 s.

**`depends_on` drops rather than fails.** A dependency naming a source
(`yolo`, `transcript`) is a requirement on the video; one naming another
aggregator orders it first. `speakers` on silent CCTV is not an error, it is a
question that does not apply, and it is reported as skipped *with the reason*
-- "needs transcript, which this video has no output for" -- because "speakers
did not run" is only useful beside why.

**A tier is a cost ceiling, and asking for a dear one still runs the cheap
ones.** `free` is arithmetic, `local` adds GPU models, `llm` adds paid calls;
they run cheapest first, so a run that dies partway has produced the free
results rather than none.

**Staleness is `inputs_fingerprint`, a hash of the chunk text actually read.**
A summary of descriptions that have since been rewritten reads perfectly, which
is precisely why it cannot be left to a reader to assume. A re-run reports
"current" and costs nothing; `--force` rebuilds. The same discipline as
`manifest_fingerprint` and `text_hash`, applied to the most expensive stage.

**Only the summary is embedded, and into its own table.** `chunk_embeddings`
answers *which twenty seconds*; a summary answers *which video*, and a video is
not a moment you can play. Putting summaries in the chunk table would need a
sentinel `chunk_id` and would then return a whole-video "moment" beside real
ones in every search. Everything else `aggregate` produces is statistical, and
a vector of a count answers nothing.

**Installing `gliner` downgraded transformers 5.15.1 -> 5.13.1.** numpy, cv2
and torch were untouched and `ver2.imports` stayed green, but the checker only
proves the import works -- CLIP was verified separately by actually embedding:
512 dims, L2 norm 1.0. That check matters because `transformers` is what the
`clip` sampler and the local embedder both run on.

**The API calls `orchestrate`, never `driver`.** `ver2/driver.py` used to hold
the whole run inline, which made it the one module breaking the rule every
stage driver follows -- "driver.py: the CLI, argparse only, no pipeline logic".
A server importing an argparse module to reach the work behind it would have
made that permanent, so the run moved to `ver2/orchestrate.py` and the CLI
became a shim over it. Verified: the CLI's output is unchanged.

Progress is a **callback**, not a print. The two callers want opposite things
from it -- a terminal wants lines as they happen, a job runner wants the latest
state to answer a poll with -- so `process(options, on_progress=...)` emits
`(stage, detail)` and neither policy lives in the pipeline.

**Slow work is queued, one job at a time.** Every heavy stage contends for the
same 8 GiB GPU: CLIP and YOLO during ingest, Whisper and pyannote during the
audio pass. Two videos at once does not halve the wall clock, it doubles the
resident weights and invites an allocator failure halfway through the more
expensive one. A queue of one is the honest shape of the hardware.

**Validation is synchronous even though the work is not.** `orchestrate.validate`
returns problems as a list rather than raising, so the CLI prints them all at
once and the API answers 422 with them -- and a bad sampler name never becomes
a job that fails a minute later. What cannot be known without opening the file
still fails in the job: `--chunking vad` on silent audio is a 202, because
whether a track has speech is not a property of the request.

**Either stream may be switched off, and the grid policy has to survive it.**
`use_video` and `use_audio` are independent, so `validate` rejects a policy
whose stream is not running: `scene` is found while decoding frames, `vad` and
`speaker` in the waveform, and asking for one without its stream is a
contradiction rather than a preference to honour differently. `uniform` is
arithmetic and works whatever is on -- audio-only, it is computed from the
decoded track duration, and `_cover` is skipped because there is no video
duration to reach. Whether a file *has* a soundtrack is not a property of the
request, so "audio only, no audio track" is caught in `process`, not
`validate`.

An audio-only run writes `timeline.json` and `transcript.json` and nothing
else. There is no describe stage to run -- a transcript is already the text
that stage would produce -- and `embed`, `retrieve` and `aggregate` work off it
unchanged. Two readers had to learn that a manifest is optional rather than
missing: `embed.driver` reads a named `transcript.json` directly, and
`db.fetch_manifest(required=False)` returns None instead of exiting, because
no manifest means no descriptions either.

**A form built from a registry defaults to whatever sorts first, and that was
`stub`.** The transcriber select offered `transcribe.available()` in order, so
the first option was the stub; an audio-only run through the browser completed
in 10.6 s, reported 42 segments and 205 words, and wrote a transcript of
`[stub0.0][stub0.1]`. Nothing was wrong enough to report. `/capabilities` now
publishes `defaults` read off `orchestrate.Options` -- transcriber, diarizer,
chunking, describer -- and the form selects those rather than position 0. The
default lives in one place, and it is the dataclass the pipeline actually uses.

**`[hidden]` needs `!important` once, not per rule.** `label.row` sets
`display: flex`, and any author `display` beats the UA stylesheet's
`[hidden] { display: none }` -- so hiding the frame-store row in audio-only
mode set the attribute and changed nothing on screen. Exactly the `#lightbox`
fault again, in a second place, three months later. One global
`[hidden] { display: none !important }` is the fix, because the next author
`display` will not remember either.

**The export surface lists what exists, not what could exist.**
`/videos/{id}/exports` is read from disk, so an audio-only video advertises no
manifest rather than offering a link that 404s -- a broken link reads as
breakage, not as a run that never had one. It is also the one place that knows
summaries live under `aggregates/` while descriptions live a level up, and the
browser's download links read it rather than assembling paths. `?download=1`
only adds a `Content-Disposition`: content negotiation would be tidier, but a
browser cannot set an `Accept` header on a plain link.

**Every summary layer is recorded; only the topmost is embedded.** `summary`
folds chunk lines 25 at a time into leaf summaries, folds those again while
more than 25 remain, then makes one final structured call. The intermediate
summaries used to be transient, which threw away a coarse account of the video
that had already been paid for -- a leaf summary covers a real span and is the
only description at that granularity, between one chunk and the whole file.
They are kept in `layers`, each part carrying `chunk_ids`, `start_ts` and
`end_ts`.

The ids travel *with* the text through every fold, so a merge summary's span is
the union of what it merged and nothing is reconstructed afterwards by parsing
a string this module formatted itself. (`llm.chunk_rows` returns `(chunk_id,
line)` for exactly this; `chunk_lines` is now a wrapper over it.) The span is
then resolved through the timeline rather than carried as numbers, the same
reason `chapters` resolves its spans instead of trusting the model's. Verified
on Chernobyl at `batch=3`: 14 chunks -> 5 leaves -> 2 merges, both levels
tiling 0.0-205.28 s contiguously with no gaps.

`reduction_levels` records how many folds happened, which is how much
paraphrase sits between the descriptions and the final text. At the shipped
batch of 25 a 14-chunk video takes the single-call path and `layers` is `[]` --
which says truthfully that no intermediate summary existed, rather than that
one was discarded.

**Storing them and embedding them are different decisions.** The layers stay
out of the index because they paraphrase material the chunk vectors already
hold: indexing them would return the same moment two or three times over under
different wordings, which is the count-bias failure the moment aggregation
guards against, reintroduced one level up. Everything else `aggregate` produces
is statistical and a vector of a count answers nothing; `chapters` carries a
summary per chapter and those are a table of contents, read in full rather than
searched.

**Job records die with the process; artifacts do not.** `GET /jobs` says so.
`GET /videos` reads the `out/` directory rather than remembering, so a
restarted server still knows everything it produced.

**One chunk grid per run, and either modality may decide it.** `timeline.py`
holds the spans and the policy that produced them, and imports nothing --
`audio` and `video` both depend on it, so neither may own it.

    uniform   arithmetic. Both sides derive it identically; nothing propagates
              and neither has to run first.
    scene     the video pass, streaming, from frame content.
    vad       the audio pass, in the gaps between speech.
    speaker   the audio pass, where the voice changes.

Propagation is one-way per run and always cheap in the same direction: a
finished transcript can be re-cut to any grid because Whisper timestamps every
word, while a sampler's decisions are made during the decode pass and cannot
be revisited. So an audio-derived policy forces audio to run first; a
video-derived one forces nothing, because transcription needs no boundaries at
all and can run concurrently. `FixedChunker` is how a grid decided elsewhere
reaches the video pipeline, and it is deliberately the dumbest chunker there
-- every judgement was made before the span list existed.

**A supplied grid is honoured exactly; the pipeline stops correcting it.** The
final-chunk correction and the tail merge both run only when the chunker
computed its own boundaries. On a `fixed` run they would silently edit a grid
another pass decided: the audio stream is routinely shorter than the video one
-- 205.264 s against 205.280 s on Chernobyl -- so "correcting" the last chunk
to the video's end made `manifest != timeline` by 16 ms, on a boundary neither
pass had chosen. And a supplied grid has already had its own `min_s` applied,
so re-applying a different one would merge a tail the other pass kept
deliberately.

The other half of that fix is in the driver: **the shared grid spans the
longer stream**, because a file is as long as its longest one. A grid built
from the audio duration alone leaves the last video frames outside every
chunk. Verified across all four policies -- `manifest`, `timeline` and
`transcript` hold identical spans, and the transcript's recorded
`timeline_fingerprint` matches, on `uniform`, `scene`, `vad` and `speaker`
alike.

**A short final chunk is merged into the one before it, under every policy.**
A grid divides the media wherever it happens to end, so the tail is uniformly
distributed over the chunk length: 97.99 s at 20 s leaves a usable 17.99 s,
but 100.4 s leaves 0.40 s. That stub is not harmless -- it costs a describer
call *per sampler*, it keeps a frame because every chunk keeps at least one,
and it is a moment retrieval can return that nobody can play. The threshold is
a fraction (`MIN_TAIL_FRACTION = 0.25`) rather than a number of seconds, so it
holds at any chunk length, and it is applied by `timeline.uniform`,
`timeline.enforce` and the pipeline alike so that "no chunk shorter than N"
means the same thing however the boundaries were derived.

The pipeline merges *after* the pass, at the point it already corrects the
final chunk's `end_ts`, so the samplers had reset at the boundary being
removed and the merged chunk carries one extra mandatory first-frame pick.
That is the honest cost of deciding late, and an extra frame is not a defect.
**`chunk_local_index` is shifted, not renumbered** -- it is the frame's
position in the chunk's decimated stream, not a sequence number over the picks
(a uniform sampler at stride 3 yields 0, 3, 6, 9), so renumbering the tail's
picks 0..n silently redefines the field. Verified: at `--chunk-duration 24` on
test1 the merged chunk reads `[0, 3, 6, ..., 24]` with every index below its
26 decimated frames, and at the shipped 20 s nothing merges -- chunk bounds
and every sampler's frame list stay identical to the manifest the current
descriptions and vectors were built from.

**Content-derived boundaries need both guards or they are unusable.**
`timeline.enforce` merges anything below `min_s` and splits anything above
`max_s`. Voice activity cuts on every pause -- on conversational audio that is
every second or two, which would shred the video into chunks too short to
describe and multiply the describer calls that dominate cost. A monologue
gives the opposite failure: zero cuts, one chunk covering the file. Measured
on Chernobyl, `--chunking speaker` on single-narrator audio found no speaker
changes and `enforce` divided the file into 7 even chunks of 29.3 s, which is
the honest outcome rather than an invented one. **`max_s` splits evenly, not
into fixed bites**: taking 30 s bites off a 62.5 s span leaves a 2.5 s
remainder, so the guard against short chunks would create one.

**Chunks with no speech are kept, with empty text.** The grid is shared, so
`chunk_id` must mean the same thing in the manifest and the transcript;
dropping the quiet ones renumbers everything after them and breaks that
correspondence silently. Measured on Chernobyl: 10 chunks on a `vad` grid, the
first empty because narration starts at 5.9 s, and all 428 words placed with
none lost.

**Audio is scanned whole; it cannot be chunked first.** Whisper carries
context across an utterance and detects language from the opening seconds, so
a 20-second window loses the sentence that straddles its edge. Diarization is
worse: speaker labels come from clustering embeddings over the *entire*
recording, so `SPEAKER_00` in one window bears no relation to `SPEAKER_00` in
the next -- chunk first and the speakers are not misaligned, they are
unnameable. Boundaries are applied *after*, and re-segmenting costs nothing
because Whisper returns a timestamp per word. That asymmetry is what lets
either modality own the chunk grid: audio can always conform to a boundary
decided elsewhere without re-running inference or losing a word.

**Transcription and diarization stay separate passes, joined by `align`.**
Whisper does not know who spoke and pyannote does not know what was said.
Keeping them apart is what lets either be swapped. A word is attributed by its
**midpoint**, because the two models estimate edges independently and word
spans routinely straddle a turn boundary by tens of milliseconds; a segment
takes the speaker who spoke most of it by duration, because a sentence can
begin during the previous speaker's tail. With no diarizer, or no speech,
every segment keeps `speaker=None` -- a transcript with no speaker information
is truthful, one where everything is `SPEAKER_00` is not.

**Measured, RTX 4060:** decode 205 s of AAC in 0.30 s (~700x realtime),
Whisper `small` float16 37.6x, pyannote 3.1 30.1x. A one-hour recording is
about four minutes of work, against one describer call per chunk on the video
side. Chernobyl: 34 segments, 428 word timestamps, 1 speaker over 18 turns
covering 89% of duration, all 34 segments attributed.

**Silence is answered without a model.** test1.mp4 is CCTV with a live but
empty microphone: RMS 0.000221, peak 0.0291, against 0.1796 for narration --
three orders of magnitude, so the 1e-3 threshold sits in a wide gap rather
than on a cliff. Whisper returns zero segments there and names the language
`cy` at p=0.41, and pyannote finds zero speakers in 0.1 s. Both are correct,
so the guard exists to skip a model load and to make "no speech" a reported
fact, not to make a fine judgement.

**`media_ts` is the only clock a decision may use.** Never wall time, never
frame counts.

**Decimation buckets on media time**, never "every Nth frame". Identical on a
clean file; self-correcting on a lossy one.

**Chunk boundaries derive from media time alone**, never from how many frames
arrived.

**Samplers reset at every chunk boundary** and every chunk keeps at least one
frame. Rate limits are enforced in the base class *before* the strategy runs,
so a rate-limited frame costs no inference.

**Pixels are borrowed.** `frame.release()` runs every iteration; anything that
outlives the loop must copy.

---

## Measured facts — do not re-derive

**Thresholds do not transfer between videos.** Same sampler, same domain
(retail CCTV), `clip 0.96`:

| video | keep rate | median 1-second similarity |
|---|---|---|
| test.mp4 | 13.4% | 0.9868 |
| test2.mp4 | 18.0% | 0.9895 |
| test1.mp4 | 59.2% | 0.9605 |

That spread is the sampler working — test1 genuinely changes ~3x more per
second. **If cost is too high use `min_interval_s` / `max_per_chunk`, not a
lower threshold**: they keep the most-changed frames and even out the
per-chunk yield, where lowering the threshold just keeps fewer and leaves the
distribution lopsided.

**Do not solve for a fixed keep rate.** Forcing e.g. 15% turns a change
sampler into a worse UniformSampler. On a frozen video, solving for 15%
produced threshold 1.000 which sampled *100%* of frames — encoder noise, since
one PNG looped for 60 s H.264-encodes to 60 *different* frames.

**`calibrate.py` reports, it does not choose.** It measures the noise floor
(the similarity when nothing happened) and prints what every threshold would
cost. Across all real footage the floor sits at >= 0.9998, so the check is a
guard against a rare pathology, not a routine input.

**Detection samplers shifted 12-15% when the reader moved OpenCV -> PyAV**
(yolo 40->45, objects 48->55; clip unchanged). Sub-LSB colour differences flip
detections near the confidence boundary. The defaults are still in the right
region but run slightly hot.

**EasyOCR is 98.1% of the text sampler's cost** (129.6 ms of 132.0). The
descriptor is 2.5 ms. `canvas_size` is the only real lever and it is **not**
free: at 736 it is 3x faster but covers only 70% of the ink 1280 finds, missing
whole lines of dense prose. Keep 1280 for prose; 736 may be fine for large
text (slides, signage) but that is a content decision.

**PaddleOCR and craft-text-detector were evaluated and rejected.** EasyOCR's
detector *is* CRAFT (`from .craft import CRAFT`), so the standalone package is
the same model in an unmaintained wrapper (4 incompatibilities with current
torch/numpy). PaddleOCR will not import here: `WinError 127` on
`cudnn_cnn64_9.dll` despite exactly-matching pinned versions.

**A difference hash is not usable for text change detection.** Measured on a
static slide: dHash scored 0.86 agreement against *itself* frame to frame,
while two *different* slides scored 0.88 — pure noise. dHash compares adjacent
pixels and most of a text region is flat background, so the sign it records
comes from sensor noise, not content. Averaging down with `INTER_AREA`,
centring and normalising (what `TextLayoutDescriptor` does) suppresses that
instead: 1.000 for a static slide, at most 0.80 between different ones.

**`objects` vocabulary is the highest-value setting**, and has no useful
default. Measured on shop CCTV: a mismatched list found 2.4 detections/frame
and labelled wire baskets "shopping bag"; a matched one found 5.1.

---

## Environment traps

**`cublas64_12.dll` is not found, on a machine where it is present.**
CTranslate2 (under faster-whisper) asks Windows for it *by name* at the first
encode, not at import. `nvidia-cublas-cu12` installs it to
`site-packages/nvidia/cublas/bin`, which is on no search path: since Python
3.8 an extension's dependencies do not resolve from `PATH`, and
`os.add_dll_directory` does not help because the load happens lazily inside an
already-initialised C++ extension. The version is a contract -- this project's
torch is cu130 and ships `cublas64_13.dll`, which is not a substitute.
`ver2/audio/cuda.py` loads each by absolute path with `ctypes.WinDLL` first,
which puts it in the process module table so the by-name request resolves.
Call `enable()` before constructing any CUDA-backed audio model.

**pyannote 4.x returns `DiarizeOutput`, not `Annotation`.** The 3.x recipe
`pipeline(audio).itertracks(yield_label=True)` raises `AttributeError`. The
annotation is `.speaker_diarization`; there is also
`.exclusive_speaker_diarization` with overlaps resolved (what this project
uses -- a word cannot belong to two speakers) and `.speaker_embeddings`, a
256-d vector per speaker, which is the only route to identity across files
since labels are per-run clusterings.

**`schema.sql` could not repair its own `fts` column, and now can.** `add
column if not exists fts` is a no-op when the column exists, so an `fts`
generated by an earlier version of the file — over `content` alone, before the
structured values were folded in — survived a re-run untouched. Nothing
reported it: the column was there, queries ran, and the lexical half simply
stopped indexing the terms it is best at. Found live on this project's
database, where `fts` existed and `structured` had never been added to either
table. The statement is now `drop column if exists fts` followed by an
unconditional add; the column is generated, so nothing is lost.

**PostgREST's cached schema is the fastest way to see what is really
deployed.** `GET /rest/v1/` with `Accept: application/openapi+json` lists every
column and RPC parameter it knows. That is how the stale generation was found —
`fts` present, `structured` absent on both tables, `export_video_descriptions`
returning no `structured`, `search_embeddings` with no `p_sampler`.

**`weights/clip/ViT-B-32.pt` (338 MB) is NOT stale.** YOLO-World embeds its
vocabulary with OpenAI CLIP, so `OpenVocabDetector` downloads it. That is a
*different* CLIP from the one `ClipChangeSampler` loads through HuggingFace —
different library, format and job. Deleting it costs a re-download.

**opencv variants shadow each other.** `opencv-python`,
`opencv-contrib-python` and `opencv-python-headless` all install `cv2`;
whichever wins depends on install order and nothing warns you. Uninstalling one
**breaks the others** (they share the directory) — repair with
`pip install --force-reinstall --no-deps opencv-python==5.0.0.93`.

**Run `python -m ver2.imports` after any install.** Adding PaddleOCR silently
downgraded numpy 2.4.4 -> 2.3.5 and swapped `cv2` 5.0.0 -> 4.10.0.

**`PYTHONIOENCODING=utf-8`** is needed for some third-party libraries that
print non-ASCII on Windows' cp1252 console.

**`stream.thread_type = "AUTO"`** is mandatory in PyAV, not an optimisation:
7.15 ms/frame without it against 3.97 with.

---

## Current state

Working and verified: both chunkers, all five samplers, manifest, frame store,
both recreate paths, 13 structural manifest checks, 13 container/codec formats,
4 refusal paths. The describe stage runs end to end against `gpt-5.4-mini`:
10/10 pairs on test1, file and Postgres identical, and the document rebuilt
from Postgres by `recovery.supabase_description` identical to the local one.

`embed` and `retrieve` are separate stages. Both indexes are live: 10 units of
test1 in Qdrant and in `chunk_embeddings`, identical `text_hash` values
in each, and `search_embeddings` fusing vector and text ranks with
`p_sampler` filtering. `--sampler clip` halves a moment's score to a single
`1/(k+1)` exactly as predicted — one description per chunk, nothing to fuse.

`recreate.py` refuses a mismatched video (compares fps, time_base, dimensions,
frame count) — `--force` overrides.

## Not built

- **A local embedder, run.** `LocalEmbedder` is written and guarded but no
  model has been downloaded; every measurement so far is OpenAI.
- **Filterable structured values.** `structured` reaches both indexes and both
  can filter on it, but the values are free text, so a filter for `cashier`
  matches everything. Needs an `enum` on the fields meant to be filtered.
- **Live sources.** `Frame.gap_before` and `Frame.discontinuity` are the seams,
  always `0`/`False` for a file.
- **Tests.** There is no test suite; verification is `imports.py`, recreate's
  byte-comparison, and the retrieval measurements recorded above.
