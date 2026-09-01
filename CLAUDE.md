# FalCONvar — working notes

Video RAG ingestion. A video goes in; a **manifest** comes out saying which
frames are worth describing, grouped into retrievable chunks, with enough
addressing information to fetch those frames back later.

Everything under `ver2/`. Version 1 has been deleted.

---

## Layout

```
ver2/
  ingest/
    source/      probe, sequential read, decimation, random access (PyAV)
    chunker/     media time -> chunk id            (uniform | scene)
    samplers/    which decimated frames to keep    (uniform|clip|yolo|objects|text)
                 policy: base.py, uniform.py, scene.py, detection.py (the base
                 for detector-driven ones) + people.py, objects.py, ocr.py
      components/  perception: detectors, descriptors, embedders. Every model
                   weight lives below this line and none above it.
    output/      manifest sinks (file | supabase | both) + frame store
    pipeline.py  ingest() -- one decode pass, feeding every stage
    driver.py    the CLI (argparse only; no pipeline logic)
    calibrate.py what a threshold would cost here, and where it must not go
  describe/
    input/       manifest (file|db), live chunk stream, pixels from the store
    describers/  the Describer protocol + a registry (stub | openai)
    vlm/         the OpenAI call + prompts.py, one prompt per sampler
    output/      description sinks (file | supabase | both)
    reader.py    describe() -- one call per (chunk, sampler)
    driver.py    the CLI
  embed/
    defaults.py  which embedder + which index, shared by both CLIs
    embedders/   Embedder protocol + registry (openai | local)
    index/       VectorIndex: qdrant (embedded) | pgvector | both, + build()
    units.py     description -> embeddable unit, keyed by a hash of its text
    indexer.py   embed only what changed
    driver.py    the CLI
  retrieve/
    search.py    query -> ranked descriptions -> ranked moments
    driver.py    the CLI
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

8212 lines, 75 files.

## Commands

```bash
python -m ver2.ingest.driver media/test2.mp4 --sampler clip --frame-store
python -m ver2.ingest.driver video.mp4 --sampler uniform,clip,yolo,objects,text \
       --min-interval 3 --chunking scene --scene-threshold 15
python -m ver2.ingest.driver video.mp4 --sampler objects --vocabulary "crate,pallet"
python -m ver2.ingest.driver video.mp4 --sink file,supabase   # both; file is primary
python -m ver2.ingest.calibrate video.mp4 --sampler clip
python -m ver2.describe.driver out/<id>/manifest.json     # describe -> json
python -m ver2.describe.driver --video-id <id> --sink file,supabase
python -m ver2.describe.driver --video-id <id> --follow    # tail a live ingest
python -m ver2.describe.driver out/<id>/manifest.json --describer openai
       --model gpt-5.4-mini --sink file,supabase          # costs money
python -m ver2.embed.driver out/<id>/descriptions.json     # -> pgvector
python -m ver2.embed.driver --video-id <id> --index pgvector,qdrant
python -m ver2.retrieve.driver "people at the checkout" --moments 3
python -m ver2.retrieve.driver "..." --sampler yolo        # one question only
python -m ver2.retrieve.driver "..." --index qdrant        # dense only, says so
python -m ver2.recovery.supabase_manifest --list          # what is published
python -m ver2.recovery.supabase_manifest <id>            # -> <id>.json
python -m ver2.recovery.supabase_description <id>         # -> descriptions json
python -m ver2.recovery.recreate out/<id>/manifest.json --out rebuilt/
python -m ver2.imports                      # after ANY install
python -m eval.queries out/<id>/descriptions.json   # -> eval/query_pairs.json
python -m eval.render_ab                    # compare renderings, paired CI
python -m eval.retrieval                    # hybrid vs dense on the shipped path
python -m eval.aggregation                  # chunk scoring, count-bias check
# schema.sql   runnable DDL: every table, function and RLS policy
# SCHEMAS.md   what every field means, local JSON and Postgres alike
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
`search_descriptions` skips its `by_text` CTE and fuses nothing. 29 query
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
rejecting any `ver2.ingest` import other than `FrameStore`.

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

**Descriptions never cascade from a manifest.** The `descriptions` table has
no foreign key to `videos` on purpose: ingest replaces a manifest wholesale
(`begin()` deletes the row, `chunks` cascades), and re-ingesting costs 20
seconds where describing costs inference. Instead each row carries
`manifest_fingerprint` — a hash of `source` minus `uri` and `config` minus
`frame_store`, so it is known before the first chunk exists, survives the video
and store being moved, and changes when the sampling changes. Staleness is a
comparison, not a deletion.

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
`fts` present, `structured` absent on both tables, `export_descriptions`
returning no `structured`, `search_descriptions` with no `p_sampler`.

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
test1 in Qdrant and in `description_embeddings`, identical `text_hash` values
in each, and `search_descriptions` fusing vector and text ranks with
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
