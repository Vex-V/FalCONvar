# Running it

Three ways in, all over the same code: a web page, an HTTP API, and a set of
CLIs. Nothing is a wrapper around anything else — the CLI and the API both call
`ver2/orchestrate.py`, and the web page calls the API.

---

## Once, before anything

```bash
pip install -r requirements.txt
python -m ver2.imports          # after ANY install, including this one
```

`imports.py` imports every library and internal module, exercises each enough to
prove it loaded a working binary, and flags packages that shadow the same import
name. It is not a formality: installing PaddleOCR once silently downgraded numpy
2.4.4 → 2.3.5 and put an opencv-contrib 4.10 alongside the opencv-python 5.0
already present, so `cv2.__version__` changed under the pipeline without a line
of it being touched.

Copy `.env.example` to `.env` and fill in what you need. Nothing is required to
run the stub paths:

```
OPENAI_API=             # describe (gpt-5.4-mini) and the openai embedder
HF_TOKEN=               # pyannote is gated; accept its terms on the hub too
SUPABASE_URL=
SUPABASE_SECRET_KEY=          # sb_secret_...      writes, bypasses RLS
SUPABASE_PUBLISHABLE_KEY=     # sb_publishable_... reads, under RLS
```

For the Postgres index, run [`schema.sql`](../schema.sql) in the Supabase SQL
editor. It is idempotent and safe to re-run.

---

## The web app

```bash
python -m uvicorn api.main:app --port 8000
```

Then open **<http://127.0.0.1:8000/>** — it redirects to `/app/`.

The page is served by the same process as the API, so there is no second server,
no CORS and no base URL to configure. Four tabs:

- **Search** — a question in, moments back, with each account of the window and
  the frames that prove it. Click a thumbnail to enlarge.
- **Process a video** — pick a file, tick **the picture** and/or **the
  soundtrack**, and only the options that choice reads are shown. A grid policy
  whose stream is switched off is dimmed rather than hidden, so *why* it is
  unavailable stays on screen. Watch the job progress live.
- **Digest** — the video-level aggregates, by tier.
- **Library** — what has been processed, per-video Describe / Embed buttons,
  and a download link for every document the video produced (plus **all ↓** for
  the lot in one file).
- **Jobs** — this session's runs, with each stage and any traceback.

Everything the form offers comes from `GET /capabilities`, so registering a
sampler in `ver2` makes it appear here with no JavaScript edited.

For development, `--reload` picks up changes to `api/` and `web/`:

```bash
python -m uvicorn api.main:app --port 8000 --reload
```

---

## The API alone

Same command; the schema is at **<http://127.0.0.1:8000/docs>**. See
[ROUTES.md](ROUTES.md) for what each endpoint does and why.

```bash
# process a file
curl -s -X POST localhost:8000/videos \
     -F file=@media/Chernobyl.mp4 \
     -F samplers=uniform:overview -F chunking=vad -F every_frames=5
# -> {"job": {"id": "..."}, "video_id": "Chernobyl", "poll": "/jobs/..."}

curl -s localhost:8000/jobs/<id>                       # poll until state=done
curl -s -X POST localhost:8000/describe -H 'content-type: application/json' \
     -d '{"video_id":"Chernobyl","sinks":["file","supabase"]}'
curl -s -X POST localhost:8000/embed -H 'content-type: application/json' \
     -d '{"video_id":"Chernobyl"}'
curl -s -X POST localhost:8000/search -H 'content-type: application/json' \
     -d '{"query":"the moment the reactor exploded","moments":3}'
```

Upload, describe and embed answer **202** with a job id — they are minutes of
GPU or paid inference. Search answers in the request.

---

## The command line

### Both streams at once

```bash
python -m ver2.driver media/x.mp4 --sampler clip --chunking uniform
python -m ver2.driver media/x.mp4 --sampler uniform:overview --every-frames 5 \
       --chunking vad --frame-store --sink file,supabase
```

`--chunking` decides which modality owns the chunk boundaries: `uniform` is
arithmetic and needs neither pass first; `scene` is the video pass; `vad` and
`speaker` are the audio pass, and so make audio run first.

Either stream can be switched off:

```bash
python -m ver2.driver media/x.mp4 --no-video --chunking vad     # sound alone
python -m ver2.driver media/x.mp4 --no-audio --chunking scene   # picture alone
```

`--no-video` writes `timeline.json` and `transcript.json` and no manifest, so
there is nothing for `describe` to do — a transcript is already the text that
stage would produce. `embed`, `retrieve` and `aggregate` all work off it
unchanged. The policy still has to be derivable from a stream that is running:
`--no-video --chunking scene` and `--no-audio --chunking vad` are both refused
before anything is decoded.

### One stage at a time

```bash
# video: which frames are worth describing, and why
python -m ver2.video.ingest.driver media/test1.mp4 --sampler clip --frame-store
python -m ver2.video.ingest.driver v.mp4 --sampler objects --vocabulary "crate,pallet"
python -m ver2.video.ingest.driver v.mp4 --sampler uniform:text --every-frames 10
python -m ver2.video.ingest.calibrate v.mp4 --sampler clip   # what a threshold costs

# audio alone: transcript + speakers, no describe stage
python -m ver2.audio.driver media/x.mp4 --chunking speaker

# describe: one call per (chunk, sampler)
python -m ver2.video.describe.driver out/<id>/manifest.json              # stub, free
python -m ver2.video.describe.driver out/<id>/manifest.json --describer openai
python -m ver2.video.describe.driver --video-id <id> --follow            # tail a live ingest

# embed, then ask
python -m ver2.embed.driver out/<id>/descriptions.json      # picks up transcript.json too
python -m ver2.embed.driver out/<id>/transcript.json        # audio-only: no descriptions exist
python -m ver2.retrieve.driver "people at the checkout" --moments 3
python -m ver2.retrieve.driver "..." --sampler transcript   # only what was said
```

### Handing the output to something else

```bash
curl localhost:8000/videos/<id>/exports              # what exists, with URLs
curl localhost:8000/videos/<id>/export > <id>.json   # all of it, one document
curl localhost:8000/videos/<id>/aggregates/summary   # just the summary
curl localhost:8000/videos/<id>/transcript           # just the transcript
```

Or read `out/<id>/` directly — the API serves the same files unchanged.

### Recovery — three files, no checkout needed

```bash
python -m ver2.recovery.supabase_manifest --list
python -m ver2.recovery.supabase_manifest <id>          # -> <id>.json
python -m ver2.recovery.supabase_description <id>
python -m ver2.recovery.recreate <id>.json --out rebuilt/ --verify out/<id>/store
```

`recovery/` imports nothing from `ver2`. Hand someone those files, a video id
and the video, and they rebuild the frame store byte for byte.

---

## Where the output goes

```
out/<video-id>/
  timeline.json      the chunk grid + the policy that produced it
  manifest.json      which frames were kept, and why
  store/             those frames, keyed by source frame index
  descriptions.json  what a VLM said about them
  transcript.json    what was said, and by whom
uploads/             what the API was posted; not cleaned up automatically
out/qdrant/          the local vector index, when --index qdrant
```

One directory per video, so a video's whole output is one thing to inspect, copy
or delete.

---

## Choosing the stack

`ver2/embed/defaults.py` decides which embedder and which index, and **both the
embed and retrieve CLIs read it** — they must name the same embedder or the
ranking is well-formed and meaningless. Flag beats environment beats the
constants:

```
FALCONVAR_INDEX=qdrant
FALCONVAR_EMBEDDER=local
FALCONVAR_EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5
```

Postgres is the default because it is the only index carrying the lexical half
of the hybrid; `--index qdrant` is dense-only and says so on every search.

---

## When it will not start

**`cublas64_12.dll` is not found**, on a machine where CUDA plainly works.
CTranslate2 (under faster-whisper) asks Windows for it *by name* at the first
encode, not at import, and the wheels put it somewhere on no search path.
`ver2/audio/cuda.py` preloads it; `python -m ver2.imports` reports whether that
succeeded rather than only that the import worked.

**pyannote refuses to download.** The model is gated: set `HF_TOKEN` *and*
accept the terms for `pyannote/speaker-diarization-3.1` on huggingface.co with
the account that token belongs to.

**Search returns nothing.** Usually the embedder does not match the one the
index was built with. That is caught rather than silently ranked, because a
mismatch across widths fails loudly while a mismatch between two models of the
same width would return a plausible ranking that means nothing.

**`PYTHONIOENCODING=utf-8`** is needed for some third-party libraries that print
non-ASCII on Windows' cp1252 console.
