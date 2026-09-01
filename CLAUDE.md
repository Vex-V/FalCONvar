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
    output/      manifest writer + frame store
    pipeline.py  ingest() -- one decode pass, feeding every stage
    driver.py    the CLI (argparse only; no pipeline logic)
    calibrate.py what a threshold would cost here, and where it must not go
  recovery/
    recreate.py  STANDALONE: rebuild a store from a manifest + the video
  imports.py     import everything, exercise it, report what loaded
```

3924 lines, 34 tracked files.

## Commands

```bash
python -m ver2.ingest.driver media/test2.mp4 --sampler clip --frame-store
python -m ver2.ingest.driver video.mp4 --sampler uniform,clip,yolo,objects,text \
       --min-interval 3 --chunking scene --scene-threshold 15
python -m ver2.ingest.driver video.mp4 --sampler objects --vocabulary "crate,pallet"
python -m ver2.ingest.calibrate video.mp4 --sampler clip
python -m ver2.recovery.recreate out/manifests/<id>.json --out rebuilt/
python -m ver2.imports                      # after ANY install
```

Defaults: manifest to `out/manifests/<video-id>.json`, store to
`out/stores/<video-id>/`. Bare `--frame-store` uses the default path.

---

## Invariants — do not break these

**`recovery/recreate.py` imports nothing from `ver2`.** Hand someone that one
file, a manifest and the video and they rebuild the store byte for byte with
only `av`, `opencv-python`, `numpy`. If it imported the pipeline it could lean
on a default living in code rather than in the manifest, and the manifest's
claim to be authoritative would go untested. `imports.py` enforces this by
AST-parsing the file; a `from ver2...` there fails the check.

**Recreate is the end-to-end oracle.** Any change to encoding, addressing or
the manifest is verified by rebuilding a store and byte-comparing. Verified
repeatedly at 24/24, 25/25, 61/61, 93/93, 350/350. If a change makes recreate
non-identical, the change is wrong.

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

**`objects` vocabulary is the highest-value setting**, and has no useful
default. Measured on shop CCTV: a mismatched list found 2.4 detections/frame
and labelled wire baskets "shopping bag"; a matched one found 5.1.

---

## Environment traps

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
4 refusal paths.

`recreate.py` refuses a mismatched video (compares fps, time_base, dimensions,
frame count) — `--force` overrides.

## Not built

- **The VLM / describer stage.** The manifest is the handoff; nothing consumes
  it yet.
- **Live sources.** `Frame.gap_before` and `Frame.discontinuity` are the seams,
  always `0`/`False` for a file.
- **Storage backend.** Manifests are files. See `SUPABASE.md` for the plan.
- **Tests.** There is no test suite; verification is `imports.py` plus
  recreate's byte-comparison.
