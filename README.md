# FalCONvar — video RAG ingestion

Turns a video into a manifest: which frames are worth describing, grouped into
retrievable chunks, with enough addressing information to fetch those frames
back later. The describer (VLM) stage reads the manifest and never touches the
pipeline.

```
ver2/ingest/
  source/      probe, sequential read, decimation, random access
  chunker/     media time -> chunk id
  samplers/    which decimated frames are worth describing
  driver.py    wires it together and writes the manifest
  calibrate.py what a threshold would cost here, and where it must not go
  recreate.py  rebuild a frame store from its manifest
```

## Running

```bash
python -m ver2.ingest.driver video.mp4 -o out/manifest.json
python -m ver2.ingest.driver video.mp4 --sampler clip --chunk-duration 20
python -m ver2.ingest.driver video.mp4 --sampler yolo --min-interval 3
python -m ver2.ingest.driver video.mp4 --sampler objects --vocabulary "crate,pallet,forklift"
python -m ver2.ingest.driver video.mp4 --sampler text --chunking scene --scene-threshold 15
python -m ver2.ingest.driver video.mp4 --sampler uniform,clip,yolo --frame-store out/store

python -m ver2.ingest.calibrate video.mp4 --sampler clip
python -m ver2.ingest.recreate out/manifest.json --out out/rebuilt
```

Samplers: `uniform`, `clip`, `yolo`, `objects`, `text`. Chunkers: `uniform`,
`scene`.

## The pipeline

```
probe ──▶ read ──┬──▶ chunker.observe()          native rate, scene cuts
                 │
                 └──▶ decimate ──▶ chunk ──▶ sampler(s) ──▶ manifest + frame store
   4485 frames        299 @ 1fps    15 windows    40 kept
```

**Probe decides the timeline once, before ingesting.** The container's
timestamps and its reported frame rate are two independently corruptible
signals, so neither is trusted: an H.264 stream copied into AVI reports 600
fps, a raw `.h264` reports no timestamps at all, and a file remuxed from a
container that never stored reorder information emits correct timestamps in
*decode* order. If both signals are unusable the file is refused, because a
wrong timeline silently produces wrong chunk boundaries.

**Reading is sequential, and that is not a simplification.** Frames reference
each other, so producing the frame at second 47 means decoding forward from
its keyframe regardless. Seeking to sample would cost *more* decode work, not
less. The reader is a generator, so exactly one frame is in flight — the
alternative is 4485 × 6 MB for a five-minute 1080p file.

**Decimation buckets on media time** rather than counting every Nth frame.
Identical on a clean file; self-correcting on a lossy one, where counting
drifts permanently after a gap and bucketing snaps back within one bucket.

**Chunk boundaries derive from media time alone**, never from how many frames
arrived, so two runs of the same video agree even if one lost frames.

**Samplers reset at every chunk boundary**, and every chunk keeps at least one
frame. Rate limits (`min_interval_s`, `max_per_chunk`) are enforced by the base
class *before* the strategy runs, so a rate-limited frame costs no inference.

## Samplers

| sampler | asks | compares |
|---|---|---|
| `uniform` | every Nth frame | nothing — a positional baseline |
| `clip` | has the scene changed? | whole-frame CLIP cosine |
| `yolo` | have the people changed? | CLIP embedding of each person *crop* |
| `objects` | have things moved or appeared? | box proximity — no embedder |
| `text` | has the text changed? | the frame masked to where text was found |

Each compares against the **last frame kept**, not the previous frame, so
change accumulates across skipped frames. The reference never crosses a chunk
boundary.

`objects` uses an open vocabulary because COCO's classes are not what most
footage is about. **The vocabulary is the configuration** — there is no useful
default. Measured on shop CCTV, a mismatched list found 2.4 detections/frame
and labelled wire baskets as "shopping bag"; a matched one found 5.1.

## Thresholds

Set them per *sampler* and per *content type*, not per video. The defaults
(`clip 0.96`, `yolo 0.83`, `objects 0.30`, `text 0.92`) differ by an order of
magnitude because they compare different things — never copy one to another.

Keep rates vary widely between videos and that is the sampler working, not a
miscalibration: measured at `clip 0.96`, three retail CCTV files kept 13.4%,
18.0% and 59.2%, and the busiest one genuinely changes three times as much per
second. **If the cost is too high, use `min_interval_s` / `max_per_chunk`**, which
keep the most-changed frames spaced out, rather than lowering the threshold,
which just keeps fewer and leaves the per-chunk distribution lopsided.

`calibrate.py` does not choose a threshold. It reports what every threshold
would cost, from one cached model pass, and measures the **noise floor** — the
similarity between adjacent frames when nothing happened. A threshold above
that samples encoder quantization rather than content: one PNG looped for 60 s
decodes to 60 *different* frames, and a threshold of 1.000 keeps all of them.

## The manifest

```jsonc
{
  "video_id": "...",
  "complete": true,              // false while a live run is still going
  "source":  { "fps": 15.0, "time_base": "1/15360", "timeline": "pts", ... },
  "config":  { "decimator": {...}, "chunker": {...}, "samplers": [...] },
  "chunks": [{
    "chunk_id": 0, "start_ts": 0.0, "end_ts": 20.0, "decimated_frames": 20,
    "samplers": {
      "clip": { "frame_count": 4, "frames": [
        { "index": 0, "media_ts": 0.0, "chunk_local_index": 0, "pts": 0, "score": 0.94 }
      ]}
    }
  }]
}
```

It is rewritten as each chunk closes, via write-to-temp plus `os.replace`, so a
reader sees a complete document or the previous one, never a torn file. A run
killed at 94% of a 23-minute video left 128 chunks and 325 frames fully usable.

Frames carry `pts` in the container's integer timebase, not just seconds:
`pts × time_base` is exact, and at a 1/1200000 timebase a rounded float lands
on the wrong frame.

## Frame store

Optional. Ingest already holds the pixels, so writing them costs ~4.8 ms/frame
against ~167 ms to seek one back out. Keyed by frame index rather than by
sampler, because samplers overlap — 186 picks across four samplers covered 114
distinct frames.

The manifest stays authoritative: `recreate.py` rebuilds a store from it plus
the source video, verified **byte-identical** on 350/350 frames.

It is not a cache for retuning thresholds. JPEG at q85/1920px perturbs pixels
by 1.6/255, three times the decoder difference that already shifts the
detection samplers by 12–15%. Sweeping a threshold wants cached *descriptors*
(2 KB per frame), which is what `calibrate.py` does.

## Formats

Anything FFmpeg opens. Verified on MP4/MOV/MKV/TS/WebM/AVI carrying H.264,
HEVC, VP9, AV1, MJPEG and ProRes, plus variable frame rate, rotated portrait
video, remuxes with scrambled reorder information, and raw elementary streams.

Extensions are not whitelisted, because the failures don't track extensions:
what breaks is a codec in a container that cannot express it (H.264 with
B-frames in AVI), no container at all, or corrupt rate metadata.

## Install

```bash
pip install -r requirements.txt
```

`av`, `opencv-python` and `numpy` are all the `uniform` sampler needs. The
model-backed samplers are imported lazily, so nothing pulls in torch,
ultralytics or easyocr until you ask for them.

## Not yet built

Live sources. `Frame.gap_before` and `Frame.discontinuity` are the seams —
always `0`/`False` for a file, and where a stream will differ. The describer
stage reads the manifest and does not exist here.
