# Putting manifests on Supabase

**Goal:** publish a manifest so anyone holding a copy of the source video can
run `recreate.py` and rebuild the frame store themselves.

**Pixels never leave the machine that ingested them.** The frame store stays
local — it is a cache, byte-identically regenerable from the manifest, so
uploading it would cost latency on every ingest and storage fees forever for
something rebuildable in about a second.

---

## Phase 0 — setup

**0.1** Create the Supabase project. Copy the project URL and the
`service_role` key. Server-side inserts need `service_role`; the `anon` key is
read-only under default RLS, which is what a manifest *consumer* should use.

**0.2** Add empty slots to `.env.example` and real values to `.env` (already
gitignored):

```
SUPABASE_URL=
SUPABASE_SERVICE_KEY=
```

**0.3** Install and immediately check the environment:

```bash
pip install supabase
python -m ver2.imports
```

`supabase` pulls in `httpx` and `pydantic`. Adding PaddleOCR previously
downgraded numpy and swapped the active `cv2` without a word, so do not skip
this.

> **Done when** `imports.py` reports `all imports OK` and
> `python -m ver2.ingest.driver media/test2.mp4 --sampler clip` still gives
> `10 frames (16.4% of decimated)`.

---

## Phase 1 — extract the manifest interface

No behaviour change. Land it on its own.

**1.1** `ver2/ingest/output/base.py`:

```python
from typing import Protocol

class ManifestSink(Protocol):
    def chunk_closed(self, chunk: dict, stats: dict | None = None) -> None: ...
    def finish(self, stats: dict | None = None) -> dict: ...
```

That is the entire surface `pipeline.py` uses.

**1.2** `pipeline.py` currently also assigns `writer.chunks` and
`writer.config` once, just before `finish()` — a late correction for the final
chunk's `end_ts` and the scene chunker's cut counters. **Fold those into
`finish(chunks=..., config=...)`** so a remote sink need not expose mutable
attributes.

**1.3** Rename `ManifestWriter` -> `FileManifestWriter`, keeping the old name
as an alias in `output/__init__.py`. Add both to `INTERNAL` in
`ver2/imports.py`.

> **Done when** a full run plus `recreate` is still byte-identical. Nothing
> about behaviour changed, so any difference is a bug introduced here.

---

## Phase 2 — schema and export function

**2.1** In the Supabase SQL editor:

```sql
create table videos (
  video_id         text primary key,
  complete         boolean not null default false,
  manifest_version int     not null,
  source           jsonb   not null,
  config           jsonb   not null,
  stats            jsonb   not null default '{}'::jsonb,
  ingested_at      timestamptz not null default now()
);

create table chunks (
  video_id         text references videos on delete cascade,
  chunk_id         int,
  start_ts         numeric not null,
  end_ts           numeric not null,
  decimated_frames int     not null,
  samplers         jsonb   not null,
  primary key (video_id, chunk_id)
);

create index on chunks (video_id, start_ts);
create index on chunks using gin (samplers jsonb_path_ops);
```

`samplers` goes in **verbatim** — every frame record keeps its `index`,
`media_ts`, `pts`, `chunk_local_index` and `score` unchanged. The only
structural change from the file is that the `chunks` array becomes rows.

**2.2** The export function. **Write this now, not later** — it is the
verification tool for Phase 3.

```sql
create or replace function export_manifest(p_video_id text)
returns jsonb language sql stable as $$
  select jsonb_build_object(
    'manifest_version', v.manifest_version,
    'video_id',         v.video_id,
    'complete',         v.complete,
    'source',           v.source,
    'config',           v.config,
    'stats',            v.stats,
    'chunks', coalesce(jsonb_agg(
        jsonb_build_object(
          'chunk_id',         c.chunk_id,
          'start_ts',         c.start_ts,
          'end_ts',           c.end_ts,
          'decimated_frames', c.decimated_frames,
          'samplers',         c.samplers)
        order by c.chunk_id) filter (where c.chunk_id is not null), '[]'::jsonb))
  from videos v
  left join chunks c using (video_id)
  where v.video_id = p_video_id
  group by v.video_id, v.manifest_version, v.complete, v.source, v.config, v.stats;
$$;
```

**2.3** Row-level security: `service_role` writes, `anon` reads. Public read is
the point — that is how a recipient fetches a manifest.

> **Done when** you hand-insert one video plus one chunk, call
> `export_manifest`, and `diff` the result against a real manifest file with no
> differences.

---

## Phase 3 — the Supabase sink

**3.1** `ver2/ingest/output/supabase_manifest.py` implementing `ManifestSink`:

| method | does |
|---|---|
| `__init__` | upsert `videos` with `complete = false` |
| `chunk_closed` | one `INSERT` into `chunks` |
| `finish` | `UPDATE videos SET complete = true, stats = ..., config = ...` |

This is where the file version's atomic rewrite disappears. `FileManifestWriter`
rewrites the whole document every time a chunk closes (write-to-temp plus
`os.replace`) because a JSON file cannot be safely appended to while someone
reads it. On the 23-minute OCR video that is **137 full rewrites**; here it is
137 inserts, and a reader polls `where chunk_id > $last` instead of re-parsing
a growing document.

**3.2** Add `--sink {file,supabase}` to `driver.py`, defaulting to `file`.

**3.3** Two things to watch:

- Postgres `numeric` round-tripping `start_ts` / `end_ts` through Python floats.
- `jsonb` **does not preserve key order**. Confirm nothing depends on it —
  nothing currently does, but the byte-comparison below will catch it if it
  starts to.

> **Done when:**
>
> ```bash
> python -m ver2.ingest.driver media/test2.mp4 --sink supabase --frame-store
> # pull the manifest back out of Postgres into exported.json, then:
> python -m ver2.recovery.recreate exported.json --out rebuilt/ \
>        --verify out/stores/test2
> ```
>
> reports **24/24 frames byte-identical**. If the round-trip loses anything —
> a rounded numeric, a dropped key, lost chunk order — recreate stops being
> identical. Use that rather than inspecting rows by hand.

---

## Phase 4 — recreate fetches a manifest by URL

This removes the download step, which is the whole point.

**4.1** In `recreate.py`, accept a URL as the manifest argument:

```python
import urllib.request

if str(arg).startswith(("http://", "https://")):
    req = urllib.request.Request(arg, headers={"apikey": key} if key else {})
    with urllib.request.urlopen(req) as r:
        manifest = json.load(r)
else:
    manifest = json.loads(Path(arg).read_text(encoding="utf-8"))
```

**`urllib` is stdlib.** The handoff stays at three dependencies — `av`,
`opencv-python`, `numpy`. Do **not** add `supabase-py` here: it would break the
"hand someone one file" property, and `imports.py` enforces that
`recovery` imports nothing from `ver2` but cannot stop dependency creep.

> **Done when** this works on a machine holding only the video and the script:
>
> ```bash
> python recreate.py https://<project>.supabase.co/rest/v1/rpc/export_manifest?... \
>        --video my_copy.mp4 --out store/
> ```

---

## Phase 5 — make the manifest portable

Two recorded fields are meaningless to a recipient:

| field | example | status |
|---|---|---|
| `config.frame_store.dir` | `out/stores/test2` | fine — recreate already degrades to *"nothing to verify against"* |
| `source.uri` | `media/test2.mp4` | provenance only; recipients pass `--video` |

`recreate.py` already refuses a mismatched video by comparing `fps`,
`time_base`, `width`, `height` and `frame_count` — it correctly rejected
`test1` for the `test2` manifest with `width 1270 != 1280`,
`frame_count 1283 != 1810`.

**That is not enough to catch a re-encode** of the right video, which would
have identical dimensions and frame count but different pixels — and would
therefore rebuild a store that is plausible but not byte-identical.

**5.1** Add a content fingerprint to `source`: SHA-256 of the first 1 MB plus
the file size. Cheap, and catches re-encodes and truncations.

**5.2** Have `verify_source` check it when present and skip when absent, so
manifests written before the field exists keep working.

> **Done when** recreate refuses a re-encoded copy of the same video, not only
> a different one.

---

## Explicitly out of scope

- **Storage bucket / `SupabaseFrameStore` / `frames` table.** The store is a
  regenerable cache.
- **`descriptions` table and pgvector.** The VLM stage does not exist, so there
  is nothing to embed and no query to validate a schema against. It is the real
  reason to be on Postgres, but building it now means migrating a guess.

## Effort

| phase | effort | touches pipeline logic |
|---|---|---|
| 0 setup | 30 min | no |
| 1 interface | 1 h | no (mechanical) |
| 2 schema + export | 1 h | no |
| 3 sink | 2 h | no |
| 4 URL fetch | 30 min | no |
| 5 fingerprint | 1 h | no |

About a day. `pipeline.py`, the samplers, the chunkers and the whole `source/`
layer are untouched throughout.

**If you only do two:** Phase 3 publishes manifests; Phase 5 stops someone
silently rebuilding from the wrong footage.
