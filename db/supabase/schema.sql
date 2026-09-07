-- ver3 · runnable DDL.
--
-- EVERYTHING LIVES IN THE `ver3` SCHEMA, not `public`. falconvar's tables are
-- still deployed alongside, and one name already collides: `video_embeddings`
-- exists there with different columns, so `create table if not exists` would
-- be a silent no-op and every write would go to the wrong shape or fail. This
-- is the same collision `data/ver3/` fixes on disk, and it was found the same
-- way -- by looking at what is actually deployed rather than assuming.
--
-- Expose it to PostgREST with:
--   Dashboard > Settings > API > Exposed schemas: add `ver3`
-- and point the client at it with `postgrest_client_options={"schema": "ver3"}`.
--
-- Everything downstream is keyed by (video_id, chunk_id), and the grid is
-- stored exactly once. `falconvar` keeps start_ts/end_ts in both video_chunks
-- and audio_chunks with nothing constraining them to agree; here there is one
-- `chunks` table and both sides join it.
--
-- CASCADE FOLLOWS COST. Whether a table cascades from the grid is decided by
-- one question: if this were deleted, what would it cost to rebuild?
--
--   rebuildable in seconds  ->  foreign key, on delete cascade
--   cost inference or a paid call  ->  NO foreign key
--
-- That is not a preference. Re-ingesting a video costs seconds where describing
-- it costs money, so a cascade from the grid into `descriptions` would mean
-- retuning a scene threshold silently destroys everything a VLM was paid to
-- produce. Staleness there is a fingerprint comparison a reader makes, never a
-- deletion the database makes.

create schema if not exists ver3;

-- ===========================================================================
-- 1 · the file
-- ===========================================================================
create table if not exists ver3.videos (
  video_id      text primary key,
  path          text not null,
  container     text not null,
  duration_s    numeric,
  has_video     boolean not null,
  has_audio     boolean not null,
  video_stream  jsonb,                    -- codec, rate, time_base, w, h, frames
  audio_stream  jsonb,                    -- codec, rate, channels
  seen_at       timestamptz not null default now()
);

-- ===========================================================================
-- 4 · THE GRID. One per video.
-- ===========================================================================
create table if not exists ver3.timelines (
  video_id      text primary key references ver3.videos on delete cascade,
  policy        text not null,            -- uniform | scene | vad | speaker
  derived_from  text not null,            -- video | audio | grid
  params        jsonb not null default '{}'::jsonb,
  fingerprint   text not null,
  duration_s    numeric not null,
  chunk_count   int not null,
  built_at      timestamptz not null default now()
);

create table if not exists ver3.chunks (
  video_id  text not null references ver3.timelines on delete cascade,
  chunk_id  int  not null,
  start_ts  numeric not null,
  end_ts    numeric not null,
  primary key (video_id, chunk_id),
  check (end_ts > start_ts)
);

create index if not exists chunks_start on ver3.chunks (video_id, start_ts);

-- ===========================================================================
-- 3 · boundary evidence. Kept because the score series is what makes
--     re-thresholding arithmetic instead of another pass over the video.
-- ===========================================================================
create table if not exists ver3.cuts (
  video_id    text primary key references ver3.videos on delete cascade,
  source      text not null,              -- video | audio
  detector    text not null,              -- content | vad | speaker
  params      jsonb not null default '{}'::jsonb,
  cut_times   numeric[] not null default '{}',
  scores      jsonb,                      -- {metric, stride, at[], values[]}
  stats       jsonb not null default '{}'::jsonb
);

-- ===========================================================================
-- 2 + 6 · the soundtrack. The raw transcript is stored before it is cut, so a
--     grid change never re-runs Whisper.
-- ===========================================================================
create table if not exists ver3.transcripts (
  video_id     text primary key references ver3.videos on delete cascade,
  timeline_fingerprint text,
  model        jsonb not null default '{}'::jsonb,
  track        jsonb not null default '{}'::jsonb,   -- rms, peak, silent
  stats        jsonb not null default '{}'::jsonb,
  segments     jsonb not null default '[]'::jsonb,
  words        jsonb not null default '[]'::jsonb,
  turns        jsonb not null default '[]'::jsonb,
  heard_at     timestamptz not null default now()
);

create table if not exists ver3.transcript_chunks (
  video_id    text not null,
  chunk_id    int  not null,
  text        text not null default '',
  word_count  int  not null default 0,
  -- {speakers: [...]} only. turns[].text IS the transcript, so putting it here
  -- would append the whole chunk a second time in anything that renders this.
  structured  jsonb not null default '{}'::jsonb,
  turns       jsonb not null default '[]'::jsonb,
  primary key (video_id, chunk_id),
  foreign key (video_id, chunk_id) references ver3.chunks on delete cascade
);

-- ===========================================================================
-- 5 · the picture. Cheap to rebuild, so it cascades.
-- ===========================================================================
create table if not exists ver3.manifests (
  video_id     text primary key references ver3.videos on delete cascade,
  timeline_fingerprint text not null,
  manifest_fingerprint text not null,
  source       jsonb not null,
  config       jsonb not null,            -- decimator, samplers[], frame_store
  stats        jsonb not null default '{}'::jsonb,
  ingested_at  timestamptz not null default now()
);

-- One row per (chunk, sampler) rather than a jsonb blob on the chunk. That is
-- what makes "which chunks did yolo pick frames in" a query rather than a
-- scan, and it is the level --sampler actually filters at.
create table if not exists ver3.chunk_samplers (
  video_id    text not null,
  chunk_id    int  not null,
  sampler_id  text not null,              -- "yolo" | "yolo:overview"
  sampler     text not null,              -- the strategy half
  question    text not null,              -- the prompt half, resolved
  frame_count int  not null,
  frames      jsonb not null,             -- [{index, media_ts, chunk_local_index, pts, score?}]
  primary key (video_id, chunk_id, sampler_id),
  foreign key (video_id, chunk_id) references ver3.chunks on delete cascade
);

create index if not exists chunk_samplers_sampler
  on ver3.chunk_samplers (video_id, sampler);

-- ===========================================================================
-- 7 · descriptions. NO FOREIGN KEY, deliberately. See the header.
-- ===========================================================================
create table if not exists ver3.descriptions (
  video_id      text not null,
  chunk_id      int  not null,
  sampler_id    text not null,
  question      text not null,
  frame_indexes int[] not null,
  frame_count   int  not null,
  description   text,
  structured    jsonb not null default '{}'::jsonb,
  model         jsonb not null default '{}'::jsonb,  -- describer, model, prompts hash
  elapsed_s     numeric,
  timeline_fingerprint text,
  manifest_fingerprint text,
  described_at  timestamptz not null default now(),
  primary key (video_id, chunk_id, sampler_id)
);

create index if not exists descriptions_chunk on ver3.descriptions (video_id, chunk_id);

-- ===========================================================================
-- 8 · embeddings. Also no foreign key, for the same reason: paid calls.
--
-- One vector space per EMBEDDER, never per sampler. A space is defined by the
-- model, not by which prompt produced the text, so everything one embedder
-- writes is comparable; the sampler is payload and querying one is a filter.
-- The embedder key is in the primary key, which is what stops 768-wide vectors
-- being ranked against 1536-wide ones.
-- ===========================================================================
create extension if not exists vector;

create table if not exists ver3.embeddings (
  video_id    text not null,
  chunk_id    int  not null,
  sampler_id  text not null,              -- + "transcript" for the audio side
  embedder    text not null,              -- name:model:dims
  text_hash   text not null,              -- embed only what changed
  content     text not null,
  structured  jsonb not null default '{}'::jsonb,
  embedding   vector(1536) not null,
  timeline_fingerprint text,
  embedded_at timestamptz not null default now(),
  primary key (video_id, chunk_id, sampler_id, embedder)
);

-- Generated, and rebuilt unconditionally rather than added if absent.
-- `add column if not exists` is a no-op when the column exists, so an `fts`
-- built by an earlier version of this file -- over `content` alone, before the
-- structured values were folded in -- survives a re-run untouched, and the
-- lexical half quietly stops indexing the terms it is best at. Nothing reports
-- it. The column is generated, so dropping it loses nothing.
alter table ver3.embeddings drop column if exists fts;
alter table ver3.embeddings add column fts tsvector
  generated always as (
    to_tsvector('english', content || ' ' || jsonb_path_query_array(
      structured, 'strict $.**?(@.type() == "string")')::text)
  ) stored;

create index if not exists embeddings_fts on ver3.embeddings using gin (fts);
create index if not exists embeddings_structured
  on ver3.embeddings using gin (structured jsonb_path_ops);
create index if not exists embeddings_vector on embeddings
  using hnsw (embedding vector_cosine_ops);

-- ===========================================================================
-- 9 · aggregates. Video-level, so not keyed by chunk at all.
-- ===========================================================================
create table if not exists ver3.aggregates (
  video_id     text not null references ver3.videos on delete cascade,
  aggregate_id text not null,             -- stats | speakers | summary | ...
  tier         text not null,             -- free | local | llm
  payload      jsonb not null,
  inputs_fingerprint text not null,       -- hash of the chunk text actually read
  built_at     timestamptz not null default now(),
  primary key (video_id, aggregate_id)
);

-- Only the summary is embedded, and into its own table. `embeddings` answers
-- *which twenty seconds*; a summary answers *which video*, and a video is not
-- a moment you can play. Putting summaries in the chunk table would need a
-- sentinel chunk_id and would return a whole-video "moment" beside real ones
-- in every search.
create table if not exists ver3.video_embeddings (
  video_id   text not null references ver3.videos on delete cascade,
  kind       text not null,               -- "summary"
  embedder   text not null,
  text_hash  text not null,
  content    text not null,
  embedding  vector(1536) not null,
  primary key (video_id, kind, embedder)
);
