-- FalCONvar: the Supabase schema, in full.
--
-- Run in the Supabase SQL editor. Writes use the secret key (sb_secret_...),
-- which bypasses RLS; reads use the publishable key (sb_publishable_...),
-- which maps to the `anon` role and is safe to hand out with a URL.
--
-- RLS enabled with no policy denies reads SILENTLY -- zero rows, no error --
-- so verify by row count under the publishable key, never by absence of an
-- exception.

-- ---------------------------------------------------------------- teardown
-- The tables were renamed to say which stream they belong to. These are the
-- names they had before: `videos`, `chunks`, `descriptions`,
-- `description_embeddings`, `transcripts`, `transcript_chunks`. Dropping them
-- is a no-op on a database that never had them, so this stays safe to re-run
-- and can be deleted once every deployment has been through it once.
--
-- The functions go too: a `create or replace` cannot rename one, and the old
-- names would otherwise survive as working RPCs returning stale shapes -- the
-- same silent-staleness failure the `fts` column had.
-- Without an argument list. Naming the types was tried and failed silently:
-- `search_descriptions(text, vector, text, text, text, int, int)` matched
-- nothing, so the function survived a full teardown and was left pointing at a
-- table that no longer existed. The bare form drops whatever overload is
-- there, and errors loudly if a name is ambiguous rather than doing nothing.
drop function if exists export_manifest;
drop function if exists export_descriptions;
drop function if exists export_transcript;
drop function if exists search_descriptions;
drop function if exists search_transcripts;

drop table if exists description_embeddings cascade;
drop table if exists transcript_chunks cascade;
drop table if exists transcripts        cascade;
drop table if exists descriptions       cascade;
drop table if exists chunks             cascade;
drop table if exists videos             cascade;

-- ---------------------------------------------------------------- manifests
create table if not exists video_manifests (
  video_id         text primary key,
  complete         boolean not null default false,
  manifest_version int     not null,
  source           jsonb   not null,
  config           jsonb   not null,
  stats            jsonb   not null default '{}'::jsonb,
  ingested_at      timestamptz not null default now()
);

create table if not exists video_chunks (
  video_id         text references video_manifests on delete cascade,
  chunk_id         int,
  start_ts         numeric not null,
  end_ts           numeric not null,
  decimated_frames int     not null,
  samplers         jsonb   not null,
  primary key (video_id, chunk_id)
);

create index if not exists video_chunks_start on video_chunks (video_id, start_ts);
create index if not exists video_chunks_samplers on video_chunks using gin (samplers jsonb_path_ops);

-- The manifest, reassembled server-side so there is no second implementation
-- of the format to drift. Verified byte-for-byte equivalent (as parsed data --
-- jsonb does not preserve key order) to what the file writer produces.
create or replace function export_video_manifest(p_video_id text)
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
  from video_manifests v
  left join video_chunks c using (video_id)
  where v.video_id = p_video_id
  group by v.video_id, v.manifest_version, v.complete, v.source, v.config, v.stats;
$$;

-- ------------------------------------------------------------- video_descriptions
-- One row per (video_id, chunk_id, sampler): the unit one describer call
-- covers. NO foreign key to video_manifests, deliberately -- ingest replaces a manifest
-- wholesale and cascades video_chunks with it, and re-ingesting costs 20 seconds
-- where describing costs inference. A cascade would let the cheap operation
-- destroy the expensive one.
--
-- manifest_fingerprint is what replaces the cascade: a hash of the settings a
-- manifest records (source minus uri, config minus frame_store), so it is
-- known before the first chunk exists, survives the video and store being
-- moved, and changes when the sampling changes. Staleness becomes a
-- comparison rather than a deletion.
create table if not exists video_descriptions (
  video_id       text  not null,
  chunk_id       int   not null,
  sampler        text  not null,
  frame_indexes  int[] not null,
  frame_count    int   not null,
  description    text,
  model          jsonb not null default '{}'::jsonb,
  elapsed_s      numeric,
  manifest_fingerprint text,
  described_at   timestamptz not null default now(),
  primary key (video_id, chunk_id, sampler)
);

create index if not exists video_descriptions_chunk on video_descriptions (video_id, chunk_id);

-- The same observation, broken out: setting, entities, actions, visible_text,
-- changes. `description` stays the prose because that is what gets embedded
-- and read; this is what a filter can use -- every moment showing a basket,
-- every moment where a sign said something.
alter table video_descriptions
  add column if not exists structured jsonb not null default '{}'::jsonb;

create index if not exists video_descriptions_structured
  on video_descriptions using gin (structured jsonb_path_ops);

create or replace function export_video_descriptions(p_video_id text)
returns jsonb language sql stable as $$
  select coalesce(jsonb_agg(jsonb_build_object(
      'chunk_id',    chunk_id,    'sampler',       sampler,
      'frame_count', frame_count, 'frame_indexes', frame_indexes,
      'description', description, 'elapsed_s',     elapsed_s,
      'structured',  structured,
      'model',       model,
      'manifest_fingerprint', manifest_fingerprint)
      order by chunk_id, sampler), '[]'::jsonb)
  from video_descriptions where video_id = p_video_id;
$$;

-- --------------------------------------------------------------------- RLS
alter table video_manifests       enable row level security;
alter table video_chunks       enable row level security;
alter table video_descriptions enable row level security;

drop policy if exists "public read" on video_manifests;
drop policy if exists "public read" on video_chunks;
drop policy if exists "public read" on video_descriptions;

create policy "public read" on video_manifests       for select to anon using (true);
create policy "public read" on video_chunks       for select to anon using (true);
create policy "public read" on video_descriptions for select to anon using (true);

-- ---------------------------------------------------------------- retrieval
create extension if not exists vector;

-- One row per (description, embedder). The embedder is part of the key so
-- several can coexist and be compared on the same corpus -- which is the whole
-- point of having an Embedder interface rather than a hardcoded model.
--
-- `embedding` is declared `vector` with NO dimension on purpose. A fixed
-- `vector(1536)` would lock the table to one embedder's width, and nomic (768),
-- bge-m3 (1024) and OpenAI (1536) all differ. Unconstrained means exact cosine
-- search only -- no ANN index -- which is correct at this scale: a few thousand
-- descriptions scan in under a millisecond. Once an embedder is settled, add
--
--     alter table chunk_embeddings add column embedding_fixed vector(1536);
--     create index on chunk_embeddings using hnsw
--            (embedding_fixed vector_cosine_ops) where embedder = '...';
--
-- and note pgvector's HNSW limit of 2000 dimensions for the `vector` type.
create table if not exists chunk_embeddings (
  video_id       text not null,
  chunk_id       int  not null,
  sampler        text not null,
  embedder       text not null,          -- "openai:text-embedding-3-small:1536"
  dims           int  not null,
  embedding      vector not null,
  content        text not null,          -- the summary, for display
  structured     jsonb not null default '{}'::jsonb,   -- for exact filtering
  text_hash      text not null,          -- staleness: re-describing changes this
  manifest_fingerprint text,
  start_ts       numeric,
  end_ts         numeric,
  frame_indexes  int[],
  indexed_at     timestamptz not null default now(),
  primary key (video_id, chunk_id, sampler, embedder)
);

-- Full text over the description, maintained by Postgres rather than by us.
-- This is half of the hybrid: the `text` sampler transcribes signage verbatim,
-- and for "the sign that says CLOSING DOWN" exact terms beat embeddings.
-- Full text over the summary *and* the structured values. Measured: BM25 over
-- summaries alone answers literal queries at MRR 0.752, and the structured
-- fields are where the verbatim signage and the exact garments live, so
-- leaving them out of the lexical index throws away the terms it is best at.
alter table chunk_embeddings
  add column if not exists structured jsonb not null default '{}'::jsonb;

-- Dropped and rebuilt rather than conditionally added, and this is the one
-- statement in the file that has to be. `add column if not exists` is a no-op
-- when the column is already there, so an `fts` generated by an EARLIER
-- version of this file -- over `content` alone, before the structured values
-- were folded in -- survives a re-run completely untouched. Nothing reports
-- it: the column exists, queries still run, and the lexical half simply stops
-- indexing the terms it is best at. Observed on this project's own database,
-- where `fts` was live and `structured` had never been added.
--
-- Nothing is lost by dropping it. The column is generated, so Postgres
-- recomputes every row from `content` and `structured`; the GIN index below
-- goes with it and is recreated by the same re-run.
alter table chunk_embeddings drop column if exists fts;

alter table chunk_embeddings
  add column fts tsvector
  generated always as (
    to_tsvector('english', content || ' ' || jsonb_path_query_array(
      structured, 'strict $.**?(@.type() == "string")')::text)
  ) stored;

create index if not exists chunk_embeddings_structured
  on chunk_embeddings using gin (structured jsonb_path_ops);

create index if not exists chunk_embeddings_fts
  on chunk_embeddings using gin (fts);
create index if not exists chunk_embeddings_video
  on chunk_embeddings (video_id, chunk_id);

-- Hybrid search, fused with Reciprocal Rank Fusion.
--
-- RRF rather than a weighted sum of scores: cosine distance and ts_rank are on
-- unrelated scales, and any weighting between them is a number nobody can
-- justify. RRF only reads the two *orderings*, so it needs no calibration --
-- a row ranked 1st by vectors and 8th by text scores 1/(k+1) + 1/(k+8).
create or replace function search_embeddings(
  p_embedder     text,
  p_query_vector vector,
  p_query_text   text default null,
  p_video_id     text default null,
  p_sampler      text default null,
  p_limit        int  default 20,
  p_rrf_k        int  default 60
)
returns table (
  video_id text, chunk_id int, sampler text,
  content text, start_ts numeric, end_ts numeric, frame_indexes int[],
  vector_rank int, text_rank int, score double precision
)
language sql stable as $$
  with candidates as (
    select e.* from chunk_embeddings e
    where e.embedder = p_embedder
      and (p_video_id is null or e.video_id = p_video_id)
      -- A filter over one shared space, not a space of its own: every row
      -- with this embedder is comparable to every other. Narrowing to one
      -- sampler asks one question's answers rather than all of them.
      and (p_sampler  is null or e.sampler  = p_sampler)
  ),
  by_vector as (
    select c.video_id, c.chunk_id, c.sampler,
           row_number() over (order by c.embedding <=> p_query_vector) as rank
    from candidates c
    order by c.embedding <=> p_query_vector
    limit greatest(p_limit * 4, 40)
  ),
  by_text as (
    select c.video_id, c.chunk_id, c.sampler,
           row_number() over (
             order by ts_rank_cd(c.fts, websearch_to_tsquery('english', p_query_text)) desc
           ) as rank
    from candidates c
    where p_query_text is not null and p_query_text <> ''
      and c.fts @@ websearch_to_tsquery('english', p_query_text)
    limit greatest(p_limit * 4, 40)
  ),
  fused as (
    select coalesce(v.video_id, t.video_id) as video_id,
           coalesce(v.chunk_id, t.chunk_id) as chunk_id,
           coalesce(v.sampler,  t.sampler)  as sampler,
           v.rank as vector_rank, t.rank as text_rank,
           coalesce(1.0 / (p_rrf_k + v.rank), 0)
         + coalesce(1.0 / (p_rrf_k + t.rank), 0) as score
    from by_vector v
    full outer join by_text t
      on v.video_id = t.video_id and v.chunk_id = t.chunk_id and v.sampler = t.sampler
  )
  select f.video_id, f.chunk_id, f.sampler,
         c.content, c.start_ts, c.end_ts, c.frame_indexes,
         f.vector_rank::int, f.text_rank::int, f.score
  from fused f
  join candidates c
    on c.video_id = f.video_id and c.chunk_id = f.chunk_id and c.sampler = f.sampler
  order by f.score desc
  limit p_limit;
$$;

alter table chunk_embeddings enable row level security;
drop policy if exists "public read" on chunk_embeddings;
create policy "public read" on chunk_embeddings for select to anon using (true);

-- ---------------------------------------------------------------- audio_transcripts
-- The audio half. Two tables rather than rows in `video_descriptions` with
-- sampler = 'transcript', which was tempting because it would inherit
-- export_video_descriptions, the embed path and retrieval for free. It is the wrong
-- trade: `model` would mean "a VLM" on some rows and "Whisper plus pyannote"
-- on others, `frame_indexes` would be empty on half of them, and `structured`
-- would hold two unrelated shapes. That reads fine until something has to
-- branch on which kind of row it is holding.
--
-- What IS shared is the vector index. A transcript chunk embeds through the
-- same `embed` stage and lands in `chunk_embeddings` with
-- sampler = 'transcript', because that table stores text with a time span and
-- a transcript chunk is exactly that. No schema change is needed there, which
-- is the evidence the embed/retrieve split was cut in the right place.
--
-- NO foreign key to `video_manifests`, for the reason `video_descriptions` has none: ingest
-- replaces a manifest wholesale and re-ingesting costs seconds, while
-- transcribing and diarizing cost model time. A cascade would let the cheap
-- operation destroy the expensive one. `timeline_fingerprint` replaces it.

-- One row per video: everything about the pass that is not per-chunk.
create table if not exists audio_transcripts (
  video_id        text primary key,
  language        text,
  language_probability numeric,
  speakers        text[]  not null default '{}',
  model           jsonb   not null default '{}'::jsonb,  -- transcriber + diarizer
  audio           jsonb   not null default '{}'::jsonb,  -- codec, rate, channels
  stats           jsonb   not null default '{}'::jsonb,
  timeline        jsonb   not null default '{}'::jsonb,  -- policy, derived_from, fingerprint
  timeline_fingerprint text,
  -- The word-level record: every segment with its words, timestamps and
  -- speaker. Stored, and not merely derivable, because it is what makes a
  -- chunk grid a decision that can be revisited -- re-cutting to a different
  -- policy is arithmetic over this, where regenerating it means paying for
  -- Whisper again. A holder of the database can therefore re-chunk; a holder
  -- of only `audio_chunks` cannot.
  segments        jsonb   not null default '[]'::jsonb,
  transcribed_at  timestamptz not null default now()
);

-- One row per (video_id, chunk_id): the current grid's view of the words.
-- Derived from `audio_transcripts.segments` plus a timeline, so this is a cache in
-- the way the frame store is a cache -- droppable, rebuildable, and not the
-- record. Chunks with no speech are kept with empty text on purpose: the grid
-- is shared with the video side, so chunk_id has to mean the same thing in
-- both, and dropping the quiet ones renumbers everything after them.
create table if not exists audio_chunks (
  video_id        text not null,
  chunk_id        int  not null,
  start_ts        numeric not null,
  end_ts          numeric not null,
  text            text not null default '',
  word_count      int  not null default 0,
  -- {speakers: [...], turns: [{speaker, start, end, text}]}. One bound record
  -- per contiguous run of one voice, so a window holding three speakers still
  -- says who said what -- the same shape the `yolo` describer returns for
  -- people, and for the same reason.
  structured      jsonb not null default '{}'::jsonb,
  timeline_fingerprint text,
  primary key (video_id, chunk_id)
);

create index if not exists audio_chunks_video on audio_chunks (video_id, start_ts);
create index if not exists audio_chunks_structured
  on audio_chunks using gin (structured jsonb_path_ops);

-- Full text over the words. Note this is NOT what retrieval searches --
-- retrieval goes through `chunk_embeddings.fts`, which is populated when
-- a transcript chunk is embedded. This one answers a different and cheaper
-- question: which video contains this phrase, without embedding anything.
alter table audio_chunks drop column if exists fts;

alter table audio_chunks
  add column fts tsvector
  generated always as (
    to_tsvector('english', text || ' ' || jsonb_path_query_array(
      structured, 'strict $.**?(@.type() == "string")')::text)
  ) stored;

create index if not exists audio_chunks_fts on audio_chunks using gin (fts);

-- The document, reassembled server-side, so there is no second implementation
-- of the format to drift from what the file writer produces.
create or replace function export_audio_transcript(p_video_id text)
returns jsonb language sql stable as $$
  select jsonb_build_object(
    'transcript_version', 1,
    'video_id',   t.video_id,
    'complete',   true,
    'timeline_fingerprint', t.timeline_fingerprint,
    'language',   t.language,
    'language_probability', t.language_probability,
    'speakers',   to_jsonb(t.speakers),
    'model',      t.model,
    'source',     jsonb_build_object('video_id', t.video_id, 'audio', t.audio),
    'stats',      t.stats,
    'timeline',   t.timeline,
    'segments',   t.segments,
    'chunks', coalesce((
      select jsonb_agg(jsonb_build_object(
               'chunk_id',   c.chunk_id,
               'start_ts',   c.start_ts,
               'end_ts',     c.end_ts,
               'text',       c.text,
               'word_count', c.word_count,
               'structured', c.structured)
             order by c.chunk_id)
      from audio_chunks c where c.video_id = t.video_id), '[]'::jsonb))
  from audio_transcripts t
  where t.video_id = p_video_id;
$$;

-- Which videos have someone saying this, without embedding anything.
create or replace function search_audio_chunks(
  p_query    text,
  p_video_id text default null,
  p_limit    int  default 20
)
returns table (
  video_id text, chunk_id int, start_ts numeric, end_ts numeric,
  text text, speakers jsonb, rank real
)
language sql stable as $$
  select c.video_id, c.chunk_id, c.start_ts, c.end_ts, c.text,
         c.structured -> 'speakers',
         ts_rank_cd(c.fts, websearch_to_tsquery('english', p_query)) as rank
  from audio_chunks c
  where c.fts @@ websearch_to_tsquery('english', p_query)
    and (p_video_id is null or c.video_id = p_video_id)
  order by rank desc
  limit p_limit;
$$;

alter table audio_transcripts       enable row level security;
alter table audio_chunks enable row level security;

drop policy if exists "public read" on audio_transcripts;
drop policy if exists "public read" on audio_chunks;

create policy "public read" on audio_transcripts       for select to anon using (true);
create policy "public read" on audio_chunks for select to anon using (true);

-- ---------------------------------------------------------------- aggregates
-- Video-level structure derived from what the chunk stages wrote: counts,
-- speaker shares, novelty ranking, a summary, chapters, events, entities.
--
-- ONE table rather than one per aggregator, and this is the opposite call from
-- keeping `video_descriptions` and `audio_transcripts` apart. There, the rows
-- had different provenance and different lifecycles, and merging them would
-- have made `model` mean two things with half the columns null. Here every row
-- is the SAME kind of thing -- a video-level document derived from the same
-- inputs -- differing only in the shape of its payload. `jsonb` is exactly for
-- that, and a new aggregator becomes a row rather than a migration.
--
-- NO foreign key to `video_manifests`, for the reason nothing else here has
-- one: re-ingesting costs seconds where an LLM aggregate costs money, and a
-- cascade would let the cheap operation destroy the expensive one.
--
-- `inputs_fingerprint` is what replaces it: a hash of the chunk text the
-- aggregator actually read. A summary of descriptions that have since been
-- rewritten is not wrong in any visible way -- it reads perfectly -- so
-- staleness has to be a comparison rather than something a reader assumes.
create table if not exists video_aggregates (
  video_id            text not null,
  aggregate_id        text not null,      -- stats, summary, chapters, events, …
  tier                text,               -- free | local | llm
  depends_on          text[] not null default '{}',
  inputs_fingerprint  text,
  config              jsonb not null default '{}'::jsonb,
  elapsed_s           numeric,
  payload             jsonb not null,
  computed_at         timestamptz not null default now(),
  primary key (video_id, aggregate_id)
);

create index if not exists video_aggregates_video on video_aggregates (video_id);
create index if not exists video_aggregates_payload
  on video_aggregates using gin (payload jsonb_path_ops);

create or replace function export_video_aggregates(p_video_id text)
returns jsonb language sql stable as $$
  select coalesce(jsonb_object_agg(aggregate_id, jsonb_build_object(
      'aggregate_id', aggregate_id, 'tier', tier,
      'depends_on', to_jsonb(depends_on),
      'inputs_fingerprint', inputs_fingerprint,
      'config', config, 'elapsed_s', elapsed_s,
      'computed_at', computed_at, 'payload', payload)), '{}'::jsonb)
  from video_aggregates where video_id = p_video_id;
$$;

-- ------------------------------------------------------------ video vectors
-- One row per (video, kind, embedder). Today the only kind is `summary`.
--
-- Separate from `chunk_embeddings` because the unit differs, not because the
-- data does. A chunk vector answers "which twenty seconds"; a summary vector
-- answers "which video" -- and a video is not a moment you can play. Putting
-- summaries in the chunk table would need a sentinel chunk_id and would then
-- return a whole-video "moment" alongside real ones in every search, competing
-- with them on a scale it does not share.
--
-- Only summaries are embedded. The rest of what `aggregate` produces is
-- statistical, and a vector of a count is not a useful thing to have.
create table if not exists video_embeddings (
  video_id       text not null,
  kind           text not null default 'summary',
  embedder       text not null,
  dims           int  not null,
  embedding      vector not null,
  content        text not null,                        -- the summary prose
  structured     jsonb not null default '{}'::jsonb,   -- key_points, topics
  text_hash      text not null,
  inputs_fingerprint text,
  indexed_at     timestamptz not null default now(),
  primary key (video_id, kind, embedder)
);

alter table video_embeddings drop column if exists fts;

alter table video_embeddings
  add column fts tsvector
  generated always as (
    to_tsvector('english', content || ' ' || jsonb_path_query_array(
      structured, 'strict $.**?(@.type() == "string")')::text)
  ) stored;

create index if not exists video_embeddings_fts on video_embeddings using gin (fts);

-- Which video, not which moment. Same RRF as `search_embeddings`, for the same
-- reason: cosine distance and ts_rank have no common scale, and RRF reads only
-- the orderings so it needs no calibration.
create or replace function search_videos(
  p_embedder     text,
  p_query_vector vector,
  p_query_text   text default null,
  p_limit        int  default 10,
  p_rrf_k        int  default 60
)
returns table (
  video_id text, kind text, content text, structured jsonb,
  vector_rank int, text_rank int, score double precision
)
language sql stable as $$
  with candidates as (
    select e.* from video_embeddings e where e.embedder = p_embedder
  ),
  by_vector as (
    select c.video_id, c.kind,
           row_number() over (order by c.embedding <=> p_query_vector) as rank
    from candidates c order by c.embedding <=> p_query_vector
    limit greatest(p_limit * 4, 40)
  ),
  by_text as (
    select c.video_id, c.kind,
           row_number() over (
             order by ts_rank_cd(c.fts, websearch_to_tsquery('english', p_query_text)) desc
           ) as rank
    from candidates c
    where p_query_text is not null and p_query_text <> ''
      and c.fts @@ websearch_to_tsquery('english', p_query_text)
    limit greatest(p_limit * 4, 40)
  ),
  fused as (
    select coalesce(v.video_id, t.video_id) as video_id,
           coalesce(v.kind, t.kind) as kind,
           v.rank as vector_rank, t.rank as text_rank,
           coalesce(1.0 / (p_rrf_k + v.rank), 0)
         + coalesce(1.0 / (p_rrf_k + t.rank), 0) as score
    from by_vector v
    full outer join by_text t on v.video_id = t.video_id and v.kind = t.kind
  )
  select f.video_id, f.kind, c.content, c.structured,
         f.vector_rank::int, f.text_rank::int, f.score
  from fused f
  join candidates c on c.video_id = f.video_id and c.kind = f.kind
  order by f.score desc limit p_limit;
$$;

alter table video_aggregates enable row level security;
alter table video_embeddings enable row level security;
drop policy if exists "public read" on video_aggregates;
drop policy if exists "public read" on video_embeddings;
create policy "public read" on video_aggregates for select to anon using (true);
create policy "public read" on video_embeddings for select to anon using (true);
