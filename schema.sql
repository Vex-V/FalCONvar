-- FalCONvar: the Supabase schema, in full.
--
-- Run in the Supabase SQL editor. Writes use the secret key (sb_secret_...),
-- which bypasses RLS; reads use the publishable key (sb_publishable_...),
-- which maps to the `anon` role and is safe to hand out with a URL.
--
-- RLS enabled with no policy denies reads SILENTLY -- zero rows, no error --
-- so verify by row count under the publishable key, never by absence of an
-- exception.

-- ---------------------------------------------------------------- manifests
create table if not exists videos (
  video_id         text primary key,
  complete         boolean not null default false,
  manifest_version int     not null,
  source           jsonb   not null,
  config           jsonb   not null,
  stats            jsonb   not null default '{}'::jsonb,
  ingested_at      timestamptz not null default now()
);

create table if not exists chunks (
  video_id         text references videos on delete cascade,
  chunk_id         int,
  start_ts         numeric not null,
  end_ts           numeric not null,
  decimated_frames int     not null,
  samplers         jsonb   not null,
  primary key (video_id, chunk_id)
);

create index if not exists chunks_video_start on chunks (video_id, start_ts);
create index if not exists chunks_samplers    on chunks using gin (samplers jsonb_path_ops);

-- The manifest, reassembled server-side so there is no second implementation
-- of the format to drift. Verified byte-for-byte equivalent (as parsed data --
-- jsonb does not preserve key order) to what the file writer produces.
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

-- ------------------------------------------------------------- descriptions
-- One row per (video_id, chunk_id, sampler): the unit one describer call
-- covers. NO foreign key to videos, deliberately -- ingest replaces a manifest
-- wholesale and cascades chunks with it, and re-ingesting costs 20 seconds
-- where describing costs inference. A cascade would let the cheap operation
-- destroy the expensive one.
--
-- manifest_fingerprint is what replaces the cascade: a hash of the settings a
-- manifest records (source minus uri, config minus frame_store), so it is
-- known before the first chunk exists, survives the video and store being
-- moved, and changes when the sampling changes. Staleness becomes a
-- comparison rather than a deletion.
create table if not exists descriptions (
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

create index if not exists descriptions_video_chunk on descriptions (video_id, chunk_id);

-- The same observation, broken out: setting, entities, actions, visible_text,
-- changes. `description` stays the prose because that is what gets embedded
-- and read; this is what a filter can use -- every moment showing a basket,
-- every moment where a sign said something.
alter table descriptions
  add column if not exists structured jsonb not null default '{}'::jsonb;

create index if not exists descriptions_structured
  on descriptions using gin (structured jsonb_path_ops);

create or replace function export_descriptions(p_video_id text)
returns jsonb language sql stable as $$
  select coalesce(jsonb_agg(jsonb_build_object(
      'chunk_id',    chunk_id,    'sampler',       sampler,
      'frame_count', frame_count, 'frame_indexes', frame_indexes,
      'description', description, 'elapsed_s',     elapsed_s,
      'structured',  structured,
      'model',       model,
      'manifest_fingerprint', manifest_fingerprint)
      order by chunk_id, sampler), '[]'::jsonb)
  from descriptions where video_id = p_video_id;
$$;

-- --------------------------------------------------------------------- RLS
alter table videos       enable row level security;
alter table chunks       enable row level security;
alter table descriptions enable row level security;

drop policy if exists "public read" on videos;
drop policy if exists "public read" on chunks;
drop policy if exists "public read" on descriptions;

create policy "public read" on videos       for select to anon using (true);
create policy "public read" on chunks       for select to anon using (true);
create policy "public read" on descriptions for select to anon using (true);

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
--     alter table description_embeddings add column embedding_fixed vector(1536);
--     create index on description_embeddings using hnsw
--            (embedding_fixed vector_cosine_ops) where embedder = '...';
--
-- and note pgvector's HNSW limit of 2000 dimensions for the `vector` type.
create table if not exists description_embeddings (
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
alter table description_embeddings
  add column if not exists structured jsonb not null default '{}'::jsonb;

alter table description_embeddings
  add column if not exists fts tsvector
  generated always as (
    to_tsvector('english', content || ' ' || jsonb_path_query_array(
      structured, 'strict $.**?(@.type() == "string")')::text)
  ) stored;

create index if not exists description_embeddings_structured
  on description_embeddings using gin (structured jsonb_path_ops);

create index if not exists description_embeddings_fts
  on description_embeddings using gin (fts);
create index if not exists description_embeddings_video
  on description_embeddings (video_id, chunk_id);

-- Hybrid search, fused with Reciprocal Rank Fusion.
--
-- RRF rather than a weighted sum of scores: cosine distance and ts_rank are on
-- unrelated scales, and any weighting between them is a number nobody can
-- justify. RRF only reads the two *orderings*, so it needs no calibration --
-- a row ranked 1st by vectors and 8th by text scores 1/(k+1) + 1/(k+8).
create or replace function search_descriptions(
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
    select e.* from description_embeddings e
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

alter table description_embeddings enable row level security;
drop policy if exists "public read" on description_embeddings;
create policy "public read" on description_embeddings for select to anon using (true);
