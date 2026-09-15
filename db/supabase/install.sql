-- FalCONvar · everything to run on Supabase, in one file, in this order.
--
-- Paste it into the SQL editor and run it. It is idempotent, so re-running it
-- is also how a change reaches a live database.
--
-- AFTERWARDS, one thing the SQL cannot do:
--   Dashboard > Settings > API > Exposed schemas: add `falconvar`
-- Until then every request returns "relation does not exist", which reads as a
-- missing table rather than a missing setting.
--
-- Writes use the secret key, which bypasses RLS; reads use the publishable key
-- (`anon`). RLS with no policy denies reads SILENTLY -- verify by row count
-- under the publishable key, never by the absence of an error.
--
-- CASCADE FOLLOWS COST. Rebuildable in seconds -> a foreign key, on delete
-- cascade. Cost inference or a paid call (`descriptions`, `embeddings`) -> no
-- foreign key; staleness is a fingerprint a reader compares. A cascade there
-- would let retuning a scene threshold delete what a model was paid to write.

-- ===========================================================================
-- 0 · schema and extension
-- ===========================================================================
create schema if not exists falconvar;
grant usage on schema falconvar to anon, service_role;
create extension if not exists vector;

-- ===========================================================================
-- 1 · the file
-- ===========================================================================
create table if not exists falconvar.videos (
  video_id      text primary key,
  path          text not null,
  container     text not null,
  duration_s    numeric,
  has_video     boolean not null,
  has_audio     boolean not null,
  video_stream  jsonb,
  audio_stream  jsonb,
  seen_at       timestamptz not null default now()
);

-- ===========================================================================
-- 2 · THE GRID. One per video; everything below joins it on (video_id, chunk_id).
-- ===========================================================================
create table if not exists falconvar.timelines (
  video_id      text primary key references falconvar.videos on delete cascade,
  policy        text not null,            -- uniform | scene | vad | speaker
  derived_from  text not null,            -- video | audio | grid
  params        jsonb not null default '{}'::jsonb,
  fingerprint   text not null,
  duration_s    numeric not null,
  chunk_count   int not null,
  built_at      timestamptz not null default now()
);

create table if not exists falconvar.chunks (
  video_id  text not null references falconvar.timelines on delete cascade,
  chunk_id  int  not null,
  start_ts  numeric not null,
  end_ts    numeric not null,
  primary key (video_id, chunk_id),
  check (end_ts > start_ts)
);

-- ===========================================================================
-- 3 · boundary evidence. The score series is what makes re-thresholding
--     arithmetic rather than another pass over the video.
-- ===========================================================================
create table if not exists falconvar.cuts (
  video_id    text primary key references falconvar.videos on delete cascade,
  source      text not null,              -- video | audio
  detector    text not null,              -- content | vad | speaker
  params      jsonb not null default '{}'::jsonb,
  cut_times   numeric[] not null default '{}',
  scores      jsonb,
  stats       jsonb not null default '{}'::jsonb
);

-- ===========================================================================
-- 4 · the soundtrack. The raw transcript is stored before it is cut, so a
--     grid change never re-runs Whisper.
-- ===========================================================================
create table if not exists falconvar.transcripts (
  video_id     text primary key references falconvar.videos on delete cascade,
  timeline_fingerprint text,
  model        jsonb not null default '{}'::jsonb,
  track        jsonb not null default '{}'::jsonb,
  stats        jsonb not null default '{}'::jsonb,
  segments     jsonb not null default '[]'::jsonb,
  words        jsonb not null default '[]'::jsonb,
  turns        jsonb not null default '[]'::jsonb,
  heard_at     timestamptz not null default now()
);

create table if not exists falconvar.transcript_chunks (
  video_id    text not null,
  chunk_id    int  not null,
  text        text not null default '',
  word_count  int  not null default 0,
  structured  jsonb not null default '{}'::jsonb,   -- {speakers} only: turns[].text IS the text
  turns       jsonb not null default '[]'::jsonb,
  primary key (video_id, chunk_id),
  foreign key (video_id, chunk_id) references falconvar.chunks on delete cascade
);

-- ===========================================================================
-- 5 · the picture. Cheap to rebuild, so it cascades.
-- ===========================================================================
create table if not exists falconvar.manifests (
  video_id     text primary key references falconvar.videos on delete cascade,
  timeline_fingerprint text not null,
  manifest_fingerprint text not null,
  source       jsonb not null,
  config       jsonb not null,
  stats        jsonb not null default '{}'::jsonb,
  ingested_at  timestamptz not null default now()
);

-- One row per (chunk, sampler RUN). A run is one pass over the frames and
-- `questions` what was asked about them: `clip:[text,scene]` is one row here
-- and two in `descriptions`.
create table if not exists falconvar.chunk_samplers (
  video_id    text not null,
  chunk_id    int  not null,
  sampler_id  text not null,              -- the run, which IS the strategy name
  questions   text[] not null default '{}',
  frame_count int  not null,
  frames      jsonb not null,
  primary key (video_id, chunk_id, sampler_id),
  foreign key (video_id, chunk_id) references falconvar.chunks on delete cascade
);

-- ===========================================================================
-- 6 · descriptions. NO FOREIGN KEY, deliberately: paid calls. See the header.
-- ===========================================================================
create table if not exists falconvar.descriptions (
  video_id      text not null,
  chunk_id      int  not null,
  sampler_id    text not null,
  question      text not null,
  frame_indexes int[] not null,
  frame_count   int  not null,
  description   text,
  structured    jsonb not null default '{}'::jsonb,
  model         jsonb not null default '{}'::jsonb,
  elapsed_s     numeric,
  timeline_fingerprint text,
  manifest_fingerprint text,
  described_at  timestamptz not null default now(),
  primary key (video_id, chunk_id, sampler_id)
);

-- ===========================================================================
-- 7 · embeddings. No foreign key either.
--
-- One vector space per EMBEDDER, never per sampler: the embedder key is in the
-- primary key and every query filters on it before a distance is taken, so one
-- column holds every width without two ever being compared.
--
-- No HNSW index: it needs a fixed width. The RPC narrows by embedder and video,
-- then scans exactly -- milliseconds at this size. One space with millions of
-- rows would want a partial expression index back:
-- `using hnsw ((embedding::vector(N)) vector_cosine_ops) where embedder = '...'`.
-- ===========================================================================
create table if not exists falconvar.embeddings (
  video_id    text not null,
  chunk_id    int  not null,
  sampler_id  text not null,              -- the pairing: "clip:text"
  sampler     text not null,              -- its halves, so each filter is an
  question    text not null,              -- equality; bare `clip` means both are `clip`
  embedder    text not null,              -- provider:model:dims
  text_hash   text not null,              -- embed only what changed
  content     text not null,
  structured  jsonb not null default '{}'::jsonb,
  embedding   vector not null,            -- any width
  -- The lexical half: the text plus every string in the structured answer.
  fts         tsvector generated always as (
                to_tsvector('english', content || ' ' || jsonb_path_query_array(
                  structured, 'strict $.**?(@.type() == "string")')::text)
              ) stored,
  embedded_at timestamptz not null default now(),
  primary key (video_id, chunk_id, sampler_id, embedder)
);

create index if not exists embeddings_question
  on falconvar.embeddings (video_id, embedder, question);
create index if not exists embeddings_fts on falconvar.embeddings using gin (fts);
create index if not exists embeddings_structured
  on falconvar.embeddings using gin (structured jsonb_path_ops);

-- ===========================================================================
-- 8 · video level: aggregates, and one vector per video from its summary.
--
-- The summary vector has its own table because `embeddings` answers *which
-- twenty seconds* and this answers *which video* -- a video is not a moment you
-- can play, so the two never share a ranking.
-- ===========================================================================
create table if not exists falconvar.aggregates (
  video_id     text not null references falconvar.videos on delete cascade,
  aggregate_id text not null,
  tier         text not null,             -- free | local | llm
  payload      jsonb not null,
  inputs_fingerprint text not null,
  built_at     timestamptz not null default now(),
  primary key (video_id, aggregate_id)
);

create table if not exists falconvar.video_embeddings (
  video_id   text not null references falconvar.videos on delete cascade,
  kind       text not null,               -- "summary"
  embedder   text not null,
  text_hash  text not null,
  content    text not null,
  embedding  vector not null,             -- any width, as `embeddings`
  primary key (video_id, kind, embedder)
);

-- ===========================================================================
-- 9 · prompts -- what a question said, at the version a run asked it under.
--
-- Provenance, not configuration: `data/prompts.json` stays authoritative, and
-- describe never reads this. `descriptions.model` records each question's hash,
-- which detects a changed prompt but cannot recover what the old one said.
-- Never deleted, so no description's hash is orphaned.
-- ===========================================================================
create table if not exists falconvar.prompts (
  name        text not null,
  version     text not null,          -- instruction + shape + system
  instruction text not null,
  shape       jsonb not null default '{}'::jsonb,
  summary     text not null default 'standard',
  builtin     boolean not null default false,
  about       text,
  first_seen  timestamptz not null default now(),
  primary key (name, version)
);

-- ===========================================================================
-- 10 · bringing an existing database to the shape above.
--
-- `create table if not exists` never changes a table that is already there, so
-- a change to a live one goes here. Each statement is a no-op once applied;
-- remove it when every deployment has run it.
-- ===========================================================================
-- Any embedder's width, not only OpenAI's 1536. The HNSW index needs a fixed
-- width, so it goes first.
drop index if exists falconvar.embeddings_vector;
alter table falconvar.embeddings       alter column embedding type vector;
alter table falconvar.video_embeddings alter column embedding type vector;

-- Written by nothing and read by nothing.
alter table falconvar.embeddings drop column if exists timeline_fingerprint;

-- No query uses these; `descriptions_chunk` repeated its primary key's prefix.
drop index if exists falconvar.chunks_start;
drop index if exists falconvar.chunk_samplers_run;
drop index if exists falconvar.descriptions_chunk;
drop index if exists falconvar.prompts_name;

-- ===========================================================================
-- 11 · the hybrid search, in the database.
--
-- RRF over two rankings of the same rows: cosine distance and ts_rank_cd have
-- no common scale, so only the orderings are fused. Ranked here because the
-- alternative moves a video's whole index over the wire per query.
--
-- `create or replace` cannot change a parameter list -- a new one adds an
-- overload and leaves the old one callable, answering without the new filter.
-- When the signature changes, drop the previous one in section 10.
-- ===========================================================================
create or replace function falconvar.search_embeddings(
  p_embedder     text,
  p_query_vector vector,
  p_query_text   text default null,
  p_video_ids    text[] default null, -- null = every video
  p_sampler      text default null,   -- the PAIRING: sampler_id = 'clip:text'
  p_question     text default null,   -- the question, whoever asked it
  p_strategy     text default null,   -- one sampler's whole output: 'clip'
  p_chunk_ids    int[] default null,  -- a set of chunks; a time window resolves to this
  p_structured   jsonb default null,  -- exact values, e.g. {"severity":"severe"}
  p_limit        int  default 20,
  p_rrf_k        int  default 60
)
returns table (
  video_id text, chunk_id int, sampler_id text, sampler text, question text,
  content text, structured jsonb, start_ts numeric, end_ts numeric,
  vector_rank int, text_rank int, score double precision
)
language sql stable as $$
  with candidates as (
    select e.* from falconvar.embeddings e
    where e.embedder = p_embedder
      and (p_video_ids  is null or e.video_id = any(p_video_ids))
      and (p_sampler    is null or e.sampler_id = p_sampler)
      and (p_question   is null or e.question = p_question)
      and (p_strategy   is null or e.sampler = p_strategy)
      and (p_chunk_ids  is null or e.chunk_id = any(p_chunk_ids))
      and (p_structured is null or e.structured @> p_structured)   -- GIN
  ),
  by_vector as (
    select c.video_id, c.chunk_id, c.sampler_id,
           row_number() over (order by c.embedding <=> p_query_vector) as rank
    from candidates c
    order by c.embedding <=> p_query_vector
    limit greatest(p_limit * 4, 40)
  ),
  -- ANY term, not every term: `websearch_to_tsquery` ANDs, so one word absent
  -- from the corpus silenced the whole lexical half. `&` -> `|` keeps its
  -- stemming and stopwords. (A negated term loosens too far; rare from a box.)
  query_or as (
    select nullif(replace(
             websearch_to_tsquery('english', coalesce(p_query_text, ''))::text,
             '&', '|'), '')::tsquery as q
  ),
  query_terms as (
    select array_agg(lexeme) as lexemes
    from unnest(to_tsvector('english', coalesce(p_query_text, '')))
  ),
  -- A floor on the loosened query: with two or more query lexemes a row must
  -- share two, so a lone stem collision is not an opinion. One word needs one.
  by_text as (
    select c.video_id, c.chunk_id, c.sampler_id,
           row_number() over (order by ts_rank_cd(c.fts, o.q) desc) as rank
    from candidates c
    cross join query_or o
    cross join query_terms t
    where o.q is not null
      and c.fts @@ o.q
      and (select count(*) from unnest(c.fts) d
            where d.lexeme = any(t.lexemes))
          >= least(2, coalesce(cardinality(t.lexemes), 1))
    limit greatest(p_limit * 4, 40)
  ),
  fused as (
    select coalesce(v.video_id,   t.video_id)   as video_id,
           coalesce(v.chunk_id,   t.chunk_id)   as chunk_id,
           coalesce(v.sampler_id, t.sampler_id) as sampler_id,
           v.rank as vector_rank, t.rank as text_rank,
           coalesce(1.0 / (p_rrf_k + v.rank), 0)
         + coalesce(1.0 / (p_rrf_k + t.rank), 0) as score
    from by_vector v
    full outer join by_text t
      on  v.video_id   = t.video_id
      and v.chunk_id   = t.chunk_id
      and v.sampler_id = t.sampler_id
  )
  select f.video_id, f.chunk_id, f.sampler_id, c.sampler, c.question,
         c.content, c.structured,
         k.start_ts, k.end_ts,                -- from the grid, the one place a span lives
         f.vector_rank::int, f.text_rank::int, f.score
  from fused f
  join candidates c
    on  c.video_id   = f.video_id
    and c.chunk_id   = f.chunk_id
    and c.sampler_id = f.sampler_id
  left join falconvar.chunks k
    on k.video_id = f.video_id and k.chunk_id = f.chunk_id
  -- RRF ties are exact and common; the vector rank breaks them, since that
  -- half has an opinion on every query.
  order by f.score desc,
           f.vector_rank asc nulls last,
           f.video_id, f.chunk_id, f.sampler_id
  limit p_limit;
$$;

-- ===========================================================================
-- 12 · row level security and grants. LAST: RLS enabled before its policy
--      denies reads in the gap, silently. Read-only for `anon`.
-- ===========================================================================
do $$
declare t text;
begin
  foreach t in array array[
    'videos','timelines','chunks','cuts','transcripts','transcript_chunks',
    'manifests','chunk_samplers','descriptions','embeddings','aggregates',
    'video_embeddings','prompts'
  ] loop
    execute format('alter table falconvar.%I enable row level security', t);
    execute format('drop policy if exists "public read" on falconvar.%I', t);
    execute format(
      'create policy "public read" on falconvar.%I for select to anon using (true)', t);
  end loop;
end $$;

grant select  on all tables in schema falconvar to anon;
grant all     on all tables in schema falconvar to service_role;
grant execute on function falconvar.search_embeddings(
  text, vector, text, text[], text, text, text, int[], jsonb, int, int)
  to anon, service_role;

-- ===========================================================================
-- Verify under the PUBLISHABLE key, not this editor -- the editor is a
-- superuser and shows rows whether or not `anon` can:
--
--   select count(*) from falconvar.videos;          -- 0 is fine; an error is not
--
-- And from the client: python -m falconvar.rag.retrieve "..." <id> --index supabase
-- ===========================================================================
