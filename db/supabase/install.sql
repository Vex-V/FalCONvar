-- FalCONvar · everything to run on Supabase, in one file, in this order.
--
-- Paste the whole thing into the SQL editor and run it. It is idempotent:
-- running it twice is a no-op except where a re-run is the point (see `fts`),
-- so this is also how to apply a change to any one part -- there is no
-- separate patch file to keep in step with it.
--
-- AFTERWARDS, one thing the SQL cannot do:
--   Dashboard > Settings > API > Exposed schemas: add `falconvar`
-- PostgREST only serves schemas on that list, and until it is there every
-- request returns a "relation does not exist" that looks like a missing table
-- rather than a missing setting.
--
-- Writes use the secret key, which bypasses RLS. Reads use the publishable
-- key, which maps to `anon`.
--
-- RLS ENABLED WITH NO POLICY DENIES READS SILENTLY -- zero rows, no error. So
-- verify by row count under the publishable key, never by absence of an
-- exception.
--
-- CASCADE FOLLOWS COST. Whether a table cascades from the grid is decided by
-- one question: if this were deleted, what would it cost to rebuild?
--   rebuildable in seconds        -> foreign key, on delete cascade
--   cost inference or a paid call -> NO foreign key; staleness is a
--                                    fingerprint a reader compares
-- Re-ingesting costs seconds where describing costs money, so a cascade from
-- the grid into `descriptions` would mean retuning a scene threshold silently
-- destroying everything a VLM was paid to produce.

-- ===========================================================================
-- 0 · schema and extension
-- ===========================================================================
create schema if not exists falconvar;

grant usage on schema falconvar to anon, authenticated, service_role;

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

create index if not exists chunks_start on falconvar.chunks (video_id, start_ts);

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
  -- {speakers: [...]} only. turns[].text IS the transcript, so putting it here
  -- would append the whole chunk a second time in anything that renders this.
  structured  jsonb not null default '{}'::jsonb,
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

-- One row per (chunk, sampler) rather than a jsonb blob on the chunk. That is
-- what makes "which chunks did yolo pick frames in" a query rather than a scan.
-- One row per sampler RUN: one pass over the frames, and the list of questions
-- asked about what it kept. Selecting frames is the expensive half, so a run
-- is shared and `questions` is an array -- `clip:[text,scene]` is one row here
-- and two rows in `descriptions`.
create table if not exists falconvar.chunk_samplers (
  video_id    text not null,
  chunk_id    int  not null,
  sampler_id  text not null,              -- the run: "clip" | "uniform"
  sampler     text not null,              -- the strategy
  questions   text[] not null default '{}',
  frame_count int  not null,
  frames      jsonb not null,
  primary key (video_id, chunk_id, sampler_id),
  foreign key (video_id, chunk_id) references falconvar.chunks on delete cascade
);

-- Rebuilt rather than added if absent, for the reason `fts` is below: `add
-- column if not exists` is a no-op on a table that already has the old scalar
-- `question`, so a re-run would leave the old shape in place and the writer
-- would fail on a column that is not there.
alter table falconvar.chunk_samplers drop column if exists question;
alter table falconvar.chunk_samplers
  add column if not exists questions text[] not null default '{}';

create index if not exists chunk_samplers_sampler
  on falconvar.chunk_samplers (video_id, sampler);

-- ===========================================================================
-- 6 · descriptions. NO FOREIGN KEY, deliberately. See the header.
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

create index if not exists descriptions_chunk
  on falconvar.descriptions (video_id, chunk_id);

-- ===========================================================================
-- 7 · embeddings. Also no foreign key, for the same reason: paid calls.
--
-- One vector space per EMBEDDER, never per sampler. The embedder key is in the
-- primary key, which is what stops 768-wide vectors being ranked against
-- 1536-wide ones. The sampler is payload, and querying one is a filter.
-- ===========================================================================
create table if not exists falconvar.embeddings (
  video_id    text not null,
  chunk_id    int  not null,
  sampler_id  text not null,              -- the pairing: "clip:text"
  -- The two halves as their own columns. Filtering by question is the query a
  -- person makes -- "the text on screen", not "what the CLIP sampler said" --
  -- and it is not a suffix match on `sampler_id`, because a bare id like
  -- `clip` means the question *is* the strategy name.
  sampler     text not null default '',
  question    text not null default '',
  embedder    text not null,              -- name:model:dims
  text_hash   text not null,              -- embed only what changed
  content     text not null,
  structured  jsonb not null default '{}'::jsonb,
  embedding   vector(1536) not null,
  timeline_fingerprint text,
  embedded_at timestamptz not null default now(),
  primary key (video_id, chunk_id, sampler_id, embedder)
);

-- ADDED EXPLICITLY, because `create table if not exists` above is a no-op on a
-- database that already has the table -- so a column declared only inside it is
-- never added, and the first statement to reference it fails. Same trap as
-- `fts` below and `chunk_samplers.questions` above; this file is re-run against
-- live databases, so every column added after the first deployment needs its
-- own `alter`.
alter table falconvar.embeddings
  add column if not exists sampler  text not null default '';
alter table falconvar.embeddings
  add column if not exists question text not null default '';

-- Rebuilt unconditionally rather than added if absent. `add column if not
-- exists` is a no-op when the column exists, so an `fts` built by an earlier
-- version of this file -- over `content` alone, before the structured values
-- were folded in -- survives a re-run untouched, and the lexical half quietly
-- stops indexing the terms it is best at. Nothing reports it. The column is
-- generated, so dropping it loses nothing.
alter table falconvar.embeddings drop column if exists fts;
alter table falconvar.embeddings add column fts tsvector
  generated always as (
    to_tsvector('english', content || ' ' || jsonb_path_query_array(
      structured, 'strict $.**?(@.type() == "string")')::text)
  ) stored;

-- Backfill for rows written before the two columns existed. Derivable, so no
-- vector is touched and nothing is re-embedded: `clip:text` splits, and a bare
-- id means the question is the strategy name.
update falconvar.embeddings
   set sampler  = split_part(sampler_id, ':', 1),
       question = case when position(':' in sampler_id) > 0
                       then split_part(sampler_id, ':', 2)
                       else sampler_id end
 where sampler = '' or question = '';

create index if not exists embeddings_question
  on falconvar.embeddings (video_id, embedder, question);
create index if not exists embeddings_fts on falconvar.embeddings using gin (fts);
create index if not exists embeddings_structured
  on falconvar.embeddings using gin (structured jsonb_path_ops);
create index if not exists embeddings_vector on falconvar.embeddings
  using hnsw (embedding vector_cosine_ops);

-- ===========================================================================
-- 8 · aggregates. Video-level, so not keyed by chunk at all.
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

-- NOTHING WRITES THIS YET. `units.py` builds units from descriptions and
-- transcripts only, so this table stays empty after a complete run. It is kept
-- because the shape is the decided one: a summary belongs in its own table
-- rather than in `embeddings`, which answers *which twenty seconds* where a
-- summary answers *which video* -- and a video is not a moment you can play.
-- Putting summaries in the chunk table would need a sentinel chunk_id and
-- would return a whole-video "moment" beside real ones in every search.
create table if not exists falconvar.video_embeddings (
  video_id   text not null references falconvar.videos on delete cascade,
  kind       text not null,               -- "summary"
  embedder   text not null,
  text_hash  text not null,
  content    text not null,
  embedding  vector(1536) not null,
  primary key (video_id, kind, embedder)
);

-- ===========================================================================
-- 9 · the hybrid search, in the database.
--
-- RRF over two rankings of the same rows. Cosine distance and ts_rank_cd have
-- no common scale, and any weight between them would be invented; RRF reads
-- only the orderings, so it needs no calibration. Fusing costs nothing when
-- one half has no opinion, and the half with no opinion is silent rather than
-- wrong -- which is the whole argument for fusing rather than choosing.
--
-- Ranked here rather than in Python because the alternative moves a video's
-- whole index over the wire per query.
-- ===========================================================================
-- The old signature is dropped explicitly: `create or replace` cannot change a
-- parameter list, so adding `p_question` creates a second overload and leaves
-- the previous one callable -- which would answer without the new filter and
-- look like the filter silently doing nothing.
drop function if exists falconvar.search_embeddings(
  text, vector, text, text, text, int, int);

create or replace function falconvar.search_embeddings(
  p_embedder     text,
  p_query_vector vector,
  p_query_text   text default null,
  p_video_id     text default null,
  p_sampler      text default null,
  p_question     text default null,
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
      and (p_video_id is null or e.video_id = p_video_id)
      -- A filter over one shared space, not a space of its own: every row with
      -- this embedder is comparable to every other. Narrowing to one sampler
      -- asks one question's answers rather than all of them -- and gives up the
      -- agreement signal, since a chunk can then contribute at most one term.
      and (p_sampler is null or e.sampler_id = p_sampler)
      -- The question, across whichever samplers asked it. Independent of
      -- `p_sampler`: both given narrows to one pairing, which is the same as
      -- naming the pairing outright.
      and (p_question is null or e.question = p_question)
  ),
  by_vector as (
    select c.video_id, c.chunk_id, c.sampler_id,
           row_number() over (order by c.embedding <=> p_query_vector) as rank
    from candidates c
    order by c.embedding <=> p_query_vector
    limit greatest(p_limit * 4, 40)
  ),
  -- ANY term, not every term.
  --
  -- `websearch_to_tsquery` ANDs its terms, so one word absent from the corpus
  -- silences the whole lexical half: measured here, "reactor exploded" ranked
  -- 2 rows and "the moment the reactor exploded" ranked 0, because "moment"
  -- appears in no chunk. That is the opposite of what fusing wants -- a half
  -- with a partial opinion reporting none -- and it disagrees with the other
  -- two backends, which both score any overlap.
  --
  -- Replacing `&` with `|` in the rendered tsquery keeps the parser's stemming,
  -- stopword removal and phrase handling and only loosens the conjunction.
  -- `ts_rank_cd` then does the work AND was doing badly: more overlap ranks
  -- higher, rather than partial overlap ranking nowhere.
  --
  -- A negated term (`-word` becomes `!'word'`) is the one case this loosens
  -- too far, turning an exclusion into "or anything lacking it". Rare from a
  -- search box, and RRF bounds the damage to one term of two.
  query_or as (
    select nullif(replace(
             websearch_to_tsquery('english', coalesce(p_query_text, ''))::text,
             '&', '|'), '')::tsquery as q
  ),
  by_text as (
    select c.video_id, c.chunk_id, c.sampler_id,
           row_number() over (order by ts_rank_cd(c.fts, o.q) desc) as rank
    from candidates c cross join query_or o
    where o.q is not null and c.fts @@ o.q
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
         -- The span comes from the grid, which is the one place it is stored.
         k.start_ts, k.end_ts,
         f.vector_rank::int, f.text_rank::int, f.score
  from fused f
  join candidates c
    on  c.video_id   = f.video_id
    and c.chunk_id   = f.chunk_id
    and c.sampler_id = f.sampler_id
  left join falconvar.chunks k
    on k.video_id = f.video_id and k.chunk_id = f.chunk_id
  order by f.score desc
  limit p_limit;
$$;

-- ===========================================================================
-- 10 · row level security, and grants.
--
-- Read-only for `anon`; writes need the secret key, which bypasses RLS. Run
-- this LAST: enabling RLS before the policies exist denies reads in the gap,
-- and denies them silently.
-- ===========================================================================
alter table falconvar.videos             enable row level security;
alter table falconvar.timelines          enable row level security;
alter table falconvar.chunks             enable row level security;
alter table falconvar.cuts               enable row level security;
alter table falconvar.transcripts        enable row level security;
alter table falconvar.transcript_chunks  enable row level security;
alter table falconvar.manifests          enable row level security;
alter table falconvar.chunk_samplers     enable row level security;
alter table falconvar.descriptions       enable row level security;
alter table falconvar.embeddings         enable row level security;
alter table falconvar.aggregates         enable row level security;
alter table falconvar.video_embeddings   enable row level security;

do $$
declare t text;
begin
  foreach t in array array[
    'videos','timelines','chunks','cuts','transcripts','transcript_chunks',
    'manifests','chunk_samplers','descriptions','embeddings','aggregates',
    'video_embeddings'
  ] loop
    execute format('drop policy if exists "public read" on falconvar.%I', t);
    execute format(
      'create policy "public read" on falconvar.%I for select to anon using (true)', t);
  end loop;
end $$;

grant select on all tables in schema falconvar to anon, authenticated;
grant all    on all tables in schema falconvar to service_role;
grant execute on function falconvar.search_embeddings(
  text, vector, text, text, text, text, int, int) to anon, authenticated, service_role;

-- Anything created later gets the same treatment without re-running the grants.
alter default privileges in schema falconvar
  grant select on tables to anon, authenticated;
alter default privileges in schema falconvar
  grant all on tables to service_role;

-- ===========================================================================
-- Verify. Under the PUBLISHABLE key, not this editor -- the editor runs as a
-- superuser and will show rows whether or not `anon` can.
--
--   select count(*) from falconvar.videos;          -- 0 is fine; an error is not
--   select * from falconvar.search_embeddings(
--     'openai:text-embedding-3-small:1536',
--     array_fill(0::real, array[1536])::vector, 'test');
--
-- And in the client: `python -m falconvar.rag.retrieve "..." <id> --index supabase`
-- ===========================================================================
