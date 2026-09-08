-- Drop everything the pipeline owns, then re-run install.sql.
--
-- Separate from install.sql because install.sql is safe to re-run against a
-- live database and this is not: it destroys every row, including descriptions
-- and embeddings that cost money to produce. Nothing calls it; it is run by
-- hand, deliberately.
--
-- The later statements remove what earlier versions left behind: a `ver3`
-- schema from before the rename, and the tables the pipeline before that wrote
-- into `public`. Nothing writes to either now. Harmless on a database that
-- never had them.

drop schema if exists falconvar cascade;

drop schema if exists ver3 cascade;

drop table if exists
  public.video_manifests, public.video_chunks, public.video_descriptions,
  public.chunk_embeddings, public.audio_transcripts, public.audio_chunks,
  public.video_aggregates, public.video_embeddings
cascade;

drop function if exists public.search_embeddings(
  text, vector, text, text, text, int, int);
