"""Finding the moment, given a question.

The stage that makes the rest of it worth building. Ingest decides which frames
matter, describe says what is in them, and this turns those descriptions into
something a question can be asked of.

What it indexes is one description -- one (chunk, sampler) pair -- because that
is the unit a describer call produced and the unit whose text answers a
particular kind of question. What it *returns* is a chunk: a window of media
time with an in and an out point, and the exact frame indexes to show as
evidence. The two are not the same thing, and the gap between them is where
the ranking happens.

Embeddings are derived, never authoritative. The descriptions they come from
are the record; an index is a cache that can be dropped and rebuilt, which is
why every vector carries a hash of the text it was made from.
"""
