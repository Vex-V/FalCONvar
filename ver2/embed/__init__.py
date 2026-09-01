"""Descriptions in, vectors out.

The stage between describe and retrieve. What it embeds is one description --
one ``(chunk, sampler)`` pair -- because that is the unit a describer call
produced, and the unit whose text answers one particular kind of question.
Merging a chunk's descriptions into a single vector would average away the
reason there is more than one of them.

Nothing this stage writes is a record. Embeddings are derived: the
descriptions they come from are authoritative, and an index is a cache that
can be dropped and rebuilt from them. That is why every vector carries a hash
of the text it was made from -- a stale vector has to be detectable rather
than merely regrettable.

It owns the index the way ingest owns the frame store: the stage that writes a
store is the stage that defines it. `retrieve` reads this one, and reaches for
exactly two things -- an ``Embedder``, because a question must be embedded by
the same model as the descriptions or the vectors are not comparable, and a
``VectorIndex``, because that is where they were put.
"""
