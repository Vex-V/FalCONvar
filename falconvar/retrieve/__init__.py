"""Finding the moment, given a question.

The stage that makes the rest of it worth building. Ingest decides which
frames matter, describe says what is in them, embed makes those descriptions
comparable to a question -- and this asks the question.

What an index ranks is **descriptions**. What a person wants back is a
**moment**: a window of media time with an in point, an out point, and the
exact frame indexes to show as evidence. Those are different units, and
closing the gap between them is the whole of what this module does.

It reaches into `embed` for two things and no more: the embedder registry,
because a query embedded by a different model than the descriptions produces a
ranking that looks fine and means nothing; and the index, because that is
where the vectors are. How a unit is assembled, what it hashes to, and when it
is re-embedded are embed's business and do not survive into a search.
"""
