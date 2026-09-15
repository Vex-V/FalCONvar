"""video_rag -- a video in, a searchable index of moments out, and the search.

The extraction half of FalCONvar, and a complete RAG engine on its own:

    media        1  what streams the file carries
    audio        2  the soundtrack, scanned whole
    boundaries 3+4  THE GRID, from the picture or the soundtrack
    video        5  which frames each sampler keeps
    cut          6  the transcript, onto the grid
    describe     7  one model answer per (chunk, sampler:question)
    embed        8  both modalities to vectors, keyed by a hash of the text
    retrieve        a query to ranked moments over what embed built

`driver.py` runs them in the order the grid policy implies, the way
`workflow.py` runs this driver and `aggregates`'. Nothing in this package reads
anything `aggregates` produced.
"""
