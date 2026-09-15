"""FalCONvar -- video in, searchable moments and higher-level answers out.

Two tiers, each with one driver over the components in its folder:

    video_rag/   extraction and search. The picture and the soundtrack read onto
                 one chunk grid, described, embedded, and queryable -- a
                 complete RAG engine on its own.
    aggregates/  answers over what video_rag extracted: counts, speakers,
                 summaries, chapters, events, entities. Never reads the video.
    shared/      what both tiers need: paths, env, document contracts, storage,
                 model providers.

`workflow.py` runs video_rag's driver, then aggregates'. Components exchange
files rather than objects, and every one is `run(video_id, ...) -> Produced`.
"""
