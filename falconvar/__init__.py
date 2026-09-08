"""FalCONvar -- video RAG ingestion.

A video goes in; both the picture and the soundtrack are read onto one chunk
grid, and a searchable index of moments comes out.

Nine components, each `run(video_id, ...) -> Produced`, exchanging files rather
than objects: `media`, `audio`, `boundaries`, `video`, `cut`, `describe`,
`embed`, `aggregate`, and `retrieve` over what they built. `workflow.py` calls
them in the order the chosen grid policy implies.
"""
