"""Video: pixels in, descriptions out.

Two stages that only ever speak through what they wrote down. `ingest` decides
which frames are worth describing and records *why* each was kept; `describe`
reads those frames back out of the store and asks a VLM one question per
sampler, because the reason a frame was kept is the most useful thing the
manifest knows.

They are grouped here because they share a subject -- the video stream -- not
because they share code. `describe` imports exactly one name from `ingest`
(`FrameStore`) and `imports.py` enforces that by AST-parsing every file below
this line.

`embed` and `retrieve` are deliberately NOT here. They consume text with a
time span, and a transcript segment is that as much as a description is, so
they are shared with `audio/` rather than owned by either.
"""
