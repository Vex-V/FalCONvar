"""Turning a manifest into descriptions.

Ingest decides which frames are worth describing; this package reads those
frames back and records what a model says about them. It consumes ingest's
output and ingest knows nothing about it -- the same one-way boundary that
recovery/ sits behind.

The unit of work is one (chunk, sampler) pair: every frame a given sampler
kept inside one chunk, described together. A frame chosen by two samplers is
therefore described twice, deliberately -- what the person detector saw and
what the scene-change detector saw are different questions -- but it is only
ever *read* once, because the frame cache is keyed by index.
"""
