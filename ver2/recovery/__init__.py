"""Rebuilding what ingestion produced, from the manifest alone.

Separate from ingest/ because it consumes the manifest rather than producing
one -- the same boundary the describer stage sits behind. It reads ingest's
output and its source video, and ingest knows nothing about it.
"""
