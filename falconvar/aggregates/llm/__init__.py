"""The llm tier: paid calls over the same rendering of the video.

Runs last, after `statistics` and `model`, so a run that dies here has still
produced the arithmetic.

**Ask for a word count, not "several sentences".** Once structured fields
arrived, `falconvar`'s model sized its summary as one field among many: 105
median words against 363 in the prose-only era. Saying "at least 150 words",
and that the fields *index* the summary rather than replace it, took it back to
246. The summary is the only text here that gets embedded, so its length is a
retrieval parameter rather than a style preference.

**Spans are resolved, never trusted.** A model is asked for chunk ids, which it
can copy; times it would invent.

`SYSTEM` and `BATCH` live here because all three share them; each aggregator
keeps its own response schema beside the class that uses it.
"""

from __future__ import annotations

#: How many chunk lines go into one fold. Above this, `summary` folds twice.
BATCH = 25

SYSTEM = (
    "You summarise video content for a retrieval index. Report only what the "
    "supplied descriptions say. Do not speculate about intent or identity. "
    "Every chunk id you cite must be one that appears in the input."
)

__all__ = ["BATCH", "SYSTEM"]
