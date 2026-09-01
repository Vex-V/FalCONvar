"""What a describer is.

The real describer is a VLM and does not exist yet. Everything around it does,
so the interface is fixed here and `stub.py` implements it: the reader can be
built, run and verified end to end before a model is chosen.

``context`` carries what the model should not have to re-derive -- which video,
which chunk, which sampler and that sampler's own settings. A person
detector's picks and a scene-change detector's picks are answers to different
questions, so a prompt that ignores which sampler chose the frames is throwing
away the most useful thing the manifest knows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from ..input.frames import LoadedFrame


@dataclass
class Description:
    """What a describer returns: prose, plus the structure behind it.

    Two parts because they are consumed by different things. ``summary`` is the
    paragraph -- it is what gets embedded, what full-text search indexes, and
    what a person reads. ``fields`` is the same observation broken out, and it
    is what a filter can use: every moment where a shopping basket was visible,
    every moment where a sign said something.

    Keeping the prose as its own field rather than reassembling it from the
    parts is deliberate. A summary written as prose by the model reads better
    than one stitched together from lists, and retrieval quality depends on the
    text that gets embedded being good text.
    """

    summary: str
    fields: dict[str, Any] = field(default_factory=dict)


class Describer(Protocol):
    def describe(self, images: Sequence[LoadedFrame],
                 context: dict[str, Any]) -> Description:
        """One description for these frames, taken together."""
        ...

    def config(self) -> dict[str, Any]:
        """Recorded alongside the output, so a description says what produced it."""
        ...
