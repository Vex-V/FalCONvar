"""Sampler contract.

A sampler sees the decimated stream one frame at a time, with pixels still
attached, and answers yes or no. It cannot look ahead, cannot revisit a frame
it declined, and cannot buffer the chunk -- the same constraints a live source
imposes, so the file path and the stream path behave identically.

Two rate constraints apply to *every* sampler, enforced here rather than
reimplemented per strategy:

  ``min_interval_s``  smallest gap allowed between two kept frames. 5 means
                      kept frames are at least 5 seconds apart. It is a
                      minimum *interval*, i.e. a maximum frequency -- worth
                      keeping the inversion straight.
  ``max_per_chunk``   hard ceiling on frames kept from one chunk.

Both short-circuit *before* the strategy runs, so a rate-limited frame costs
no model inference -- which is the difference between a cap that saves money
and one that merely reduces output.

Every chunk keeps at least one frame: the first frame offered is kept whatever
the strategy thinks of it, so no chunk is left without something to describe.

A sampler also carries a ``prompt``: the question the describe stage should put
to the frames it kept. It is *not* the sampler's own business what that question
is -- this class never reads it -- but it is the sampler's business to record
it, because the manifest is where describe finds out. Every sampler takes one,
so any strategy can be paired with any question: `yolo:overview` keeps frames
where the people changed and asks for prose rather than the structured people
call. Left unset the question is the sampler's own name, which is the default
worth having: the manifest records *why* a frame was kept, and why it was kept
is the best guide to what to ask about it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..source import Frame


class Sampler(ABC):
    """Base class for frame selection strategies.

    Subclasses implement :meth:`propose`. The rate limits, the per-chunk cap
    and the one-frame-per-chunk guarantee live here so they behave identically
    no matter which strategy is running.
    """

    name: str = "base"

    def __init__(
        self,
        min_interval_s: float = 0.0,
        max_per_chunk: Optional[int] = None,
        sampler_id: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> None:
        if min_interval_s < 0:
            raise ValueError("min_interval_s must be >= 0")
        if max_per_chunk is not None and max_per_chunk < 1:
            raise ValueError("max_per_chunk must be >= 1; every chunk keeps a frame")
        self.min_interval_s = min_interval_s
        self.max_per_chunk = max_per_chunk
        # The manifest keys frames by sampler, so two samplers in one run need
        # distinct ids. ``name`` identifies the strategy and is enough when
        # each appears once; running the same strategy twice with different
        # settings needs an explicit id to tell the results apart.
        self._sampler_id = sampler_id
        # Which question these frames are for. Opaque here: the names live in
        # `describe/vlm/prompts.py` and are validated by whoever built this,
        # because a sampler knowing the prompt registry would be an edge from
        # ingest to describe that the module graph does not have.
        self.prompt = prompt
        self._kept_in_chunk = 0
        self._last_kept_ts: Optional[float] = None

    @property
    def sampler_id(self) -> str:
        """The manifest key, and the key every later stage inherits from it.

        A pairing is keyed by both halves -- `yolo:overview` -- because the two
        facts are independent and the id has to survive either changing. Keying
        by the question alone was the earlier rule, and it stops working the
        moment any sampler can ask any question: `yolo:overview` and
        `clip:overview` would both be `overview`, colliding on a key while
        holding different frames. Keying by the strategy alone loses the
        question, which is the more useful half for reading a search result.

        An unmodified sampler keeps its bare name, so nothing that does not
        pair a question changes.
        """
        if self._sampler_id:
            return self._sampler_id
        if self.prompt and self.prompt != self.name:
            return f"{self.name}:{self.prompt}"
        return self.name

    def reset(self, chunk_id: int) -> None:
        """Called once when a new chunk opens, before any frame is offered."""
        self._kept_in_chunk = 0
        self._last_kept_ts = None
        self.on_reset(chunk_id)

    def on_reset(self, chunk_id: int) -> None:
        """Subclass hook for clearing strategy state at a chunk boundary."""

    def accepts(self, frame: Frame, chunk_local_index: int) -> bool:
        """Final decision for one frame. Do not override -- implement propose."""
        if self.max_per_chunk is not None and self._kept_in_chunk >= self.max_per_chunk:
            # Chunk is full. Skipping here rather than inside the strategy is
            # what makes the cap free instead of merely quiet.
            return False

        first_of_chunk = self._last_kept_ts is None
        if (
            not first_of_chunk
            and frame.media_ts - self._last_kept_ts < self.min_interval_s
        ):
            return False

        # The strategy still sees every frame it is allowed to see, so its own
        # state stays coherent; the guarantee is layered on top.
        keep = self.propose(frame, chunk_local_index) or first_of_chunk

        if keep:
            self._kept_in_chunk += 1
            self._last_kept_ts = frame.media_ts
        return keep

    # ---- the two halves of a change sampler, split so a calibrator can cache
    # the expensive half and replay the cheap one at many thresholds. Running a
    # model once per frame and comparing thousands of times is what makes a
    # threshold sweep affordable.

    def describe(self, frame: Frame):
        """The model's output for this frame: an embedding, a detection set.

        Returns None for samplers that decide on position rather than content.
        """
        return None

    def compare(self, current, reference) -> Optional[float]:
        """Similarity in [0, 1] between two descriptions. Lower means changed."""
        return None

    @abstractmethod
    def propose(self, frame: Frame, chunk_local_index: int) -> bool:
        """The strategy's opinion, before rate limits and the chunk guarantee.

        ``chunk_local_index`` counts decimated frames in the current chunk,
        starting at 0.
        """

    def last_score(self) -> Optional[float]:
        """Whatever the last decision was based on, for the manifest.

        Content-driven samplers record their similarity here so a threshold
        can be retuned by reading a run's output instead of rerunning a model
        over the video. Positional samplers have nothing to record.
        """
        return None

    def _base_config(self) -> dict:
        # `prompt` is omitted when unset rather than written as null, so a
        # sampler that pairs no question produces exactly the config it always
        # did -- and `manifest_fingerprint`, which is a hash of this, does not
        # move for every video already ingested.
        config = {
            "id": self.sampler_id,
            "name": self.name,
            "min_interval_s": self.min_interval_s,
            "max_per_chunk": self.max_per_chunk,
        }
        if self.prompt is not None:
            config["prompt"] = self.prompt
        return config

    def config(self) -> dict:
        """Serialised into the manifest so a run can be reproduced."""
        return self._base_config()
