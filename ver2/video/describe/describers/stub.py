"""A describer that needs no model.

Deterministic on purpose: two runs over the same manifest produce byte-identical
output, which is what makes the resume path testable. It stays useful after a
real model exists, as the way to exercise frame loading, both sinks, resume and
following without paying for inference.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..input.frames import LoadedFrame
from ..vlm import prompts
from .base import Description


class StubDescriber:
    """Deterministic text, no model, no network. A ``Describer``.

    Deterministic on purpose: two runs over the same manifest must produce
    byte-identical output, which is what makes the resume path testable.
    """

    name = "stub"

    def describe(self, images: Sequence[LoadedFrame],
                 context: dict[str, Any]) -> Description:
        span = f"{context['start_ts']:.1f}-{context['end_ts']:.1f}s"
        indexes = ", ".join(str(f.index) for f in images)
        return Description(
            summary=(f"[stub] {context['sampler']} chunk {context['chunk_id']} "
                     f"({span}): {len(images)} frames [{indexes}]"),
            # The same keys a real describer would fill for this sampler, so
            # a stub run exercises the shape the document has to hold.
            fields={key: "" if key == "setting" else []
                    for key in prompts.owned_by(
                        prompts.question_for(context),
                        context.get("chunk_questions", ()))},
        )

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "params": {}}
