"""A describer that loads nothing.

Deliberately obvious in its output. `falconvar` learned why: a form defaulting
to the alphabetically-first option produced a complete-looking run whose
content was `[stub0.0][stub0.1]`, and nothing was wrong enough to report.

It still fills the *same keys* a real describer would for this question, so a
stub run exercises the shape the document has to hold rather than a simpler one.
"""

from __future__ import annotations

from typing import Any, Sequence

from .. import prompts
from ..base import Description, register
from ..frames import LoadedFrame


@register
class StubDescriber:
    name = "stub"

    def describe(self, images: Sequence[LoadedFrame],
                 context: dict[str, Any]) -> Description:
        span = f"{context['start_ts']:.1f}-{context['end_ts']:.1f}s"
        indexes = ", ".join(str(f.index) for f in images)
        question = prompts.question_for(context)
        owned = prompts.owned_by(question, context.get("chunk_questions", ()))
        return Description(
            summary=(f"[stub] {context['sampler']} chunk {context['chunk_id']} "
                     f"({span}): {len(images)} frames [{indexes}]"),
            fields={key: "" if key == "setting" else [] for key in owned},
        )

    def config(self) -> dict[str, Any]:
        return {"describer": self.name, "prompts": prompts.version()}
