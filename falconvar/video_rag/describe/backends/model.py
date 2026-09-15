"""The describer: frames in, one structured answer out, from any chat model.

A ``Describer``, so the reader neither knows nor cares which provider answers
or whether the call crosses a network. Everything it needs arrives in
``describe(images, context)``; which provider, and how that provider is asked
for a shape, is `shared/models/llm.py`'s business.

The frames are already JPEG in hand -- ``FrameStore.read_bytes`` hands them
over exactly as ingest wrote them -- so the stored bytes are sent as they are.
No decode, no resize, no re-encode. That is deliberate beyond efficiency: at
1024 px a VLM misread a burnt-in clock as 11:17:40 when it read 11:17:19 and
got it right at 1920, so the store is written at full width and nothing
downstream is allowed to quietly shrink it.

Cost is per image, and the reader calls this once per (chunk, sampler) pair
with every frame that sampler kept. On the reference run that is 10 calls
carrying 105 images for 80 distinct frames -- samplers overlap, and a frame
chosen by two of them is deliberately described twice, once per question.

The model must accept images. Nothing here can tell a vision model from a
text one before the call; the provider's refusal is passed through.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ....shared.models import llm, providers
from .. import prompts
from ..base import DescriberUnavailable, Description
from ..frames import LoadedFrame

#: Structured answers are far longer than prose: one object per person, with
#: appearance, clothing, role and action, runs several times the length of a
#: sentence naming them. At 700 the people schema truncated mid-string on a
#: busy frame and the JSON came back unparseable, so this is sized for the
#: verbose case rather than the average one.
DEFAULT_MAX_TOKENS = 2000


class ModelDescriber:
    """Describes one (chunk, sampler) pair with a single call."""

    def __init__(self, spec: Optional[str] = None,
                 max_output_tokens: int = DEFAULT_MAX_TOKENS,
                 client: Any = None) -> None:
        try:
            self.llm = llm.Model(spec, role="describe", client=client)
        except providers.ProviderError as exc:
            raise DescriberUnavailable(str(exc)) from None
        self.name = self.llm.name
        self.model = self.llm.model
        self.max_output_tokens = max_output_tokens

    @property
    def concurrency(self) -> int:
        """Runs in flight at once: the provider's cap on calls."""
        return self.llm.concurrency

    # -- request assembly, kept separate so it is testable without a network -
    def parts_for(self, images: Sequence[LoadedFrame],
                  context: dict[str, Any]) -> list[dict[str, Any]]:
        """The instruction, then every frame labelled with its own timestamp."""
        # The question, not the sampler: a positional sampler may be standing
        # in for one it is not. See prompts.question_for.
        parts = [llm.text(prompts.for_sampler(prompts.question_for(context),
                                              context, len(images)))]
        for position, frame in enumerate(images, start=1):
            parts.append(llm.text(prompts.frame_label(
                frame.index, frame.media_ts, position, len(images))))
            parts.append(llm.image(frame.jpeg))
        return parts

    async def describe(self, images: Sequence[LoadedFrame],
                       context: dict[str, Any]) -> Description:
        where = f"chunk {context['chunk_id']} / {context['sampler']}"
        if not images:
            # The pipeline guarantees every chunk keeps at least one frame, so
            # this means the manifest and the store disagree about something.
            raise DescriberUnavailable(f"no frames for {where}")
        question = prompts.question_for(context)
        try:
            # The question's own schema, not a shared one: a call about people
            # has no field to put the room in, so it cannot spend tokens
            # repeating what the scene question already said.
            payload = await self.llm.generate(
                self.parts_for(images, context),
                {"name": f"description_{question}",
                 "schema": prompts.schema_for(question)},
                prompts.SYSTEM, self.max_output_tokens)
        except llm.LLMUnavailable as exc:
            raise DescriberUnavailable(f"{self.llm.key}, {where}: {exc}") from None

        if not isinstance(payload, dict):
            raise DescriberUnavailable(f"{self.llm.key}, {where}: answer was not an object")
        summary = (payload.get("summary") or "").strip()
        if not summary:
            # An empty answer is not a description. Returning it would write a
            # blank into the document and mark the pair done.
            raise DescriberUnavailable(f"{self.llm.key}, {where}: no summary in the answer")
        return Description(summary=summary,
                           fields={k: v for k, v in payload.items() if k != "summary"})

    def config(self) -> dict[str, Any]:
        """Half of the resume key. For `openai` this is exactly the dict the
        OpenAI-only describer returned, so nothing already described becomes
        stale by the provider layer existing."""
        return {
            "name": self.name,
            "params": {
                "model": self.model,
                "max_output_tokens": self.max_output_tokens,
                # Per question, so a description says which shape produced it,
                # and how that shape was enforced.
                "response": f"{self.llm.response_mode}/description_<sampler>",
                # The prompt hashes are NOT here. They are per question, and
                # which questions a run asks is a property of the manifest
                # rather than of the describer -- `reader` owns that half.
            },
        }


__all__ = ["DEFAULT_MAX_TOKENS", "ModelDescriber"]
