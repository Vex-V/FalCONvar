"""The OpenAI call: frames in, one description out.

A ``Describer``, so the reader neither knows nor cares that this one crosses a
network. Everything it needs arrives in ``describe(images, context)``; what it
returns is text.

The frames are already JPEG in hand -- ``FrameStore.read_bytes`` hands them
over exactly as ingest wrote them -- so this base64s the stored bytes and sends
them. No decode, no resize, no re-encode. That is deliberate beyond mere
efficiency: at 1024 px a VLM misread a burnt-in clock as 11:17:40 when it read
11:17:19 and got it right at 1920, so the store is written at full width and
nothing downstream is allowed to quietly shrink it.

Cost is per image, and the reader will call this once per (chunk, sampler)
pair with every frame that sampler kept. On the reference run that is 10 calls
carrying 105 images for 80 distinct frames -- samplers overlap, and a frame
chosen by two of them is deliberately described twice, once per question.
Check the numbers a run reports before pointing it at a long video.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Optional, Sequence

from ..describers.base import Description
from ..input.frames import LoadedFrame
from . import prompts

#: What the caller asked for. Model ids change faster than this file will, so
#: it is a plain default rather than a validated constant -- pass --model to
#: use another, and the API is the authority on whether it exists.
DEFAULT_MODEL = "gpt-5.4-mini"
#: Structured answers are far longer than prose: one object per person, with
#: appearance, clothing, role and action, runs several times the length of a
#: sentence naming them. At 700 the people schema truncated mid-string on a
#: busy frame and the JSON came back unparseable, so this is sized for the
#: verbose case rather than the average one.
DEFAULT_MAX_TOKENS = 2000

#: The conventional name first, then the one this project's .env.example uses.
KEY_VARS = ("OPENAI_API_KEY", "OPENAI_API")


class DescriberUnavailable(Exception):
    """No key, no SDK, or the API refused in a way retrying will not fix."""


def _api_key(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    for name in KEY_VARS:
        value = os.environ.get(name)
        if value:
            return value
    raise DescriberUnavailable(
        "no OpenAI key: set " + " or ".join(KEY_VARS) + " in .env "
        "(it is gitignored) or in the environment"
    )


class OpenAIDescriber:
    """Describes one (chunk, sampler) pair with a single vision call."""

    name = "openai"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_output_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: Optional[str] = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:            # pragma: no cover
                raise DescriberUnavailable(
                    "the openai package is not installed: pip install openai"
                ) from exc
            client = OpenAI(api_key=_api_key(api_key))
        self.client = client

    # -- request assembly, kept separate so it is testable without a network -
    def content_for(
        self, images: Sequence[LoadedFrame], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """The instruction, then every frame labelled with its own timestamp."""
        parts: list[dict[str, Any]] = [{
            "type": "input_text",
            # The question, not the sampler: a positional sampler may be
            # standing in for one it is not. See prompts.question_for.
            "text": prompts.for_sampler(prompts.question_for(context), context,
                                        len(images)),
        }]
        for position, frame in enumerate(images, start=1):
            parts.append({
                "type": "input_text",
                "text": prompts.frame_label(frame.index, frame.media_ts,
                                            position, len(images)),
            })
            encoded = base64.b64encode(frame.jpeg).decode("ascii")
            parts.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encoded}",
            })
        return parts

    def describe(self, images: Sequence[LoadedFrame],
                 context: dict[str, Any]) -> Description:
        if not images:
            # The pipeline guarantees every chunk keeps at least one frame, so
            # this means the manifest and the store disagree about something.
            raise DescriberUnavailable(
                f"no frames for chunk {context['chunk_id']} / {context['sampler']}"
            )
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=prompts.SYSTEM,
                input=[{"role": "user", "content": self.content_for(images, context)}],
                max_output_tokens=self.max_output_tokens,
                # Structured, not prose-and-hope. `strict` makes the API itself
                # enforce the shape, so a malformed answer is impossible rather
                # than something to defend against with a parser.
                # The sampler's own schema, not a shared one: a call about
                # people has no field to put the room in, so it cannot spend
                # tokens repeating what the scene sampler already said.
                text={"format": {
                    "type": "json_schema",
                    "name": f"description_{prompts.question_for(context)}",
                    "strict": True,
                    # Resolved the same way as the instruction, so the shape
                    # asked for and the question asked always agree.
                    "schema": prompts.schema_for(
                        prompts.question_for(context),
                        context.get("chunk_samplers", ())),
                }},
            )
        except Exception as exc:                   # noqa: BLE001
            message = str(exc)
            if "model" in message.lower() and "not" in message.lower():
                raise DescriberUnavailable(
                    f"the API rejected model {self.model!r}: {message}\n"
                    "Pass --model with an id this account can use."
                ) from None
            raise DescriberUnavailable(
                f"OpenAI call failed for chunk {context['chunk_id']} / "
                f"{context['sampler']}: {message}"
            ) from None

        # Truncation is the failure this schema invites, and it does not look
        # like a failure: the API returns a valid response whose text happens
        # to stop mid-string. Ask the response itself rather than inferring it
        # from a JSON error further down.
        if getattr(response, "status", None) == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None),
                             "reason", "unknown")
            raise DescriberUnavailable(
                f"response cut short for chunk {context['chunk_id']} / "
                f"{context['sampler']} ({reason}). The structured schemas are "
                f"verbose on busy frames; raise --max-output-tokens above "
                f"{self.max_output_tokens}."
            )

        raw = getattr(response, "output_text", None)
        if not raw or not raw.strip():
            # An empty answer is not a description. Returning it would write a
            # blank into the document and mark the pair done, so it stops here.
            raise DescriberUnavailable(
                f"empty response for chunk {context['chunk_id']} / "
                f"{context['sampler']} (model {self.model})"
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            # Should not happen under strict schema; if it ever does, the run
            # stops rather than storing a JSON string as if it were prose.
            raise DescriberUnavailable(
                f"response was not JSON for chunk {context['chunk_id']} / "
                f"{context['sampler']}: {exc}"
            ) from None

        summary = (payload.get("summary") or "").strip()
        if not summary:
            raise DescriberUnavailable(
                f"no summary in response for chunk {context['chunk_id']} / "
                f"{context['sampler']} (model {self.model})"
            )
        return Description(
            summary=summary,
            fields={k: v for k, v in payload.items() if k != "summary"},
        )

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": {
                "model": self.model,
                "max_output_tokens": self.max_output_tokens,
                # Per sampler, so a description says which shape produced it.
                "response": "json_schema/description_<sampler>",
                # Editing a prompt changes the output, so it has to change the
                # resume key too, or a re-run silently keeps the old answers.
                "prompts": prompts.version(),
            },
        }
