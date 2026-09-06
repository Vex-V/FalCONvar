"""The text-only model call, and the key both stages need.

Owned by no stage and imported by two. `describe` asks a vision model about
frames; `aggregate` asks a text model about what `describe` wrote. They differ
in what they send, not in how they authenticate or how they ask for a strict
shape, and the credential logic in particular is the kind that ends up copied
four times before anyone notices -- which is exactly what happened to the
Supabase connection before `db.py` existed.

Nothing here imports from the project, like `db` and `timeline`.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

#: The conventional name first, then the one this project's .env.example uses.
KEY_VARS = ("OPENAI_API_KEY", "OPENAI_API")

#: Text-only work is cheaper than vision work and the aggregate stage does a
#: lot of it, so this is a separate default from the describer's.
DEFAULT_MODEL = "gpt-5.4-mini"


class LLMUnavailable(Exception):
    """No key, no SDK, or an API refusal that retrying will not fix."""


def api_key(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    for name in KEY_VARS:
        value = os.environ.get(name)
        if value:
            return value
    raise LLMUnavailable(
        "no OpenAI key: set " + " or ".join(KEY_VARS) + " in .env "
        "(it is gitignored) or in the environment")


def client(explicit: Optional[str] = None) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:                          # pragma: no cover
        raise LLMUnavailable(
            "the openai package is not installed: pip install openai") from exc
    return OpenAI(api_key=api_key(explicit))


def complete(prompt: str, schema: Optional[dict[str, Any]] = None,
             model: Optional[str] = None, system: Optional[str] = None,
             max_output_tokens: int = 4000,
             api: Any = None) -> Any:
    """One text completion, optionally constrained to a strict JSON schema.

    `strict` makes the API enforce the shape, so a malformed answer is
    impossible rather than something to defend against with a parser -- the
    same reasoning as the describer's per-sampler schemas.

    Truncation is asked of the response rather than inferred from a JSON error
    further down: a cut-off answer is a valid response whose text happens to
    stop mid-string, and the aggregate schemas are verbose on a long video.
    """
    api = api or client()
    request: dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": max_output_tokens,
    }
    if system:
        request["instructions"] = system
    if schema is not None:
        request["text"] = {"format": {"type": "json_schema", "strict": True,
                                      **schema}}
    try:
        response = api.responses.create(**request)
    except Exception as exc:                            # noqa: BLE001
        raise LLMUnavailable(f"the API call failed: {exc}") from None

    if getattr(response, "status", None) == "incomplete":
        reason = getattr(getattr(response, "incomplete_details", None),
                         "reason", "unknown")
        raise LLMUnavailable(
            f"response cut short ({reason}); raise max_output_tokens above "
            f"{max_output_tokens} or reduce how much is being summarised")

    text = (getattr(response, "output_text", "") or "").strip()
    if not text:
        raise LLMUnavailable("the model returned nothing")
    if schema is None:
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMUnavailable(f"response was not JSON under a strict schema: {exc}") from None
