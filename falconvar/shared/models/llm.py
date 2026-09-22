"""Asking a model for text or a strict JSON shape, whoever serves it.

Owned by no stage and imported by two. `describe` sends frames; `aggregate`
sends text. Both want an answer in a fixed shape, and both reach it through
`Model` -- so a provider is added once, here, rather than once per stage.

What differs between providers is the wire format, and that is all this owns:

    openai     Responses API, `text.format` json_schema with `strict`
    chat       Chat Completions. `response_format` json_schema by default;
               json_object, or the schema in the prompt, where a server has no
               schema support (`Provider.structured`)
    anthropic  Messages API, the schema as the input of a forced tool

**Async, because the calls are independent and the wait is the network.**
`generate` is a coroutine. A stage with many calls gathers them, and
`Provider.concurrency` caps how many are in flight at once -- per model, per
event loop. A stage with one call awaits it under `asyncio.run`.

**A client per call, not per model.** An async client's connections belong to
the event loop that opened them, and each stage runs its own loop: a client
kept on the model outlives the first loop and fails when the second reuses it.
A fresh client's handshake is small beside a call that takes seconds.

**The OpenAI request is byte-for-byte the one this module sent before it knew
any other provider.** Resume in `describe` is keyed on the describer's config,
and every description already paid for was made on this path -- so the
native request stayed native rather than being folded into Chat Completions,
which would have been one code path and a reason to re-describe everything.

Truncation is asked of the response rather than inferred from a JSON error
further down: a cut-off answer is a valid response whose text happens to stop
mid-string, and the structured schemas are verbose on busy frames.
"""

from __future__ import annotations

import asyncio
import base64
import json
import weakref
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, Sequence

from . import providers as providers_mod
from ..errors import Unavailable

#: Retries on the Anthropic path for a rate limit or an overloaded server. The
#: OpenAI SDK already retries these inside a call; the Messages path is plain
#: httpx, and concurrent calls are exactly what meets a 429.
RETRIES = 2
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})


class LLMUnavailable(Unavailable):
    """No key, or an API refusal that retrying will not fix."""


def text(value: str) -> dict[str, Any]:
    """A part of a request: some text."""
    return {"type": "text", "text": value}


def image(data: bytes, mime: str = "image/jpeg") -> dict[str, Any]:
    """A part of a request: an image, as the bytes the store already holds."""
    return {"type": "image", "data": data, "mime": mime}


def _data_uri(part: dict[str, Any]) -> str:
    return f"data:{part['mime']};base64,{base64.b64encode(part['data']).decode('ascii')}"


def parse_json(raw: str) -> Any:
    """JSON out of a model's text, tolerating the fences a prompt-only answer
    tends to arrive in. Under a server-enforced schema the first `loads` is the
    only one that ever runs."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    body = raw.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = body.find("{"), body.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(body[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"response was not JSON: {exc}") from None
    raise LLMUnavailable("response was not JSON: no object in it")


def _schema_prompt(schema: dict[str, Any]) -> str:
    return ("Answer with a single JSON object and nothing else -- no prose, no "
            "code fence. It must match this JSON Schema exactly:\n"
            + json.dumps(schema["schema"], separators=(",", ":")))


def _retry_after(response: Any, attempt: int) -> float:
    """The server's `retry-after` when it sends one, else 1 s then 2 s."""
    try:
        return min(60.0, max(0.0, float(response.headers.get("retry-after", ""))))
    except ValueError:
        return float(2 ** attempt)


class Model:
    """One provider and one model: the thing a stage asks.

    Resolved on construction, so an unknown provider fails before any work.
    The key is read on the first call, so an aggregator that turns out to be
    skipped never needed one.
    """

    def __init__(self, spec: Optional[str] = None, role: str = "llm",
                 client: Any = None) -> None:
        name, chosen = providers_mod.choose(role, spec)
        if name in providers_mod.OFFLINE.values():
            raise providers_mod.ProviderError(f"{name!r} is not a model provider")
        self.provider = providers_mod.get(name)
        if not self.provider.can_chat:
            raise providers_mod.ProviderError(f"{name} serves no answers")
        self.model: str = chosen or ""
        #: Calls in flight at once, from the provider.
        self.concurrency: int = self.provider.concurrency
        self._client = client
        self._gates: "weakref.WeakKeyDictionary[Any, asyncio.Semaphore]" = (
            weakref.WeakKeyDictionary())

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def key(self) -> str:
        """`provider:model`. What an aggregate records it was made by."""
        return f"{self.name}:{self.model}"

    @property
    def response_mode(self) -> str:
        """How the shape is enforced -- part of a describer's resume key."""
        if self.provider.protocol == "chat":
            return self.provider.structured
        return "tool" if self.provider.protocol == "anthropic" else "json_schema"

    # -- the one entry point -------------------------------------------------
    async def generate(self, parts: Sequence[dict[str, Any]],
                       schema: Optional[dict[str, Any]] = None,
                       system: Optional[str] = None,
                       max_output_tokens: int = 4000) -> Any:
        """Text, or a parsed object when `schema` (`{name, schema}`) is given."""
        protocol = self.provider.protocol
        call = {"openai": self._responses, "chat": self._chat,
                "anthropic": self._anthropic}.get(protocol)
        if call is None:
            raise LLMUnavailable(f"{self.name} ({protocol}) serves no answers")
        async with self._gate():
            try:
                return await call(parts, schema, system, max_output_tokens)
            except providers_mod.ProviderUnavailable as exc:
                raise LLMUnavailable(str(exc)) from None

    async def complete(self, prompt: str, schema: Optional[dict[str, Any]] = None,
                       system: Optional[str] = None,
                       max_output_tokens: int = 4000) -> Any:
        return await self.generate([text(prompt)], schema, system, max_output_tokens)

    # -- shared --------------------------------------------------------------
    def _gate(self) -> asyncio.Semaphore:
        """This loop's cap on calls in flight. One per loop, because a
        semaphore belongs to the loop that first waits on it."""
        loop = asyncio.get_running_loop()
        if loop not in self._gates:
            self._gates[loop] = asyncio.Semaphore(max(1, self.concurrency))
        return self._gates[loop]

    @asynccontextmanager
    async def _openai(self) -> AsyncIterator[Any]:
        if self._client is not None:
            yield self._client
            return
        from openai import AsyncOpenAI
        key = providers_mod.api_key(self.provider)
        options: dict[str, Any] = {
            # The SDK refuses an empty key even for a server that ignores it.
            "api_key": key or "not-needed"}
        url = providers_mod.base_url(self.provider)
        if url:
            options["base_url"] = url
        async with AsyncOpenAI(**options) as client:
            yield client

    @asynccontextmanager
    async def _http(self) -> AsyncIterator[Any]:
        if self._client is not None:
            yield self._client
            return
        import httpx
        async with httpx.AsyncClient(timeout=600) as client:
            yield client

    def _failed(self, exc: Exception) -> LLMUnavailable:
        message = str(exc)
        lowered = message.lower()
        if "connect" in lowered and self.provider.local:
            return LLMUnavailable(
                f"could not reach {self.name} at {providers_mod.base_url(self.provider)} "
                f"-- is it running? ({message})")
        if "model" in lowered and ("not" in lowered or "unknown" in lowered):
            return LLMUnavailable(
                f"{self.name} rejected model {self.model!r}: {message}. "
                "Name a model this provider serves.")
        return LLMUnavailable(f"the {self.name} call failed: {message}")

    @staticmethod
    def _cut_short(reason: Any, limit: int) -> LLMUnavailable:
        return LLMUnavailable(
            f"response cut short ({reason}); raise max_output_tokens above "
            f"{limit} or reduce how much is being sent")

    def _finish(self, raw: Optional[str], schema: Optional[dict[str, Any]]) -> Any:
        body = (raw or "").strip()
        if not body:
            raise LLMUnavailable(f"{self.key} returned nothing")
        return body if schema is None else parse_json(body)

    # -- openai: Responses ---------------------------------------------------
    async def _responses(self, parts, schema, system, limit) -> Any:
        if len(parts) == 1 and parts[0]["type"] == "text":
            # A bare string, as a text-only request has always been sent.
            content: Any = parts[0]["text"]
        else:
            content = [{"type": "input_text", "text": p["text"]} if p["type"] == "text"
                       else {"type": "input_image", "image_url": _data_uri(p)}
                       for p in parts]
        request: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": limit,
        }
        if system:
            request["instructions"] = system
        if schema is not None:
            request["text"] = {"format": {"type": "json_schema", "strict": True,
                                          **schema}}
        async with self._openai() as client:
            try:
                response = await client.responses.create(**request)
            except Exception as exc:                    # noqa: BLE001
                raise self._failed(exc) from None
        if getattr(response, "status", None) == "incomplete":
            raise self._cut_short(getattr(getattr(response, "incomplete_details", None),
                                          "reason", "unknown"), limit)
        return self._finish(getattr(response, "output_text", ""), schema)

    # -- chat: Chat Completions ----------------------------------------------
    async def _chat(self, parts, schema, system, limit) -> Any:
        mode = self.provider.structured if schema is not None else None
        parts = list(parts)
        if mode in ("json_object", "prompt"):
            parts.append(text(_schema_prompt(schema)))

        if all(p["type"] == "text" for p in parts):
            content: Any = "\n\n".join(p["text"] for p in parts)
        else:
            content = [{"type": "text", "text": p["text"]} if p["type"] == "text"
                       else {"type": "image_url", "image_url": {"url": _data_uri(p)}}
                       for p in parts]
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        request: dict[str, Any] = {"model": self.model, "messages": messages}
        if mode == "json_schema":
            request["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema["name"], "strict": True, "schema": schema["schema"]}}
        elif mode == "json_object":
            request["response_format"] = {"type": "json_object"}

        field = self.provider.token_field
        async with self._openai() as client:
            try:
                response = await client.chat.completions.create(**request, **{field: limit})
            except Exception as exc:                    # noqa: BLE001
                other = ("max_completion_tokens" if field == "max_tokens" else "max_tokens")
                if field not in str(exc):
                    raise self._failed(exc) from None
                try:
                    response = await client.chat.completions.create(**request,
                                                                    **{other: limit})
                except Exception as again:              # noqa: BLE001
                    raise self._failed(again) from None

        if not getattr(response, "choices", None):
            raise LLMUnavailable(f"{self.key} returned no choices")
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise self._cut_short("length", limit)
        message = choice.message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise LLMUnavailable(f"{self.key} refused: {refusal}")
        return self._finish(getattr(message, "content", None), schema)

    # -- anthropic: Messages -------------------------------------------------
    async def _anthropic(self, parts, schema, system, limit) -> Any:
        key = providers_mod.api_key(self.provider)
        content = [{"type": "text", "text": p["text"]} if p["type"] == "text"
                   else {"type": "image", "source": {
                       "type": "base64", "media_type": p["mime"],
                       "data": base64.b64encode(p["data"]).decode("ascii")}}
                   for p in parts]
        body: dict[str, Any] = {"model": self.model, "max_tokens": limit,
                                "messages": [{"role": "user", "content": content}]}
        if system:
            body["system"] = system
        if schema is not None:
            # A forced tool whose input IS the schema. The model has to call it,
            # and its arguments come back as an object -- no prose to parse.
            body["tools"] = [{"name": schema["name"],
                              "description": "Record the answer in exactly this shape.",
                              "input_schema": schema["schema"]}]
            body["tool_choice"] = {"type": "tool", "name": schema["name"]}
        headers = {"anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        if key:
            headers["x-api-key"] = key
        url = (providers_mod.base_url(self.provider) or "").rstrip("/") + "/v1/messages"

        async with self._http() as http:
            for attempt in range(RETRIES + 1):
                try:
                    response = await http.post(url, json=body, headers=headers,
                                               timeout=600)
                except Exception as exc:                # noqa: BLE001
                    raise self._failed(exc) from None
                if response.status_code not in RETRY_STATUS or attempt == RETRIES:
                    break
                await asyncio.sleep(_retry_after(response, attempt))
        if response.status_code >= 400:
            try:
                detail = response.json().get("error", {}).get("message") or response.text
            except ValueError:
                detail = response.text
            raise self._failed(RuntimeError(f"{response.status_code}: {detail}"))

        data = response.json()
        if data.get("stop_reason") == "max_tokens":
            raise self._cut_short("max_tokens", limit)
        blocks = data.get("content") or []
        if schema is not None:
            for block in blocks:
                if block.get("type") == "tool_use":
                    return block.get("input") or {}
            raise LLMUnavailable(f"{self.key} answered without the {schema['name']} tool")
        return self._finish("".join(b.get("text", "") for b in blocks
                                    if b.get("type") == "text"), None)


__all__ = ["LLMUnavailable", "Model", "image", "parse_json", "text"]
