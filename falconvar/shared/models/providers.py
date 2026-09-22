"""Which model answers a call, and how to reach it.

Three stages ask a model for something: `describe` sends frames and wants a
structured answer, `aggregate` sends text and wants one, and `embed` and
`retrieve` want vectors. This is the one place that knows who can serve those,
so a stage names a provider and a model and nothing else.

A provider is a name and a **protocol** -- the wire format, which is the only
thing that really differs between them:

    openai     OpenAI's own API: Responses for answers, /embeddings for vectors
    chat       anything serving OpenAI's Chat Completions and /embeddings --
               Ollama, LM Studio, llama.cpp, vLLM, Gemini, Mistral, Groq,
               OpenRouter, Together, DeepSeek, xAI, Voyage
    anthropic  the Messages API. Answers only: Anthropic serves no vectors
    local      a Hugging Face model loaded into this process. Vectors only

Built-ins are declared below. `data/providers.json` adds an endpoint -- a vLLM
box, a company gateway -- or overrides a built-in's fields. Keys never live in
that file: it names the environment variables to read them from.

**A default is resolved when a call is made, never at import.** `.env` is read
at the top of an entry point, later than module constants resolve, so a default
captured in a constant would depend on import order. Each role reads its
variable at the moment it needs it.

**A choice is one string.** `ollama/gemma3:4b` names the provider and the model
together -- in a flag, a form field or `.env` alike -- so no surface carries a
second field for the model, and nothing has to decide which of two fields wins.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, fields, replace
from typing import Any, Optional

from .. import env, paths
from ..errors import FalconvarError, Unavailable

PROTOCOLS = ("openai", "chat", "anthropic", "local")
ROLES = ("describe", "llm", "embed")

#: How a `chat` provider is asked for a shape. `json_schema` is enforced by the
#: server; the other two put the schema in the prompt and parse what comes back.
STRUCTURED = ("json_schema", "json_object", "prompt")

#: What a role uses when neither the call nor the environment names a provider.
FALLBACK = "openai"

#: role -> the variable naming its provider, or `provider/model`.
ENV: dict[str, str] = {
    "describe": "FALCONVAR_DESCRIBER",
    "llm": "FALCONVAR_LLM",
    "embed": "FALCONVAR_EMBEDDER",
}

#: Names that answer a role without being a provider: they load nothing and
#: call nothing, so they have no model and need no key.
OFFLINE: dict[str, str] = {"describe": "stub", "embed": "hash"}

NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class ProviderError(FalconvarError, ValueError):
    """An unknown provider, one that cannot do what was asked, or no model."""


class ProviderUnavailable(Unavailable):
    """A provider that needs a key, and none is set."""


@dataclass(frozen=True)
class Provider:
    name: str
    protocol: str
    base_url: Optional[str] = None
    #: Variables the key is read from, first set wins. Names, never values.
    key_vars: tuple[str, ...] = ()
    chat_model: Optional[str] = None
    embed_model: Optional[str] = None
    can_chat: bool = True
    can_embed: bool = True
    structured: str = "json_schema"
    #: Accepts `dimensions` on /embeddings, so a width can be asked for rather
    #: than only checked.
    dimensions: bool = False
    #: Wants `input_type: query | document` on /embeddings (Voyage).
    input_type: bool = False
    #: `max_tokens` or `max_completion_tokens`. OpenAI's reasoning models
    #: refuse the first and older servers do not know the second, so a refusal
    #: naming the field is retried once with the other.
    token_field: str = "max_tokens"
    #: Calls in flight at once. A cloud API takes many; a server on this
    #: machine usually answers one at a time, and a queue there only adds
    #: timeouts.
    concurrency: int = 8
    #: Runs on this machine: no key required, and nothing leaves it.
    local: bool = False
    about: str = ""
    builtin: bool = True

    def can(self, role: str) -> bool:
        return self.can_embed if role == "embed" else self.can_chat

    def default_model(self, role: str) -> Optional[str]:
        return self.embed_model if role == "embed" else self.chat_model


_BUILTIN: tuple[Provider, ...] = (
    Provider("openai", "openai", key_vars=("OPENAI_API_KEY", "OPENAI_API"),
             chat_model="gpt-5.4-mini", embed_model="text-embedding-3-small",
             dimensions=True, about="OpenAI"),
    Provider("anthropic", "anthropic", base_url="https://api.anthropic.com",
             key_vars=("ANTHROPIC_API_KEY",), chat_model="claude-haiku-4-5",
             can_embed=False,
             about="Claude, through the Messages API. No embeddings"),
    Provider("gemini", "chat",
             base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
             key_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
             chat_model="gemini-2.5-flash", embed_model="gemini-embedding-001",
             dimensions=True,
             about="Google Gemini, through its OpenAI-compatible endpoint"),
    Provider("mistral", "chat", base_url="https://api.mistral.ai/v1",
             key_vars=("MISTRAL_API_KEY",), chat_model="mistral-small-latest",
             embed_model="mistral-embed", about="Mistral"),
    Provider("openrouter", "chat", base_url="https://openrouter.ai/api/v1",
             key_vars=("OPENROUTER_API_KEY",), can_embed=False,
             about="OpenRouter: one key, many vendors. Name the model"),
    Provider("groq", "chat", base_url="https://api.groq.com/openai/v1",
             key_vars=("GROQ_API_KEY",), can_embed=False,
             about="Groq. Name the model"),
    Provider("together", "chat", base_url="https://api.together.xyz/v1",
             key_vars=("TOGETHER_API_KEY",),
             about="Together AI. Name the model"),
    Provider("deepseek", "chat", base_url="https://api.deepseek.com/v1",
             key_vars=("DEEPSEEK_API_KEY",), chat_model="deepseek-chat",
             can_embed=False, structured="json_object",
             about="DeepSeek. Text only, so aggregates rather than describe"),
    Provider("xai", "chat", base_url="https://api.x.ai/v1",
             key_vars=("XAI_API_KEY",), can_embed=False,
             about="xAI Grok. Name the model"),
    Provider("voyage", "chat", base_url="https://api.voyageai.com/v1",
             key_vars=("VOYAGE_API_KEY",), can_chat=False,
             embed_model="voyage-3.5", input_type=True,
             about="Voyage AI embeddings"),
    Provider("ollama", "chat", base_url="http://localhost:11434/v1",
             key_vars=("OLLAMA_API_KEY",), chat_model="gemma3:4b",
             embed_model="nomic-embed-text", local=True, concurrency=1,
             about="Ollama on this machine. `ollama pull` the model first"),
    Provider("lmstudio", "chat", base_url="http://localhost:1234/v1",
             key_vars=("LMSTUDIO_API_KEY",), local=True, concurrency=1,
             about="LM Studio's server. Name the model it has loaded"),
    Provider("llamacpp", "chat", base_url="http://localhost:8080/v1",
             key_vars=("LLAMACPP_API_KEY",), local=True, concurrency=1,
             about="llama.cpp `llama-server`. Start it with --embeddings for vectors"),
    Provider("local", "local", can_chat=False,
             embed_model="BAAI/bge-small-en-v1.5", local=True,
             about="a Hugging Face embedding model loaded into this process"),
)

#: Fields `data/providers.json` may set. `name` is the entry's key and
#: `builtin` is not the file's to claim.
_FILE_FIELDS = tuple(f.name for f in fields(Provider)
                     if f.name not in ("name", "builtin"))


# ------------------------------------------------------------------ loading

def _check(name: str, entry: Any, existing: Optional[Provider]) -> list[str]:
    """Why a `providers.json` entry cannot be used, as messages."""
    if not isinstance(entry, dict):
        return ["must be an object of fields"]
    problems = []
    if not NAME.match(name):
        # No `/`: `ollama/gemma3:4b` splits on the first one, so a provider name
        # carrying one could never be addressed.
        problems.append("name must be lowercase letters, digits, `_`, `.` or `-`")
    if name in OFFLINE.values():
        problems.append(f"{name!r} is reserved: it loads no model")
    for key in entry:
        if "key" in key and key != "key_vars":
            problems.append(f"{key!r}: keys never live in this file -- name the "
                            "environment variable in key_vars and set it in .env")
        elif key not in _FILE_FIELDS:
            problems.append(f"unknown field {key!r}; known: {', '.join(_FILE_FIELDS)}")
    protocol = entry.get("protocol", existing.protocol if existing else None)
    if protocol not in PROTOCOLS:
        problems.append(f"protocol must be one of {', '.join(PROTOCOLS)}")
    if existing is None and protocol == "chat" and not entry.get("base_url"):
        problems.append("a new chat provider needs base_url")
    if entry.get("structured", "json_schema") not in STRUCTURED:
        problems.append(f"structured must be one of {', '.join(STRUCTURED)}")
    if entry.get("token_field", "max_tokens") not in ("max_tokens",
                                                     "max_completion_tokens"):
        problems.append("token_field must be max_tokens or max_completion_tokens")
    if "concurrency" in entry and not (type(entry["concurrency"]) is int
                                       and entry["concurrency"] >= 1):
        problems.append("concurrency must be a whole number, 1 or more")
    if "key_vars" in entry and not (isinstance(entry["key_vars"], list)
                                    and all(isinstance(v, str) for v in entry["key_vars"])):
        problems.append("key_vars must be a list of variable names")
    return problems


def load() -> tuple[dict[str, Provider], list[str]]:
    """Every provider, and why any `providers.json` entry was left out.

    A bad entry is dropped and reported rather than raised: the file is
    hand-edited, and one typo taking down `/capabilities` -- and with it every
    form built from it -- would be a large failure for a small mistake.
    """
    found = {p.name: p for p in _BUILTIN}
    problems: list[str] = []
    path = paths.PROVIDERS
    if not path.exists():
        return found, problems
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return found, [f"{path}: {exc}"]
    entries = data.get("providers", {}) if isinstance(data, dict) else {}
    for name, entry in entries.items():
        existing = found.get(name)
        wrong = _check(name, entry, existing)
        if wrong:
            problems += [f"{path}: {name}: {w}" for w in wrong]
            continue
        values = {k: (tuple(v) if k == "key_vars" else v) for k, v in entry.items()}
        found[name] = (replace(existing, **values) if existing
                       else Provider(name=name, builtin=False, **values))
    return found, problems


def providers() -> dict[str, Provider]:
    return load()[0]


def get(name: str) -> Provider:
    found = providers()
    if name not in found:
        raise ProviderError(f"unknown provider {name!r}; known: {', '.join(sorted(found))}")
    return found[name]


def names(role: str) -> list[str]:
    """Every name that can serve a role, providers and offline ones alike."""
    out = [p.name for p in providers().values() if p.can(role)]
    if role in OFFLINE:
        out.append(OFFLINE[role])
    return sorted(out)


# --------------------------------------------------------------- resolving

def split(spec: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """`ollama/gemma3:4b` -> (`ollama`, `gemma3:4b`). A bare name has no model.

    On the FIRST slash, because model ids carry their own:
    `local/BAAI/bge-small-en-v1.5` is the `local` provider and a Hugging Face
    id. A head that is not a provider leaves the spec whole, so the error names
    what was typed rather than half of it.
    """
    if not spec or not spec.strip():
        return None, None
    spec = spec.strip()
    head, slash, rest = spec.partition("/")
    if slash and (head in providers() or head in OFFLINE.values()):
        return head, rest or None
    return spec, None


def choose(role: str, spec: Optional[str] = None) -> tuple[str, Optional[str]]:
    """The provider and model a call should use.

    `spec` wins, then the role's variable, then `FALLBACK` -- each a provider
    or `provider/model`. The model comes back `None` only for an offline name,
    which has none.
    """
    if role not in ROLES:
        raise ProviderError(f"unknown role {role!r}; known: {', '.join(ROLES)}")
    env.load()
    chosen, model = split(spec)
    if chosen is None:
        chosen, model = split(os.environ.get(ENV[role]))
    chosen = chosen or FALLBACK

    if chosen == OFFLINE.get(role):
        return chosen, None
    if chosen in OFFLINE.values():
        raise ProviderError(f"{chosen!r} cannot serve {role}; "
                            f"known: {', '.join(names(role))}")

    provider = get(chosen)
    if not provider.can(role):
        what = "vectors" if role == "embed" else "answers"
        raise ProviderError(f"{chosen} serves no {what}; for {role} use one of "
                            f"{', '.join(names(role))}")
    model = model or provider.default_model(role)
    if not model:
        raise ProviderError(
            f"{chosen} has no default {'embedding' if role == 'embed' else 'chat'} "
            f"model: name one, as `{chosen}/<model>`")
    return chosen, model


def base_url(provider: Provider) -> Optional[str]:
    """`<NAME>_BASE_URL` if set, else the declared one. `OLLAMA_BASE_URL` points
    the built-in at another machine without touching the file."""
    env.load()
    variable = re.sub(r"[^A-Z0-9]", "_", provider.name.upper()) + "_BASE_URL"
    return os.environ.get(variable) or provider.base_url


def api_key(provider: Provider) -> Optional[str]:
    """The key, None where none is needed, or a message naming where to put it."""
    env.load()
    for variable in provider.key_vars:
        if os.environ.get(variable):
            return os.environ[variable]
    if provider.local or not provider.key_vars:
        return None
    raise ProviderUnavailable(
        f"no key for {provider.name}: set {' or '.join(provider.key_vars)} in "
        ".env (it is gitignored) or in the environment")


def problems(role: str, spec: Optional[str] = None) -> list[str]:
    """What stops a role running, found before anything is queued.

    Unknown names and missing keys only. Whether a local server is up, or a
    model id exists, is the provider's to answer -- and asking would make
    validation a network call.
    """
    try:
        chosen, _ = choose(role, spec)
    except ProviderError as exc:
        return [str(exc)]
    if chosen in OFFLINE.values():
        return []
    try:
        api_key(get(chosen))
    except ProviderUnavailable as exc:
        return [f"{role}: {exc}"]
    return []


# -------------------------------------------------------------- publishing

def defaults() -> dict[str, str]:
    """What each role resolves to right now, as `provider/model`."""
    out: dict[str, str] = {}
    for role, field in (("describe", "describer"), ("llm", "llm"),
                        ("embed", "embedder")):
        try:
            chosen, model = choose(role)
            out[field] = f"{chosen}/{model}" if model else chosen
        except ProviderError:
            out[field] = os.environ.get(ENV[role]) or FALLBACK
    return out


def catalog() -> dict[str, Any]:
    """Every provider, whether it can run here, and why not. Never a key."""
    found, wrong = load()
    rows = []
    for provider in found.values():
        try:
            key = api_key(provider)
            ready = True
            why = ("runs on this machine; not checked whether it is up"
                   if provider.local else "key found" if key else "needs no key")
        except ProviderUnavailable:
            ready, why = False, f"set {' or '.join(provider.key_vars)}"
        rows.append({
            "name": provider.name, "protocol": provider.protocol,
            "about": provider.about, "builtin": provider.builtin,
            "local": provider.local, "chat": provider.can_chat,
            "embed": provider.can_embed, "chat_model": provider.chat_model,
            "embed_model": provider.embed_model,
            "base_url": base_url(provider), "key_vars": list(provider.key_vars),
            "structured": provider.structured if provider.protocol == "chat" else None,
            "concurrency": provider.concurrency,
            "configured": ready, "why": why,
        })
    return {"providers": rows, "problems": wrong, "file": str(paths.PROVIDERS),
            "env": dict(ENV)}


__all__ = ["ENV", "FALLBACK", "OFFLINE", "PROTOCOLS", "Provider", "ProviderError",
           "ProviderUnavailable", "ROLES", "api_key", "base_url", "catalog",
           "choose", "defaults", "get", "load", "names", "problems", "providers",
           "split"]
