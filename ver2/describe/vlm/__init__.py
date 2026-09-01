"""The model that actually looks at the frames.

Separate from `describers/` because that package is the registry and the
protocol -- the shape every describer has -- while this is one concrete
integration, with a network client, an API key and a set of prompts. Keeping
them apart means a second provider is a sibling of this folder rather than a
change to the interface, and it keeps the `openai` import behind a lazy
registry entry so a stub run never pays for it.
"""

from .openai_client import DEFAULT_MODEL, DescriberUnavailable, OpenAIDescriber

__all__ = ["DEFAULT_MODEL", "DescriberUnavailable", "OpenAIDescriber"]
