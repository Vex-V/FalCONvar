"""The model that actually looks at the frames.

Separate from `describers/` because that package is the registry and the
protocol -- the shape every describer has -- while this is one concrete
integration, with a network client, an API key and a set of prompts. Keeping
them apart means a second provider is a sibling of this folder rather than a
change to the interface, and it keeps the `openai` import behind a lazy
registry entry so a stub run never pays for it.

That laziness has to reach this file too. `prompts.py` is a leaf that imports
nothing and holds the question vocabulary, so anything choosing a question
wants to read it -- but importing a submodule runs its package's `__init__`,
and re-exporting the client here made `ver2.video.describe.vlm.prompts` pull in
`openai`. A module-level `__getattr__` keeps the names available on the package
while resolving them only when one is actually asked for.
"""

from typing import Any

__all__ = ["DEFAULT_MODEL", "DescriberUnavailable", "OpenAIDescriber"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import openai_client
        return getattr(openai_client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
