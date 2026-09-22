"""`ner` -- named entities, and which chunks each appears in.

Imported only when asked for by name: a `--tier free` run must not pull in
torch. The registry holds `\"ner:NERAggregator\"` as a string for that reason.
"""

from .driver import NERAggregator

__all__ = ["NERAggregator"]
