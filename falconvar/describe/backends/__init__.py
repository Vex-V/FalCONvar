"""The describers the registry in `base.py` resolves to.

`stub.py` loads nothing and fills the same keys a real answer would, so a stub
run exercises the shape the document has to hold. `model.py` is every real
one -- OpenAI, Anthropic, Gemini, Ollama and the rest, through `shared.llm` --
and is imported only when asked for.
"""
