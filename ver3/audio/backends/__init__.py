"""The model implementations the registries in `models.py` resolve to.

Kept apart from the component's own files so the listing says which is which:
`source.py` and `reader.py` are what `audio` does, and these are one swappable
way of doing part of it. Nothing here is imported until a registry is asked for
it by name -- between them they pull in torch, and a `--transcriber stub` run
should pay for neither.

    whisper.py    a Transcriber (faster-whisper)
    pyannote.py   a Diarizer
    cuda.py       preloads the vendored CUDA DLLs before CTranslate2 asks for
                  them by name. Only whisper needs it, so it lives here.
"""
