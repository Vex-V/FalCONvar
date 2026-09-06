"""Audio: the stream that cannot be chunked before it is understood.

Every other stage here works a window at a time. Ingest decides which frames
matter chunk by chunk; describe asks one question per (chunk, sampler). Audio
cannot be built that way, and the reason is not convenience.

**Transcription and diarization are whole-file operations.** Whisper carries
context across an utterance -- hand it a 20-second window and it loses the
sentence that straddles the boundary, and its language detection, which reads
the opening seconds, gets a fragment instead of a voice. Diarization is worse:
speaker labels come from clustering embeddings over the *whole* recording, so
`SPEAKER_00` in one window has no relation to `SPEAKER_00` in the next. Chunk
first and the speakers are not merely misaligned, they are unnameable.

So this module scans the file once, end to end, and produces a transcript with
word-level timestamps and speaker turns over the whole timeline. Only then are
boundaries applied -- and because Whisper returns a timestamp per *word*,
re-segmenting a finished transcript to any grid is free and loses nothing.
That asymmetry is what lets the video side own the chunk boundaries when the
run asks it to, without the audio side paying for it.

Nothing in here is cheap to redo and nothing in here is slow: measured on an
RTX 4060, decode runs at ~700x realtime, Whisper `small` at ~26x, pyannote at
~31x, so a one-hour recording is about four minutes of work -- trivial beside
one describer call per chunk.
"""
