"""Everything that turns finished documents into search results.

    embed/      documents -> vectors, keyed by a hash of the text
    retrieve/   a query -> ranked moments

Both modalities land here through the same door: a description and a transcript
chunk are both text with a span and some bound structure, so audio needed no
index of its own and no code past `units.from_transcript`.

`aggregate/` sits outside this on purpose. It answers *which video* rather than
*which twenty seconds*, and only its summary is embedded -- everything else it
produces is statistical, and a vector of a count answers nothing.
"""
