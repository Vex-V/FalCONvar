"""What more than one component needs.

Three modules, and two of them import nothing at all:

    paths.py      where things live. The only module that knows an artifact's
                  filename, so no caller ever concatenates one.
    documents.py  what every artifact IS. A leaf on purpose -- if a document's
                  dataclass lived in the component that produces it, `cut`
                  would import `boundaries` to read a timeline, growing exactly
                  the edges the file-handoff design exists to remove.
    sinks.py      writing a document where it was asked to go. One fan-out,
                  where `falconvar` grew four near-identical `output/` packages.
    env.py        reading `.env` at the top of an entry point. Keys only --
                  paths resolve earlier than this can run.
    llm.py        the text model call, and the one list of key names.
    db.py         the Supabase client, and the one list of ITS key names.
    rows.py       documents -> rows. The only module that knows table names.
    schemas.py    JSON Schema generated from the dataclasses above.

Nothing here knows what a component does. A module that only one component
needs does not belong here -- it belongs in that component.
"""

from __future__ import annotations

from . import documents, env, paths, sinks

__all__ = ["documents", "env", "paths", "sinks"]
