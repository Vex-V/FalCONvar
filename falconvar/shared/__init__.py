"""What more than one component needs.

    paths.py      where things live; the only module that knows a filename
    documents.py  what every artifact is
    sinks.py      writing a document to file and/or Postgres
    env.py        reading `.env` at the top of an entry point
    llm.py        the text model call, and the one list of key names
    db.py         the Supabase client, and the one list of its key names
    rows.py       documents -> rows; the only module that knows table names
    schemas.py    JSON Schema generated from the dataclasses

`paths` and `documents` import nothing. A module only one component needs does
not belong here.
"""

from __future__ import annotations

from . import documents, env, paths, sinks

__all__ = ["documents", "env", "paths", "sinks"]
