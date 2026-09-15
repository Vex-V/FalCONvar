"""What more than one component needs.

    paths.py        where things live; the only module that knows a filename
    env.py          reading `.env` at the top of an entry point

    contracts/      what components hand each other
      documents.py  what every artifact is
      schemas.py    JSON Schema generated from the dataclasses
    storage/        where a document goes
      sinks.py      writing a document to file and/or Postgres
      db.py         the Supabase client, and the one list of its key names
      rows.py       documents -> rows; the only module that knows table names
    models/         who answers a model call
      providers.py  which provider serves a role, and how to reach it
      llm.py        asking a model for text or a JSON shape

`paths` and `env` sit at the top because everything below imports them.
`paths` and `documents` import nothing. A module only one component needs does
not belong here.
"""
