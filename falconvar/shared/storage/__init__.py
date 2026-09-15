"""Where a document goes: a file, Postgres, or both.

`sinks` is the fan-out every component writes through, `db` the Supabase
client, and `rows` the only module that knows table names.
"""
