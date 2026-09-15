"""The Supabase client, and the one place that knows the key names.

Imports nothing from the components. Every stage that writes rows gets its
client here, so "which env var holds the key" is answered once -- the mistake
the embedder made by inventing its own list, where `OPENAI_API` worked for the
describer and not for it.

**Two keys, and which one you get depends on what you are doing.** The secret
key writes; the publishable one reads. A reader handed a writing key is a
larger blast radius than the job needs.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .. import env

#: falconvar's tables live in their own Postgres schema, not `public`. falconvar's
#: are still deployed alongside and `video_embeddings` collides outright, so a
#: shared schema would mean `create table if not exists` silently doing nothing
#: and every write landing in the wrong shape.
SCHEMA = "falconvar"

URL_VARS = ("SUPABASE_URL",)
SECRET_VARS = ("SUPABASE_SECRET_KEY", "SUPABASE_SERVICE_ROLE_KEY",
               "SUPABASE_SERVICE_KEY")
PUBLISHABLE_VARS = ("SUPABASE_PUBLISHABLE_KEY", "SUPABASE_ANON_KEY")


class DatabaseUnavailable(RuntimeError):
    """No URL, no key, no client library, or a server that will not answer."""


def _first(names: tuple[str, ...]) -> Optional[str]:
    env.load()
    return next((os.environ[n] for n in names if os.environ.get(n)), None)


def configured(write: bool = True) -> bool:
    """Whether a client could be built. Asked before offering the backend."""
    names = SECRET_VARS if write else PUBLISHABLE_VARS
    return bool(_first(URL_VARS) and _first(names))


def client(url: Optional[str] = None, key: Optional[str] = None,
           write: bool = True) -> Any:
    """A Supabase client. ``write=False`` uses the read-only key."""
    url = url or _first(URL_VARS)
    names = SECRET_VARS if write else PUBLISHABLE_VARS
    key = key or _first(names)
    if not url or not key:
        raise DatabaseUnavailable(
            "set SUPABASE_URL and " + " or ".join(names) +
            " in .env (it is gitignored) or in the environment")
    from supabase import ClientOptions, create_client
    try:
        return create_client(url, key,
                             options=ClientOptions(schema=SCHEMA))
    except Exception as exc:                             # noqa: BLE001
        raise DatabaseUnavailable(f"could not connect: {exc}") from None


def upsert(table: str, rows: list[dict[str, Any]], api: Any = None,
           chunk: int = 200) -> int:
    """Upsert rows in batches. Returns how many were sent.

    Batched because a three-hour video's `chunk_samplers` is thousands of rows
    and a single request would be refused on size rather than on content.
    """
    if not rows:
        return 0
    api = api or client()
    for start in range(0, len(rows), chunk):
        api.table(table).upsert(rows[start:start + chunk]).execute()
    return len(rows)


def delete_where(table: str, match: dict[str, Any], api: Any = None) -> None:
    api = api or client()
    query = api.table(table).delete()
    for column, value in match.items():
        query = query.eq(column, value)
    query.execute()


def delete_stale_chunks(table: str, video_id: str, keep_below: int,
                        api: Any = None) -> None:
    """Remove rows naming a chunk that no longer exists.

    A grid that shrank leaves rows for chunks nobody can play. The file writers
    rewrite their whole document and never have this problem; an upserting
    table keeps whatever it was not told to remove. Deleted *after* the
    upserts, so a failure leaves the previous copy whole rather than a hole.
    """
    api = api or client()
    (api.table(table).delete()
        .eq("video_id", video_id).gte("chunk_id", keep_below).execute())


def delete_except(table: str, match: dict[str, Any], column: str,
                  keep: list[Any], api: Any = None) -> None:
    """Remove the rows matching `match` whose `column` is not in `keep`.

    After the upserts, for the reason `delete_stale_chunks` is: a document that
    shrank leaves rows it no longer holds, and deleting first would leave a
    hole if the upsert failed.
    """
    api = api or client()
    query = api.table(table).delete()
    for name, value in match.items():
        query = query.eq(name, value)
    if keep:
        query = query.not_.in_(column, list(keep))
    query.execute()


__all__ = ["DatabaseUnavailable", "PUBLISHABLE_VARS", "SECRET_VARS",
           "URL_VARS", "client", "configured", "delete_except",
           "delete_stale_chunks", "delete_where", "upsert"]
