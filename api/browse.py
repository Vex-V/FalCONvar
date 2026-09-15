"""Reading back what a run wrote, one query at a time.

`/videos/{id}/artifacts/{name}` hands over a whole document, which is the right
shape for a download and the wrong one for a question. This is the other half:
the rows, filtered, ordered and paged, so a client asks "which chunks did
`clip` pick frames in" rather than fetching a manifest and scanning it.

**Read key, not write key.** Every query here goes through
`db.client(write=False)`. Nothing in this module writes, so nothing in it needs
a key that could -- and RLS then answers under the same grants a dashboard
would, which is the thing worth verifying rather than bypassing.

**Columns are probed, never restated.** One `limit 1` per table per process
tells us the deployed shape; a hard-coded list would be a second copy of
`install.sql` to keep in step with, and that file is re-run against live
databases precisely because they drift. What *is* stated here is which columns
are too wide to send by default -- a judgement about size, not a claim about
the schema, so it cannot go stale in the way a column list can.

**A page is bounded and says how big the whole result is.** `count` comes back
with every query, because "20 rows" and "20 of 4,812 rows" are different
answers and a client showing the first as the second is lying about coverage.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from falconvar.shared.storage import db

router = APIRouter(prefix="/db", tags=["database"])

#: table -> one line on what it holds. The sections of `db/supabase/install.sql`
#: in the order a run fills them, so a listing reads the way the pipeline does.
TABLES: dict[str, str] = {
    "videos": "the file: its two streams and their addressing",
    "timelines": "THE GRID: the policy that chose it, and its fingerprint",
    "chunks": "every chunk's span. Everything below joins on (video_id, chunk_id)",
    "cuts": "boundary evidence, and the score series it was thresholded from",
    "transcripts": "the raw transcript: words, segments and turns, before any grid",
    "transcript_chunks": "what was said, cut to the grid",
    "manifests": "one row per ingest: the config, and the grid it ran against",
    "chunk_samplers": "one row per (chunk, sampler RUN), with the frames it kept",
    "descriptions": "one model answer per (chunk, sampler:question)",
    "embeddings": "the text that went into the index, and the vector it became",
    "aggregates": "video-level answers, one row per (aggregator, input)",
    "entities": "who is who across chunks: one row per linked entity, with its account",
    "entity_mentions": "every observation a link profile read, the entity it joined, and any doubt",
    "video_embeddings": "one vector per video, from its summary aggregate",
    "prompts": "what each question said, at the version a run asked it under",
    "aggregate_definitions": "what each aggregate prompt or link profile said, at the version used",
}

#: Columns not worth sending unless they are asked for. A 1536-wide vector and
#: a tsvector are payload no reader looks at, and `transcripts` carries every
#: word timestamp in the file -- megabytes, to answer a question about a header.
HEAVY: dict[str, tuple[str, ...]] = {
    "embeddings": ("embedding", "fts"),
    "transcripts": ("words", "segments", "turns"),
    "cuts": ("scores",),
}

#: How each table is ordered when the caller does not say. Chunk order wherever
#: there is one, because a grid read out of order is a list rather than a
#: timeline.
ORDER: dict[str, tuple[str, ...]] = {
    "chunks": ("video_id", "chunk_id"),
    "transcript_chunks": ("video_id", "chunk_id"),
    "chunk_samplers": ("video_id", "chunk_id", "sampler_id"),
    "descriptions": ("video_id", "chunk_id", "sampler_id"),
    "embeddings": ("video_id", "chunk_id", "sampler_id"),
    "aggregates": ("video_id", "aggregate_id"),
    "entities": ("video_id", "aggregate_id", "entity_id"),
    "entity_mentions": ("video_id", "aggregate_id", "chunk_id", "mention_key"),
    # Not keyed by a video at all: name, then newest version first.
    "prompts": ("name",),
    "aggregate_definitions": ("name",),
}

#: The PostgREST operators a client may name. An allowlist rather than a
#: pass-through: an unknown operator should be a 422 naming the ones that exist
#: rather than an AttributeError from inside the driver.
OPS = ("eq", "neq", "gt", "gte", "lt", "lte", "like", "ilike", "is",
       "in", "cs", "cd")

_columns: dict[str, list[str]] = {}
_lock = threading.Lock()


def columns_of(table: str) -> list[str]:
    """The deployed columns, read off one row. Cached for the process.

    Empty when the table is empty, which is the honest answer: a filter built
    from this offers what is really there, and a table with no rows has nothing
    to filter on.
    """
    with _lock:
        if table in _columns:
            return _columns[table]
    try:
        found = (db.client(write=False).table(table)
                 .select("*").limit(1).execute())
        names = sorted(found.data[0]) if found.data else []
    except db.DatabaseUnavailable:
        raise
    except Exception:                                     # noqa: BLE001
        names = []
    with _lock:
        _columns[table] = names
    return names


def _select_for(table: str, include_heavy: bool) -> str:
    """`*` minus whatever is too wide to send, as PostgREST wants it."""
    heavy = HEAVY.get(table, ())
    if include_heavy or not heavy:
        return "*"
    known = columns_of(table)
    if not known:
        return "*"
    return ",".join(c for c in known if c not in heavy)


class Filter(BaseModel):
    column: str
    op: str = Field("eq", description="one of /db/tables.ops")
    value: Any = None


class Query(BaseModel):
    """One page of one table."""

    table: str
    filters: list[Filter] = Field(default_factory=list)
    order: Optional[str] = None
    desc: bool = False
    limit: int = Field(50, ge=1, le=1000)
    offset: int = Field(0, ge=0)
    include_heavy: bool = Field(
        False, description="send the vector and the tsvector too")


@router.get("/status")
def status() -> dict[str, Any]:
    """Whether this deployment can read the database, and what is in it.

    Answered rather than left to be discovered: RLS with no policy denies reads
    *silently* -- zero rows, no error -- so a client that could not tell "not
    configured" from "nothing ingested" would show an empty table for both.
    """
    if not db.configured(write=False):
        return {"configured": False, "reachable": False, "schema": db.SCHEMA,
                "error": "set SUPABASE_URL and "
                         + " or ".join(db.PUBLISHABLE_VARS) + " in .env"}
    counts: dict[str, Any] = {}
    try:
        api = db.client(write=False)
        for table in TABLES:
            # `*` with `head`, not a named column. Counting by `video_id`
            # assumed every table is keyed by a video, and `prompts` is not --
            # it is keyed by (name, version), so naming that column returned
            # `42703 column prompts.video_id does not exist` and took the whole
            # endpoint down with it. `head` asks for the count and no rows, so
            # `*` costs nothing even on a table carrying a 1536-wide vector.
            found = (api.table(table).select("*", count="exact", head=True)
                     .execute())
            counts[table] = found.count
    except db.DatabaseUnavailable as exc:
        return {"configured": False, "reachable": False, "schema": db.SCHEMA,
                "error": str(exc)}
    except Exception as exc:                              # noqa: BLE001
        return {"configured": True, "reachable": False, "schema": db.SCHEMA,
                "error": f"{type(exc).__name__}: {exc}"}
    return {"configured": True, "reachable": True, "schema": db.SCHEMA,
            "counts": counts}


@router.get("/tables")
def tables() -> dict[str, Any]:
    """Every table, what it holds, and the columns actually deployed."""
    out = []
    for name, about in TABLES.items():
        out.append({"name": name, "about": about,
                    "columns": columns_of(name),
                    "heavy": list(HEAVY.get(name, ())),
                    "order": list(ORDER.get(name, ()))})
    return {"schema": db.SCHEMA, "tables": out, "ops": list(OPS)}


@router.post("/query")
def query(request: Query) -> dict[str, Any]:
    """One page of rows, with the size of the whole result beside it."""
    if request.table not in TABLES:
        raise HTTPException(404, {"error": f"unknown table {request.table!r}",
                                  "known": list(TABLES)})
    try:
        api = db.client(write=False)
    except db.DatabaseUnavailable as exc:
        raise HTTPException(503, {"error": str(exc)}) from None

    select = _select_for(request.table, request.include_heavy)
    builder = api.table(request.table).select(select, count="exact")

    for one in request.filters:
        if one.op not in OPS:
            raise HTTPException(422, {"error": f"unknown operator {one.op!r}",
                                      "known": list(OPS)})
        if one.op != "is" and (one.value is None or one.value == ""):
            continue
        if one.op == "in":
            values = (one.value if isinstance(one.value, list)
                      else [v.strip() for v in str(one.value).split(",")])
            builder = builder.in_(one.column, values)
        elif one.op == "is":
            builder = builder.is_(one.column, one.value)
        else:
            builder = getattr(builder, one.op)(one.column, one.value)

    order = ([request.order] if request.order
             else list(ORDER.get(request.table, ())))
    for column in order:
        builder = builder.order(column, desc=request.desc)

    try:
        found = builder.range(request.offset,
                              request.offset + request.limit - 1).execute()
    except Exception as exc:                              # noqa: BLE001
        # The message as PostgREST wrote it: "column does not exist" and
        # "operator does not apply" are both correctable by the caller, and
        # neither is diagnosable from a 500 with no text in it.
        raise HTTPException(422, {"error": f"{type(exc).__name__}: {exc}"}) from None

    return {"table": request.table, "rows": found.data or [],
            "count": found.count, "offset": request.offset,
            "limit": request.limit, "select": select}


__all__ = ["HEAVY", "OPS", "ORDER", "TABLES", "columns_of", "query", "router",
           "status", "tables"]
