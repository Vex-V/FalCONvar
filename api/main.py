"""HTTP in front of the pipeline.

Routes, request shapes and status codes. The work is `api/service.py`, which is
the pipeline in terms a request can supply; the pipeline proper is `ver2/`, and
nothing here reaches past `service`.

Three kinds of endpoint, and the difference is how long they take:

  **immediate**   listing videos, reading an artifact, serving a frame, and
                  `/search` -- one embedding call and one query, tens of
                  milliseconds. These answer in the request.
  **queued**      `/videos` (upload and process), `/describe`, `/embed`. Each
                  is minutes of GPU or inference, so they return a job id and
                  the caller polls `/jobs/{id}`.
  **introspective** `/capabilities` reads the registries, so a sampler added
                  to `ver2` appears in the API without anyone editing a list.

Uploads are written to `uploads/` under the id they will be processed as, and
the id is derived from the filename rather than chosen by the client, because
it is also the key every table and every output directory uses.
"""

from __future__ import annotations

import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api import service
from api.jobs import Runner, progress
from ver2 import db, orchestrate

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Read `.env` once, before any request can need a key.

    Every stage function loads it too, because they are callable on their own.
    Doing it here as well means a missing key is a startup concern rather than
    something discovered by a job that has already run for a minute.
    """
    db.load_env()
    yield


app = FastAPI(
    title="FalCONvar",
    version="2.0",
    description="Video and audio in, searchable moments out.",
    lifespan=lifespan,
)
runner = Runner()

#: Anything else in a filename becomes an underscore. The id keys four
#: Postgres tables, an output directory and a Qdrant payload, so it has to
#: survive being a path segment and a URL without quoting.
SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_id(name: str) -> str:
    cleaned = SAFE.sub("_", Path(name).stem).strip("_")
    if not cleaned:
        raise HTTPException(422, "could not derive a video id from that filename")
    return cleaned


# --------------------------------------------------------------- what it can do
@app.get("/capabilities")
def capabilities() -> dict[str, Any]:
    return service.available()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "queued": runner.pending()}


# ------------------------------------------------------------------- ingesting
@app.post("/videos", status_code=202)
async def upload(
    file: UploadFile = File(..., description="the media file"),
    samplers: str = Form("clip", description="comma-separated; `uniform:text` allowed"),
    chunking: str = Form("uniform"),
    chunk_duration: float = Form(20.0),
    every_seconds: float = Form(3.0),
    vocabulary: Optional[str] = Form(None, description="objects: the class list"),
    threshold: Optional[float] = Form(None),
    per_second: float = Form(1.0),
    frame_store: bool = Form(True),
    use_audio: bool = Form(True),
    transcriber: str = Form("whisper"),
    diarizer: str = Form("pyannote"),
    language: Optional[str] = Form(None),
    sinks: str = Form("file"),
    video_id: Optional[str] = Form(None),
) -> dict[str, Any]:
    """Accept a file and queue the run. 202 with a job id, not the result.

    Validation happens here, synchronously, so a bad sampler name is a 422 the
    caller sees immediately rather than a job that fails a minute later.
    """
    vid = safe_id(video_id or file.filename or "video")
    service.UPLOADS.mkdir(parents=True, exist_ok=True)
    target = service.UPLOADS / f"{vid}{Path(file.filename or '').suffix or '.mp4'}"
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    names = [s.strip() for s in samplers.split(",") if s.strip()]
    try:
        built = service.build_samplers(names, {
            "every_seconds": every_seconds, "vocabulary": vocabulary,
            "threshold": threshold})
    except (ValueError, KeyError) as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(422, str(exc)) from None

    options = orchestrate.Options(
        media=target, video_id=vid, chunking=chunking,
        chunk_duration=chunk_duration, samplers=built, per_second=per_second,
        frame_store=frame_store, use_audio=use_audio, transcriber=transcriber,
        diarizer=diarizer, language=language,
        sinks=[s.strip() for s in sinks.split(",") if s.strip()])
    problems = orchestrate.validate(options)
    if problems:
        target.unlink(missing_ok=True)
        raise HTTPException(422, {"problems": problems})

    job = runner.submit("ingest", vid,
                        lambda j: service.ingest(options, on_progress=progress(j)))
    return {"job": job.as_dict(), "video_id": vid,
            "poll": f"/jobs/{job.id}"}


class DescribeRequest(BaseModel):
    video_id: str
    describer: str = "openai"
    model: Optional[str] = None
    sinks: list[str] = Field(default_factory=lambda: ["file"])
    limit: Optional[int] = Field(None, description="stop after this many calls")


@app.post("/describe", status_code=202)
def describe(request: DescribeRequest) -> dict[str, Any]:
    """Queue a describer pass over an already-ingested video."""
    job = runner.submit("describe", request.video_id, lambda j: service.describe(
        request.video_id, request.describer, request.model, request.sinks,
        request.limit))
    return {"job": job.as_dict(), "poll": f"/jobs/{job.id}"}


class EmbedRequest(BaseModel):
    video_id: str
    embedder: Optional[str] = None
    model: Optional[str] = None
    indexes: Optional[list[str]] = None


@app.post("/embed", status_code=202)
def embed(request: EmbedRequest) -> dict[str, Any]:
    """Queue embedding of a video's descriptions and transcript."""
    job = runner.submit("embed", request.video_id, lambda j: service.embed(
        request.video_id, request.embedder, request.model, request.indexes))
    return {"job": job.as_dict(), "poll": f"/jobs/{job.id}"}


# ------------------------------------------------------------------------ jobs
@app.get("/jobs")
def jobs() -> dict[str, Any]:
    return {"queued": runner.pending(),
            "note": "job records live in memory and are lost on restart; "
                    "what they produced is on disk and in Postgres",
            "jobs": [j.as_dict() for j in runner.all()]}


@app.get("/jobs/{job_id}")
def job(job_id: str) -> dict[str, Any]:
    found = runner.get(job_id)
    if found is None:
        raise HTTPException(404, f"no job {job_id}")
    return found.as_dict()


# -------------------------------------------------------------------- browsing
@app.get("/videos")
def list_videos() -> dict[str, Any]:
    return {"videos": service.videos()}


@app.get("/videos/{video_id}/{name}")
def artifact(video_id: str, name: str) -> Any:
    if name not in ("manifest", "timeline", "descriptions", "transcript"):
        raise HTTPException(404, "expected manifest, timeline, descriptions "
                                 "or transcript")
    try:
        return service.artifact(video_id, name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from None


@app.get("/videos/{video_id}/frames/{index}")
def frame(video_id: str, index: int) -> FileResponse:
    """The evidence for a moment: one JPEG, as ingest wrote it."""
    try:
        return FileResponse(service.frame_path(video_id, index),
                            media_type="image/jpeg")
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from None


# ----------------------------------------------------------------- the RAG half
class SearchRequest(BaseModel):
    query: str
    video_id: Optional[str] = Field(None, description="restrict to one video")
    sampler: Optional[str] = Field(
        None, description="one question only: yolo, text, transcript, overview. "
                          "Note this gives up cross-sampler agreement")
    moments: int = 5
    limit: int = Field(20, description="descriptions ranked before folding")
    embedder: Optional[str] = None
    model: Optional[str] = None
    indexes: Optional[list[str]] = None


@app.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    """Answer in the request: one embedding call and one query.

    Each moment carries `frame_indexes`, which `/videos/{id}/frames/{index}`
    serves -- so a client can show the evidence without a second round trip to
    work out what to ask for.
    """
    try:
        found = service.search(
            request.query, request.video_id, request.sampler, request.moments,
            request.limit, request.embedder, request.model, request.indexes)
    except Exception as exc:                            # noqa: BLE001
        raise HTTPException(502, f"search failed: {exc}") from None
    return {"query": request.query, "moments": found,
            "frames": f"/videos/{{video_id}}/frames/{{index}}"}


# ------------------------------------------------------------------- the site
# Served by the same app as the API, at /app rather than /, because `GET
# /videos` is an API route and a site mounted at the root would shadow it.
# Same origin either way, so the browser client needs no CORS and no base URL.
WEB = Path(__file__).resolve().parents[1] / "web"
if WEB.is_dir():
    app.mount("/app", StaticFiles(directory=WEB, html=True), name="web")

    @app.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        return RedirectResponse("/app/")
