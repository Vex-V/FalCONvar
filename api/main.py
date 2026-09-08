"""The HTTP surface: 13 routes over the pipeline.

Three shapes of route:

    immediate  reading what exists, and searching
    queued     anything that decodes, transcribes or pays a model: a 202 with
               a job id, and the caller polls
    uniform    `POST /videos/{id}/run/{component}` runs any component, because
               every one of them is `run(video_id, ...) -> Produced`

`workflow.validate` answers synchronously, so a contradictory request is a 422
rather than a job that fails a minute later. `docs/ROUTES.md` records the rest
of the reasoning; `/docs` is the authority on shapes.
"""

from __future__ import annotations

import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from api import service
from api.jobs import Runner, progress
from falconvar import workflow
from falconvar.shared import env, paths

runner = Runner()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Read `.env` once, before any request can need a key.

    Every component loads it too, because each is callable on its own. Doing it
    here as well makes a missing key a startup concern rather than something
    discovered by a job that has already run for a minute.
    """
    env.load()
    yield


app = FastAPI(
    title="FalCONvar",
    version="3.0",
    description="A video goes in; a searchable index of moments comes out.",
    lifespan=lifespan,
)


def safe_id(name: str) -> str:
    stem = Path(name).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return cleaned or "video"


# ------------------------------------------------------------- introspection

@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    return {"ok": True, "queued": runner.pending()}


@app.get("/capabilities", tags=["meta"])
def capabilities() -> dict[str, Any]:
    """Everything this deployment can be asked for, read from the registries."""
    return service.available()


# ------------------------------------------------------------------ running

@app.post("/videos", status_code=202, tags=["run"])
async def upload(
    file: UploadFile = File(..., description="the media file"),
    policy: str = Form(workflow.Options.policy,
                       description="uniform | scene | vad | speaker"),
    sampler: str = Form(workflow.Options.sampler,
                        description="comma-separated; `yolo:overview` allowed"),
    use_video: bool = Form(True),
    use_audio: bool = Form(True),
    describer: str = Form(workflow.Options.describer),
    embedder: str = Form(workflow.Options.embedder),
    tier: str = Form(workflow.Options.tier, description="free | local | llm"),
    sink: str = Form(workflow.Options.sink, description="where documents go"),
    index: str = Form(workflow.Options.index, description="where vectors go"),
    video_id: Optional[str] = Form(None),
) -> dict[str, Any]:
    """Accept a file and queue the whole pipeline. 202 with a job id."""
    vid = safe_id(video_id or file.filename or "video")
    service.UPLOADS.mkdir(parents=True, exist_ok=True)
    target = service.UPLOADS / f"{vid}{Path(file.filename or '').suffix or '.mp4'}"
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    options = workflow.Options(
        source=target, video_id=vid, policy=policy, sampler=sampler,
        use_video=use_video, use_audio=use_audio, describer=describer,
        embedder=embedder, tier=tier, sink=sink, index=index)

    problems = workflow.validate(options)
    if problems:
        target.unlink(missing_ok=True)
        raise HTTPException(422, {"problems": problems})

    job = runner.submit("workflow", vid,
                        lambda j: service.run_workflow(options, progress(j)))
    return {"job": job.as_dict(), "video_id": vid}


class ComponentRequest(BaseModel):
    """Whatever the component takes. Passed through untouched.

    Deliberately open: the components own their own parameters and validate
    them, so restating each one here would be a second copy of every default
    to keep in step.
    """

    params: dict[str, Any] = Field(default_factory=dict)


@app.post("/videos/{video_id}/run/{component}", status_code=202, tags=["run"])
def run_component(video_id: str, component: str,
                  request: ComponentRequest = ComponentRequest()
                  ) -> dict[str, Any]:
    """Run ONE component against a video that already exists.

    The uniform route. `component` is any name in `/capabilities.components`
    except `media`, which is what an upload does.
    """
    if component not in service.COMPONENTS:
        raise HTTPException(404, {"error": f"unknown component {component!r}",
                                  "known": list(service.COMPONENTS)})
    if not paths.exists(video_id, "media"):
        raise HTTPException(404, {"error": f"{video_id} has not been uploaded"})

    job = runner.submit(component, video_id,
                        lambda j: service.run_component(component, video_id,
                                                        **request.params))
    return {"job": job.as_dict()}


# --------------------------------------------------------------------- jobs

@app.get("/jobs", tags=["jobs"])
def jobs() -> dict[str, Any]:
    return {
        "jobs": [j.as_dict() for j in runner.all()],
        "queued": runner.pending(),
        # Said plainly rather than discovered: the record of a run is in
        # memory, the thing it produced is on disk and in Postgres.
        "note": "job records die with the process; artifacts do not",
    }


@app.get("/jobs/{job_id}", tags=["jobs"])
def job(job_id: str) -> dict[str, Any]:
    found = runner.get(job_id)
    if found is None:
        raise HTTPException(404, {"error": "no such job",
                                  "note": "records die with the process"})
    return found.as_dict()


# ------------------------------------------------------------------ reading

@app.get("/videos", tags=["read"])
def list_videos() -> dict[str, Any]:
    """Every video with an output directory, read from disk."""
    return {"videos": service.videos()}


@app.get("/videos/{video_id}", tags=["read"])
def video_detail(video_id: str) -> dict[str, Any]:
    if not paths.home(video_id).exists():
        raise HTTPException(404, {"error": f"no video {video_id!r}"})
    return service.exports(video_id)


@app.get("/videos/{video_id}/artifacts/{name}", tags=["read"])
def artifact(video_id: str, name: str, download: bool = False) -> Any:
    """One document. `?download=1` only adds a Content-Disposition.

    Content negotiation would be tidier, but a browser cannot set an Accept
    header on a plain link.
    """
    try:
        document = service.artifact(video_id, name)
    except paths.UnknownArtifact:
        raise HTTPException(404, {"error": f"unknown artifact {name!r}",
                                  "known": list(service.ARTIFACTS)}) from None
    except FileNotFoundError as exc:
        raise HTTPException(404, {"error": str(exc)}) from None
    if not download:
        return document
    return JSONResponse(document, headers={
        "Content-Disposition": f'attachment; filename="{video_id}-{name}.json"'})


@app.get("/videos/{video_id}/aggregates", tags=["read"])
def aggregates(video_id: str) -> dict[str, Any]:
    return {"video_id": video_id,
            "aggregates": service.exports(video_id)["aggregates"]}


@app.get("/videos/{video_id}/aggregates/{name}", tags=["read"])
def one_aggregate(video_id: str, name: str) -> dict[str, Any]:
    path = paths.artifact(video_id, "aggregates") / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, {"error": f"{video_id} has no {name} aggregate"})
    from falconvar.shared import sinks
    return sinks.read_json(path)


@app.get("/videos/{video_id}/frames/{index}", tags=["read"])
def frame(video_id: str, index: int) -> FileResponse:
    """One stored frame, by the read index the manifest names it with."""
    try:
        return FileResponse(service.frame_path(video_id, index),
                            media_type="image/jpeg")
    except FileNotFoundError as exc:
        raise HTTPException(404, {"error": str(exc)}) from None


# ----------------------------------------------------------------- searching

class SearchRequest(BaseModel):
    query: str
    video_id: str
    moments: int = 5
    sampler: Optional[str] = Field(
        None, description="narrow to one question's answers. Gives up the "
                          "agreement signal: a chunk can then contribute at "
                          "most one term, so scores roughly halve")
    embedder: str = workflow.Options.embedder
    index: str = workflow.Options.index
    model: Optional[str] = None


@app.post("/search", tags=["search"])
def search(request: SearchRequest) -> dict[str, Any]:
    """Ranked moments. Immediate -- one embedding call and one query.

    The query is embedded with the embedder named here, which must be the one
    that built the index: a mismatch across widths fails loudly, but two models
    of the same width return a well-formed ranking that means nothing.
    """
    try:
        found = service.search(request.query, request.video_id,
                               embedder=request.embedder, model=request.model,
                               moments=request.moments, sampler=request.sampler,
                               index_name=request.index)
    except FileNotFoundError as exc:
        raise HTTPException(404, {"error": str(exc)}) from None
    except (KeyError, ValueError) as exc:
        raise HTTPException(422, {"error": str(exc)}) from None
    return {"query": request.query, "video_id": request.video_id,
            "moments": found}
