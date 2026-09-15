"""The HTTP surface over the pipeline.

Three shapes of route:

    immediate  reading what exists, and searching
    queued     anything that decodes, transcribes or pays a model: a 202 with
               a job id, and the caller polls
    uniform    `POST /videos/{id}/run/{component}` runs any component, because
               every one of them is `run(video_id, ...) -> Produced`

`browse.router` adds a fourth thing to read: the rows themselves, filtered and
paged, which is the question shape a whole-document download cannot answer.
`web/` is mounted at `/app` when it exists -- static files, no build step,
generating every form from `/capabilities` rather than restating it.

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

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api import browse, service
from api.jobs import Runner, progress
from falconvar import workflow
from falconvar.video_rag.describe import library
from falconvar.video_rag.embed import EmbedderUnavailable
from falconvar.shared import env, paths
from falconvar.shared.storage import db

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
                        description="comma-separated; a sampler may carry one "
                                    "question (`yolo:overview`) or several "
                                    "(`clip:[text,scene]`, or `clip:text+scene`). "
                                    "Specs naming the same sampler are merged "
                                    "into one pass over the frames."),
    use_video: bool = Form(True),
    use_audio: bool = Form(True),
    describer: Optional[str] = Form(
        None, description="a provider, or provider/model (`ollama/gemma3:4b`). "
                          "Blank: FALCONVAR_DESCRIBER, then openai"),
    embedder: Optional[str] = Form(
        None, description="a provider, or provider/model (`local`). "
                          "Blank: FALCONVAR_EMBEDDER, then openai"),
    llm: Optional[str] = Form(
        None, description="who answers `tier=llm`: a provider, or provider/model. "
                          "Blank: FALCONVAR_LLM, then openai"),
    tier: str = Form(workflow.Options.tier, description="free | local | llm"),
    sink: str = Form(workflow.Options.sink, description="where documents go"),
    index: str = Form(workflow.Options.index, description="where vectors go"),
    video_id: Optional[str] = Form(None),
    run: bool = Form(True, description="false: register the file and stop, so "
                                       "the caller can drive the components "
                                       "itself with per-stage settings"),
    response: Response = None,          # noqa: B008 -- set the code per branch
) -> dict[str, Any]:
    """Accept a file. Queue the whole pipeline (202), or just register it (201).

    `run=false` exists because the workflow deliberately carries no per-stage
    tuning -- a scene grid with a 30-second floor, or a uniform stride of 5,
    is set on the component that owns it. Without this a caller wanting those
    had to run the pipeline once on defaults first, paying for a describe it
    was about to redo. It runs `media` and nothing else, because that is what
    makes the video addressable by `POST /videos/{id}/run/{component}` -- and
    it is a container probe, so it is answered rather than queued.
    """
    vid = safe_id(video_id or file.filename or "video")
    service.UPLOADS.mkdir(parents=True, exist_ok=True)
    target = service.UPLOADS / f"{vid}{Path(file.filename or '').suffix or '.mp4'}"
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    if not run:
        try:
            produced = service.register(target, vid, sink)
        except Exception as exc:                          # noqa: BLE001
            target.unlink(missing_ok=True)
            raise HTTPException(422, {"error": str(exc)}) from None
        if response is not None:
            response.status_code = 201
        return {"video_id": produced.video_id, "media": produced.as_dict(),
                "next": f"/videos/{produced.video_id}/run/{{component}}",
                "components": [c for c in service.COMPONENTS]}

    options = workflow.Options(
        source=target, video_id=vid, policy=policy, sampler=sampler,
        use_video=use_video, use_audio=use_audio, describer=describer or None,
        embedder=embedder or None, llm=llm or None,
        tier=tier, sink=sink, index=index)

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


# ------------------------------------------------------------------ prompts

class PromptRequest(BaseModel):
    """A custom question: what to ask, and what shape to answer in.

    Two ways to answer the second half. Name a shipped shape with `shape`, or
    bring your own with `fields` -- which wins when both are given.

    `fields` is a builder, not a JSON Schema. The schema reaches the API with
    `strict: true`, whose subset is narrow, so it is *generated* from the
    builder: a raw schema over HTTP could express something the model API
    refuses, and that failure would land after the frames are read with the
    call about to be paid for.

    Shipped shapes stay shipped: a custom shape cannot take a built-in shape's
    name, and none of them can claim `fallback`.
    """

    name: str = Field(..., description="lowercase, no colon: `sampler:question` splits on one")
    instruction: str = Field(..., description="may use {n}, {span}, {vocabulary}")
    shape: str = Field("scene", description="a name from /prompts.shapes. "
                                            "Ignored when `fields` is given")
    about: str = Field("", description="a note for whoever reads the list later")
    fields: Optional[dict[str, Any]] = Field(
        None, description="bring your own shape: {name: {type, about}} where "
                          "type is 'text' or 'list'. Add `of` "
                          "({key: description}) to make each list entry an "
                          "object, or `one_of` ([...]) to fix the vocabulary")
    summary: str = Field("standard", description="with `fields`: 'standard' "
                                                 "(>=150 words) or 'brief' "
                                                 "(4-5 sentences)")


@app.get("/prompts", tags=["prompts"])
def list_prompts() -> dict[str, Any]:
    """Every question a sampler may be paired with, and the shapes available."""
    return service.prompt_list()


@app.get("/prompts/{name}", tags=["prompts"])
def get_prompt(name: str) -> dict[str, Any]:
    """One question, with the response schema a call would actually be given."""
    try:
        return service.prompt_get(name)
    except library.PromptError as exc:
        raise HTTPException(404, {"error": str(exc)}) from None


@app.post("/prompts", status_code=201, tags=["prompts"])
def add_prompt(request: PromptRequest) -> dict[str, Any]:
    """Add a custom question. Immediate -- it writes a file, it runs nothing.

    Built-ins cannot be replaced: they ship in the package so that every
    deployment's `yolo` means the same thing, and a request that could shadow
    one would make a run unreproducible from the repo.
    """
    try:
        return service.prompt_add(request.name, request.instruction,
                                  shape=request.shape, about=request.about,
                                  fields=request.fields,
                                  summary=request.summary)
    except library.Protected as exc:
        # 409, not 422: the request is well-formed and the name exists. There
        # is nothing to correct except which name it asks for.
        raise HTTPException(409, {"error": str(exc)}) from None
    except library.PromptError as exc:
        raise HTTPException(422, {"problems": str(exc).split("; ")}) from None


@app.delete("/prompts/{name}", status_code=204, tags=["prompts"])
def delete_prompt(name: str) -> None:
    """Remove a custom question. Descriptions already written are untouched.

    Nothing cascades: a description cost a paid call, and it records the
    question it was asked, so deleting the question does not make the answer
    untrue. It only stops new runs asking it.
    """
    try:
        service.prompt_remove(name)
    except library.Protected as exc:
        raise HTTPException(409, {"error": str(exc)}) from None
    except library.PromptError as exc:
        raise HTTPException(404, {"error": str(exc)}) from None


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
    from falconvar.shared.storage import sinks
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
    """One search. The scope is a SET of videos, and `level` picks granularity.

    There is no second endpoint for "search every video" or for "which video":
    one video, three, or all of them is the same question asked over a
    different set, and a set of one is not a special case.
    """

    query: str
    video_ids: Optional[list[str]] = Field(
        None, description="which videos to search. Omit for every video")
    video_id: Optional[str] = Field(
        None, description="shorthand for a scope of one. `video_ids` wins")
    level: str = Field(
        "moment", description="`moment` ranks chunks within the scope; "
                              "`video` ranks whole videos by their summary. "
                              "The moment filters do not apply to `video`")
    moments: int = 5
    sampler: Optional[str] = Field(
        None, description="narrow to one pairing, e.g. `clip:text`")
    question: Optional[str] = Field(
        None, description="narrow to one question across every sampler that "
                          "asked it, e.g. `text`. Either filter gives up the "
                          "agreement signal: a chunk contributes fewer terms, "
                          "so scores fall")
    strategy: Optional[str] = Field(
        None, description="one sampler's whole output, whatever it was asked, "
                          "e.g. `clip`. Not a prefix of `sampler`: a bare id "
                          "like `clip` means the question IS the strategy name")
    chunk_ids: Optional[list[int]] = Field(
        None, description="narrow to a set of chunks. The drill-down: search, "
                          "read the ids back, then ask for more about those")
    window: int = Field(
        0, ge=0, le=20,
        description="widen `chunk_ids` by this many neighbours each side -- "
                    "`more context around chunk 6` is usually 5, 6, 7")
    after: Optional[float] = Field(
        None, description="seconds. Resolved to chunk ids through the grid, "
                          "which is the one place a span is stored")
    before: Optional[float] = Field(None, description="seconds")
    structured: Optional[dict[str, Any]] = Field(
        None, description="exact structured values, e.g. "
                          "{\"severity\": \"severe\"}. Only meaningful where "
                          "a shape fixed the vocabulary with `one_of`")
    candidates: int = Field(
        20, ge=1, le=500,
        description="units ranked per half before they are fused. Deeper is a "
                    "better fusion and a slower query")
    embedder: Optional[str] = Field(
        None, description="a provider, or provider/model -- the one that built "
                          "the index. Blank resolves exactly as `embed` does")
    index: str = workflow.Options.index


@app.post("/search", tags=["search"])
def search(request: SearchRequest) -> dict[str, Any]:
    """Search. Immediate -- one embedding call and one query.

    **Scope is a set.** `video_ids` names the videos to search; omitting it
    searches every one. A single `video_id` is the one-element shorthand, not a
    different route -- searching one video, three, or all of them is the same
    question over a different set.

    **`level` picks granularity.** `moment` ranks chunks and honours every
    filter; `video` ranks whole videos by their summary out of
    `video_embeddings`, which answers *which video* rather than *which twenty
    seconds*. They are one endpoint but never one ranking: a whole-video
    "moment" beside real ones is a result nobody can play.

    The query is embedded with the embedder named here, which must be the one
    that built the index: a mismatch across widths fails loudly, but two models
    of the same width return a well-formed ranking that means nothing.
    """
    if request.level not in ("moment", "video"):
        raise HTTPException(422, {"error": f"unknown level {request.level!r}",
                                  "known": ["moment", "video"]})
    scope = request.video_ids
    if scope is None and request.video_id:
        scope = [request.video_id]

    if request.level == "video":
        # The moment filters narrow inside a video, so they have nothing to say
        # about which video. Said out loud rather than ignored.
        ignored = [name for name, value in
                   (("sampler", request.sampler), ("question", request.question),
                    ("strategy", request.strategy), ("chunk_ids", request.chunk_ids),
                    ("after", request.after), ("before", request.before),
                    ("structured", request.structured)) if value]
        try:
            found = service.search_videos(request.query,
                                          embedder=request.embedder,
                                          limit=request.moments)
        except Exception as exc:                          # noqa: BLE001
            raise HTTPException(422, {"error": f"{type(exc).__name__}: {exc}"}) from None
        if scope:
            found = [v for v in found if v["video_id"] in scope]
        out: dict[str, Any] = {"query": request.query, "level": "video",
                               "scope": scope, "videos": found}
        if not found:
            out["note"] = ("nothing in video_embeddings for this embedder -- "
                           "run `aggregates --tier llm --index supabase`, which "
                           "stores each summary as its video's vector")
        if ignored:
            out["ignored"] = (f"{', '.join(ignored)} narrow inside a video, so "
                              "they do not apply to level=video")
        return out

    try:
        found = service.search(request.query, video_ids=scope,
                               embedder=request.embedder,
                               moments=request.moments, sampler=request.sampler,
                               question=request.question,
                               strategy=request.strategy,
                               chunk_ids=request.chunk_ids,
                               window=request.window,
                               after=request.after, before=request.before,
                               structured=request.structured,
                               candidates=request.candidates,
                               index_name=request.index)
    except FileNotFoundError as exc:
        raise HTTPException(404, {"error": str(exc)}) from None
    except (KeyError, ValueError) as exc:
        raise HTTPException(422, {"error": str(exc)}) from None
    except (EmbedderUnavailable, db.DatabaseUnavailable) as exc:
        # The deployment, not the request: no key, a local server that is
        # down, a model it does not serve, a database missing its RPC. A 500
        # said none of that.
        raise HTTPException(503, {"error": str(exc)}) from None
    return {"query": request.query, "level": "moment", "scope": scope,
            "moments": found["moments"], "notes": found["notes"]}


# ------------------------------------------------------------- reading rows

# The rows a run wrote, filtered and paged. Its own module because nothing in
# it touches the pipeline: it reads Postgres under the publishable key.
app.include_router(browse.router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """The app when there is one, the schema otherwise.

    Decided from the directory rather than assumed: `web/` is optional, and a
    redirect to a mount that does not exist is a 404 that reads as breakage
    rather than as a client nobody installed.
    """
    return RedirectResponse("/app/" if WEB.exists() else "/docs")


WEB = Path(__file__).resolve().parent.parent / "web"


class Client(StaticFiles):
    """The web app, revalidated on every load.

    `StaticFiles` sends `last-modified` and no `Cache-Control`, which leaves a
    browser free to guess a freshness lifetime from the file's age -- and it
    does. Measured here: an edited `app.js` was served from disk cache without
    a request, so the page ran the previous version and the fault looked like
    the *new* code being wrong. `no-cache` still allows a 304; it only forbids
    answering without asking.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["cache-control"] = "no-cache"
        return response


if WEB.exists():
    # Mounted last, and under a prefix. `html=True` serves `index.html` for the
    # directory, so `/app/` is the page.
    #
    # A prefix rather than `/`: a mount at the root shadows nothing already
    # declared, but it would make every future route a question of whether a
    # file of that name exists, and a 404 from a static directory reads as a
    # missing page rather than a missing route.
    app.mount("/app", Client(directory=WEB, html=True), name="app")
