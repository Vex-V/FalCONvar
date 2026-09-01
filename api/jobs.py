"""Work that outlasts a request.

Processing a video takes minutes -- one decode pass, a Whisper pass, a
describer call per (chunk, sampler) -- so an HTTP handler cannot wait for it.
A job is started, an id comes back, and the caller polls.

**One worker, not a pool.** Every heavy stage here contends for the same 8 GiB
GPU: CLIP and YOLO during ingest, Whisper and pyannote during the audio pass.
Running two videos at once does not halve the wall clock, it doubles the
resident weights and invites an allocator failure halfway through the more
expensive of them. A queue of one is the honest shape of the hardware, and it
makes "why is my job still queued" answerable rather than mysterious.

**State lives in memory and dies with the process.** That is a real limitation
and not a hidden one: `GET /jobs` says so, and the artifacts a finished job
produced are on disk and in Postgres regardless. Restarting loses the record
that a job ran, never the thing it made.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Job:
    """One unit of background work and everything a poller may ask about it."""

    id: str
    kind: str
    video_id: str
    state: str = "queued"                 # queued | running | done | failed
    stage: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    queued_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    #: Every progress callback, in order. A poller that missed the middle of a
    #: run can still see what happened rather than only where it ended up.
    history: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        elapsed = ((self.finished_at or time.time()) - self.started_at
                   if self.started_at else None)
        return {
            "id": self.id, "kind": self.kind, "video_id": self.video_id,
            "state": self.state, "stage": self.stage, "detail": self.detail,
            "result": self.result, "error": self.error,
            "queued_at": self.queued_at, "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": round(elapsed, 2) if elapsed is not None else None,
            "history": self.history,
        }


class Runner:
    """A single background worker and the jobs it has seen."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: "queue.Queue[tuple[Job, Callable]]" = queue.Queue()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._work, daemon=True,
                                        name="falconvar-worker")
        self._thread.start()

    def submit(self, kind: str, video_id: str, work: Callable[[Job], Any]) -> Job:
        """Queue ``work``. It is handed the Job so it can report progress."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, video_id=video_id)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put((job, work))
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.queued_at,
                          reverse=True)

    def pending(self) -> int:
        return self._queue.qsize()

    def _work(self) -> None:
        while True:
            job, work = self._queue.get()
            job.state, job.started_at = "running", time.time()
            try:
                job.result = work(job)
                job.state = "done"
            except Exception as exc:                    # noqa: BLE001
                # The message and the traceback, because a failure four stages
                # into a pipeline is not diagnosable from its last line alone
                # and there is no terminal here to have watched it happen.
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.detail = {"traceback": traceback.format_exc()[-4000:]}
            finally:
                job.finished_at = time.time()
                self._queue.task_done()


def progress(job: Job) -> Callable[[str, dict], None]:
    """An `on_progress` callback that writes into a job.

    The other half of the split `orchestrate.process` was built around: the CLI
    prints these, a job stores the latest and appends to a history.
    """
    def report(stage: str, detail: dict) -> None:
        job.stage = stage
        job.detail = detail
        job.history.append({"stage": stage, "at": round(time.time(), 3),
                            **{k: v for k, v in detail.items() if k != "traceback"}})
    return report
