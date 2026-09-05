"""The loop: one video's documents in, video-level results out.

One pass, one shared `Context`, so the three documents are parsed once however
many aggregators read them. Each result is handed to the sink as it lands
rather than collected and written at the end -- an LLM aggregator failing on
the fifth of eight must not take the four cheap ones with it.

Nothing here decides what an aggregate is or where it goes. It resolves the
order, builds the context, calls each aggregator, and keeps count.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from . import build, resolve_order
from .base import Context


@dataclass
class Result:
    video_id: str
    produced: dict[str, dict] = field(default_factory=dict)
    current: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    empty: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"video_id": self.video_id,
                "produced": sorted(self.produced),
                "current": sorted(self.current),
                "empty": sorted(self.empty),
                "skipped": self.skipped,
                "failed": self.failed,
                "elapsed_s": round(self.elapsed_s, 3)}


def aggregate(
    ctx: Context,
    names: Sequence[str],
    sink: Optional[Any] = None,
    on_progress: Optional[Callable[[str, dict], None]] = None,
    existing: Optional[dict[str, str]] = None,
    force: bool = False,
) -> Result:
    """Run every requested aggregator that can run, in dependency order.

    ``existing`` maps aggregate_id to the inputs fingerprint it was last built
    from. Anything whose fingerprint still matches is left alone: an aggregate
    derived from descriptions that have not changed cannot have changed either,
    and re-running the LLM tier over an unchanged video is the most expensive
    no-op available here. ``force`` rebuilds regardless.
    """
    say = on_progress or (lambda stage, detail: None)
    started = time.perf_counter()
    ordered, skipped = resolve_order(names, ctx.sources)
    result = Result(video_id=ctx.video_id, skipped=skipped)
    fingerprint = ctx.fingerprint()
    say("plan", {"order": ordered, "skipped": skipped, "sources": ctx.sources,
                 "fingerprint": fingerprint})

    for name in ordered:
        if not force and (existing or {}).get(name) == fingerprint:
            result.current.append(name)
            say("current", {"aggregator": name})
            continue
        aggregator = build(name)
        began = time.perf_counter()
        try:
            payload = aggregator.aggregate(ctx)
        except Exception as exc:                        # noqa: BLE001
            # One aggregator's failure is not the run's. A paid LLM call that
            # times out should not lose the arithmetic that already succeeded.
            result.failed[name] = f"{type(exc).__name__}: {exc}"
            say("failed", {"aggregator": name, "error": result.failed[name]})
            continue

        if payload is None:
            # Distinct from an empty result: None means the question does not
            # apply to this video, and recording it as an answer would claim
            # otherwise.
            result.empty.append(name)
            say("empty", {"aggregator": name})
            continue

        record = {
            "aggregate_id": name,
            "video_id": ctx.video_id,
            "tier": aggregator.tier,
            "depends_on": list(aggregator.depends_on),
            "inputs_fingerprint": fingerprint,
            "config": aggregator.config(),
            "elapsed_s": round(time.perf_counter() - began, 3),
            "payload": payload,
        }
        result.produced[name] = record
        if sink is not None:
            sink.write(record)
        say("done", {"aggregator": name, "elapsed_s": record["elapsed_s"]})

    result.elapsed_s = time.perf_counter() - started
    return result


def context_for(video_id: str, out_root: Path = Path("out")) -> Context:
    """The context for one video, read off disk."""
    return Context.from_dir(video_id, Path(out_root) / video_id)
