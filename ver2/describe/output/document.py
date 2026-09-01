"""The description document, written as each description lands.

Same discipline as the manifest: rewrite to a temporary file and ``os.replace``
it into place, so a reader always sees a whole document and never a torn one,
and the file is valid from before the first description rather than only after
the last. A run that dies at chunk 40 of 137 leaves the first 40 usable.

The skeleton is built from the manifest up front -- every chunk, every sampler,
with ``description: null`` -- so the file states what *will* be described as
well as what has been. ``processed`` on a chunk is true only when every one of
its samplers has an answer.

A fingerprint of the manifest's chunks is stored alongside. Descriptions are
about specific frames; if the manifest that named those frames has changed,
the descriptions describe something that is no longer being claimed, and
resuming onto them would silently mix two runs. Re-ingesting a video without
changing the sampling produces the same chunks and the same fingerprint, so
that case correctly resumes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from ..vlm import prompts

DESCRIPTION_VERSION = 1


def fingerprint(manifest: dict[str, Any]) -> str:
    """A short, stable hash of the run that produced this manifest.

    Over the *settings*, not the resulting chunks, for two reasons. It is
    known before a single chunk exists, so a reader following a live ingest
    can compute it at the start; and chunks are a deterministic function of
    these settings anyway -- re-ingesting the same video with the same flags
    was verified to reproduce a byte-identical manifest.

    Two fields are excluded because they say where things are rather than what
    was done: ``source.uri`` and ``config.frame_store``. Describing the same
    footage from a copied file, or from a store in another directory, is the
    same work and must not read as a different run.
    """
    source = {k: v for k, v in manifest["source"].items() if k != "uri"}
    config = {k: v for k, v in (manifest.get("config") or {}).items()
              if k != "frame_store"}
    payload = json.dumps({"source": source, "config": config}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def skeleton(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Every (chunk, sampler) the manifest names, waiting to be described."""
    chunks = []
    for chunk in manifest["chunks"]:
        chunks.append({
            "chunk_id": chunk["chunk_id"],
            "start_ts": chunk["start_ts"],
            "end_ts": chunk["end_ts"],
            "processed": False,
            "samplers": {
                sampler: {
                    "frame_count": block["frame_count"],
                    "frame_indexes": [f["index"] for f in block["frames"]],
                    "description": None,
                    "structured": {},
                }
                for sampler, block in chunk["samplers"].items()
            },
        })
    return chunks


class DescriptionDocument:
    """Keeps a valid description document on disk throughout. A ``DescriptionSink``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.video_id = ""
        self.model: dict[str, Any] = {}
        self.manifest_fingerprint = ""
        self.source: dict[str, Any] = {}
        self.chunks: list[dict[str, Any]] = []
        self.stats: dict[str, Any] = {}
        self.complete = False
        self._resumed: set[tuple[int, str]] = set()

    def begin(self, video_id: str, manifest: dict[str, Any], model: dict[str, Any]) -> None:
        self.video_id = video_id
        self.model = model
        self.manifest_fingerprint = fingerprint(manifest)
        self.source = {
            "uri": manifest["source"]["uri"],
            "video_id": manifest["video_id"],
            "manifest_version": manifest.get("manifest_version"),
        }
        self.chunks = skeleton(manifest)
        self._resumed = self._recover()
        self.complete = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()

    def _recover(self) -> set[tuple[int, str]]:
        """Carry over descriptions from an earlier pass over the same manifest."""
        if not self.path.exists():
            return set()
        try:
            previous = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return set()
        if previous.get("manifest_fingerprint") != self.manifest_fingerprint:
            # Not an error and not resumable: the manifest changed, so these
            # describe frames this run is not being asked about.
            return set()
        if previous.get("model") != self.model:
            # Nor is a different describer's work. Switching from the stub to a
            # real model, or between models, must not be silently skipped as
            # "already described" -- that reads as success and does nothing.
            return set()
        done: set[tuple[int, str]] = set()
        held = {(c["chunk_id"], s): block
                for c in previous.get("chunks", [])
                for s, block in c.get("samplers", {}).items()}
        for chunk in self.chunks:
            for sampler, block in chunk["samplers"].items():
                old = held.get((chunk["chunk_id"], sampler))
                if old and old.get("description") is not None:
                    block.update(old)
                    done.add((chunk["chunk_id"], sampler))
            self._mark(chunk)
        return done

    def existing(self) -> set[tuple[int, str]]:
        return set(self._resumed)

    @staticmethod
    def _mark(chunk: dict[str, Any]) -> None:
        chunk["processed"] = all(block["description"] is not None
                                 for block in chunk["samplers"].values())
        # The chunk's structure is the union of its samplers', which is only
        # unambiguous because no two of them own the same key. Recomputed on
        # every write rather than stored alongside, so it cannot drift from
        # the per-sampler answers it comes from.
        chunk["structured"] = prompts.merge({
            sampler: block.get("structured") or {}
            for sampler, block in chunk["samplers"].items()
        })

    def document(self) -> dict[str, Any]:
        return {
            "description_version": DESCRIPTION_VERSION,
            "video_id": self.video_id,
            "complete": self.complete,
            "manifest_fingerprint": self.manifest_fingerprint,
            "source": self.source,
            "model": self.model,
            "stats": self.stats,
            "chunks": self.chunks,
        }

    def _chunk(self, record: dict[str, Any]) -> dict[str, Any]:
        """The entry for this chunk, created if the skeleton has never seen it.

        A follower has no skeleton: it is handed the manifest header before any
        chunk exists, so every chunk it describes is new. Creating on demand is
        what lets one document writer serve both a finished manifest and a live
        one.
        """
        for chunk in self.chunks:
            if chunk["chunk_id"] == record["chunk_id"]:
                return chunk
        chunk = {
            "chunk_id": record["chunk_id"],
            "start_ts": record.get("start_ts"),
            "end_ts": record.get("end_ts"),
            "processed": False,
            "samplers": {},
        }
        self.chunks.append(chunk)
        self.chunks.sort(key=lambda c: c["chunk_id"])
        return chunk

    def described(self, record: dict[str, Any]) -> None:
        chunk = self._chunk(record)
        block = chunk["samplers"].setdefault(record["sampler"], {})
        block.update({
            "frame_count": record["frame_count"],
            "frame_indexes": record["frame_indexes"],
            "description": record["description"],
            "structured": record.get("structured") or {},
            "elapsed_s": record.get("elapsed_s"),
        })
        self._mark(chunk)
        self._write()

    def finish(self, stats: Optional[dict] = None) -> dict[str, Any]:
        if stats is not None:
            self.stats = stats
        # `all` over nothing is true, and a document holding nothing is the one
        # case where claiming completeness does real harm -- a consumer would
        # read zero descriptions as a finished answer.
        self.complete = bool(self.chunks) and all(
            chunk["processed"] for chunk in self.chunks)
        self._write()
        return self.document()

    def _write(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.document(), indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
