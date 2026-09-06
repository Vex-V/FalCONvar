"""Import everything and report what is actually loaded.

Installing anything into this environment can silently move a dependency the
pipeline relies on. Adding PaddleOCR downgraded numpy 2.4.4 -> 2.3.5 and put an
opencv-contrib 4.10 alongside the opencv-python 5.0 that was already there, so
`cv2.__version__` changed under the pipeline without a single line of it being
touched. Nothing failed loudly; it just became a different environment.

So this imports every external library and every internal module, exercises
each one just enough to prove it loaded a working binary rather than a stub,
and prints versions. Run it after any install.

    python -m falconvar.imports
    python -m falconvar.imports --verbose

Named ``imports`` rather than ``import`` because ``import`` is a keyword: a
module called that can be run as a script but never imported, which defeats
the point of being able to call it from a test.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class Result:
    name: str
    ok: bool
    version: str = ""
    detail: str = ""
    error: str = ""
    optional: bool = False


# --------------------------------------------------------------------------- #
# external libraries -- import, then do the smallest real thing
# --------------------------------------------------------------------------- #

def _check_numpy():
    import numpy as np
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert float(a.sum()) == 15.0
    return np.__version__, f"dtype {a.dtype}"


def _check_cv2():
    import cv2
    import numpy as np
    img = np.zeros((16, 24, 3), dtype=np.uint8)
    small = cv2.resize(img, (12, 8), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small)
    assert ok and len(buf) > 0
    rot = cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)
    assert rot.shape[:2] == (12, 8)
    # Several opencv distributions can coexist and shadow each other; the file
    # that actually got imported is the only reliable identification.
    return cv2.__version__, Path(cv2.__file__).parent.name


def _check_av():
    import av
    # A codec lookup proves the compiled FFmpeg libraries are reachable.
    codec = av.codec.Codec("h264", "r")
    return av.__version__, f"h264 decoder: {codec.long_name[:38]}"


def _check_torch():
    import torch
    x = torch.randn(8, 8)
    assert (x @ x).shape == (8, 8)
    if torch.cuda.is_available():
        g = torch.randn(8, 8, device="cuda")
        _ = (g @ g).sum().item()          # forces a real kernel launch
        return torch.__version__, f"cuda {torch.version.cuda} on {torch.cuda.get_device_name(0)}"
    return torch.__version__, "CPU only"


def _check_transformers():
    import transformers
    from transformers import CLIPModel, CLIPProcessor  # noqa: F401
    return transformers.__version__, "CLIP classes importable"


def _check_ultralytics():
    import ultralytics
    from ultralytics import YOLO, YOLOWorld  # noqa: F401
    return ultralytics.__version__, "YOLO + YOLOWorld importable"


def _check_easyocr():
    import easyocr  # noqa: F401
    return metadata.version("easyocr"), "detector used; recognition never runs"


def _check_scenedetect():
    import scenedetect
    from scenedetect import ContentDetector, FrameTimecode
    d = ContentDetector(threshold=27.0)
    assert hasattr(d, "process_frame")
    _ = FrameTimecode(0, 30.0)
    return scenedetect.__version__, "ContentDetector.process_frame present"


def _check_openai():
    import openai
    from openai import OpenAI  # noqa: F401
    return metadata.version("openai"), "describer backend; not needed for stub runs"


def _check_qdrant():
    from qdrant_client import QdrantClient  # noqa: F401
    return metadata.version("qdrant-client"), "local vector index; runs embedded"


def _check_supabase():
    import supabase
    from supabase import create_client  # noqa: F401
    return metadata.version("supabase"), "manifest sink; not needed for file runs"


def _check_paddleocr():
    from paddleocr import TextDetection
    TextDetection(model_name="PP-OCRv5_mobile_det")
    return metadata.version("paddleocr"), "TextDetection constructed"


def _check_faster_whisper():
    from faster_whisper import WhisperModel  # noqa: F401

    from falconvar.audio import cuda
    loaded = cuda.enable()
    # The import proves nothing about CUDA: CTranslate2 resolves cublas at the
    # first encode, so a run can import cleanly and fail minutes later.
    need = [n for n in ("cublasLt64_12.dll", "cublas64_12.dll") if n not in loaded]
    detail = ("cuBLAS preloaded" if not need
              else f"CPU only -- missing {', '.join(need)}")
    return metadata.version("faster-whisper"), detail


def _check_pyannote():
    import pyannote.audio
    from pyannote.audio import Pipeline  # noqa: F401
    return pyannote.audio.__version__, "gated: needs HF_TOKEN and accepted terms"


EXTERNAL: list[tuple[str, Callable, bool]] = [
    ("numpy", _check_numpy, False),
    ("opencv (cv2)", _check_cv2, False),
    ("av (PyAV)", _check_av, False),
    ("torch", _check_torch, False),
    ("transformers", _check_transformers, False),
    ("ultralytics", _check_ultralytics, False),
    ("easyocr", _check_easyocr, False),
    ("scenedetect", _check_scenedetect, False),
    ("openai", _check_openai, True),
    ("qdrant-client", _check_qdrant, True),
    ("supabase", _check_supabase, True),
    ("faster-whisper", _check_faster_whisper, True),
    ("pyannote.audio", _check_pyannote, True),
    ("paddleocr", _check_paddleocr, True),
]


# --------------------------------------------------------------------------- #
# internal modules and the names other code imports from them
# --------------------------------------------------------------------------- #

INTERNAL: list[tuple[str, tuple[str, ...]]] = [
    ("falconvar.timeline", ("Timeline", "uniform", "from_cuts", "enforce")),
    ("falconvar.video.ingest.source.types", ("Frame", "SourceInfo")),
    ("falconvar.video.ingest.source.probe", ("probe", "UnusableSource", "MAX_PLAUSIBLE_FPS")),
    ("falconvar.video.ingest.source.reader", ("read_frames", "ROTATIONS")),
    ("falconvar.video.ingest.source.decimate", ("Decimator",)),
    ("falconvar.video.ingest.source.fetch", ("FrameFetcher",)),
    ("falconvar.video.ingest.source", ("Frame", "SourceInfo", "probe", "read_frames",
                            "Decimator", "FrameFetcher", "UnusableSource")),
    ("falconvar.video.ingest.chunker.base", ("Chunker",)),
    ("falconvar.video.ingest.chunker.uniform", ("UniformChunker",)),
    ("falconvar.video.ingest.chunker.scene", ("SceneChunker",)),
    ("falconvar.video.ingest.chunker.fixed", ("FixedChunker",)),
    ("falconvar.video.ingest.chunker", ("Chunker", "UniformChunker", "build", "available")),
    ("falconvar.video.ingest.samplers.base", ("Sampler",)),
    ("falconvar.video.ingest.samplers.uniform", ("UniformSampler",)),
    ("falconvar.video.ingest.samplers.scene", ("ClipChangeSampler",)),
    ("falconvar.video.ingest.samplers.detection", ("DetectionChangeSampler",)),
    ("falconvar.video.ingest.samplers.people", ("PersonChangeSampler",)),
    ("falconvar.video.ingest.samplers.objects", ("ObjectChangeSampler",)),
    ("falconvar.video.ingest.samplers.ocr", ("TextChangeSampler",)),
    ("falconvar.video.ingest.samplers.components.detectors", ("Detection", "ObjectDetector", "YoloPersonDetector",
                                        "OpenVocabDetector", "TextRegionDetector",
                                        "weight_path", "WEIGHTS_DIR")),
    ("falconvar.video.ingest.samplers.components.embedders", ("FrameEmbedder", "CLIPEmbedder")),
    ("falconvar.video.ingest.samplers.components", ("Detection", "RegionDescriptor",
                                        "FrameEmbedder", "CLIPEmbedder")),
    ("falconvar.video.ingest.samplers.components.descriptors", ("RegionDescriptor", "CropEmbeddingDescriptor",
                                          "BoxGeometryDescriptor", "TextLayoutDescriptor")),
    ("falconvar.video.ingest.samplers", ("Sampler", "UniformSampler", "build", "available")),
    ("falconvar.video.ingest.output.base", ("ManifestSink",)),
    ("falconvar.video.ingest.output.manifest", ("FileManifestWriter", "MANIFEST_VERSION")),
    ("falconvar.video.ingest.output.multi", ("MultiSink",)),
    ("falconvar.fanout", ("FanOut",)),
    ("falconvar.db", ("client_from_env", "fetch_manifest", "fetch_descriptions",
                 "manifest_header", "load_env")),
    ("falconvar.video.ingest.output.supabase_manifest", ("SupabaseManifestWriter",)),
    ("falconvar.video.ingest.output.store", ("FrameStore",)),
    ("falconvar.video.ingest.output", ("FileManifestWriter", "ManifestSink", "MultiSink",
                            "SupabaseManifestWriter", "FrameStore",
                            "MANIFEST_VERSION")),
    ("falconvar.video.ingest.pipeline", ("ingest", "Chunk", "Result")),
    ("falconvar.video.ingest.calibrate", ("analyse", "render", "replay", "collect", "Report", "Window")),
    ("falconvar.video.ingest.driver", ("report", "main")),
    # recovery is deliberately standalone: it must import nothing from falconvar,
    # so that a manifest plus this one file is enough to rebuild a store.
    ("falconvar.video.describe.describers.base", ("Describer",)),
    ("falconvar.video.describe.describers.stub", ("StubDescriber",)),
    ("falconvar.video.describe.describers", ("Describer", "StubDescriber", "build", "available")),
    ("falconvar.video.describe.input.frames", ("FrameSource", "LoadedFrame", "StoreUnavailable")),
    ("falconvar.video.describe.input.follow", ("follow_chunks", "client_from_env")),
    ("falconvar.video.describe.input.manifest", ("from_file", "from_supabase", "header")),
    ("falconvar.video.describe.input", ("FrameSource", "StoreUnavailable", "follow_chunks",
                             "from_file", "from_supabase", "header")),
    ("falconvar.video.describe.vlm.prompts", ("SYSTEM", "BY_SAMPLER", "for_sampler")),
    ("falconvar.video.describe.vlm.openai_client", ("OpenAIDescriber", "DescriberUnavailable",
                                         "DEFAULT_MODEL")),
    ("falconvar.video.describe.vlm", ("OpenAIDescriber", "DescriberUnavailable", "DEFAULT_MODEL")),
    ("falconvar.video.describe.reader", ("describe", "chunks_of")),
    ("falconvar.video.describe.output.base", ("DescriptionSink",)),
    ("falconvar.video.describe.output.document", ("DescriptionDocument", "fingerprint",
                                       "DESCRIPTION_VERSION")),
    ("falconvar.video.describe.output.multi", ("MultiDescriptionSink",)),
    ("falconvar.video.describe.output.supabase", ("SupabaseDescriptions",)),
    ("falconvar.video.describe.output", ("DescriptionDocument", "DescriptionSink",
                              "MultiDescriptionSink", "SupabaseDescriptions",
                              "fingerprint", "DESCRIPTION_VERSION")),

    ("falconvar.audio.source", ("probe", "load", "Track", "AudioInfo", "SAMPLE_RATE")),
    ("falconvar.audio.cuda", ("enable", "NEEDED")),
    ("falconvar.audio.transcribe.base", ("Transcriber", "Transcript", "Segment", "Word")),
    ("falconvar.audio.transcribe.stub", ("StubTranscriber",)),
    ("falconvar.audio.transcribe.whisper", ("WhisperTranscriber", "TranscriberUnavailable")),
    ("falconvar.audio.transcribe", ("Transcriber", "Transcript", "build", "available")),
    ("falconvar.audio.diarize.base", ("Diarizer", "Diarization", "Turn")),
    ("falconvar.audio.diarize.pyannote_diarizer", ("PyannoteDiarizer", "DiarizerUnavailable")),
    ("falconvar.audio.diarize", ("Diarizer", "Diarization", "NoDiarizer", "build", "available")),
    ("falconvar.audio.align", ("attribute", "split_on_speaker", "speaker_for")),
    ("falconvar.audio.segment.cut", ("to_chunks",)),
    ("falconvar.audio.segment", ("build", "available", "to_chunks", "vad_cuts",
                            "speaker_cuts", "speech_spans")),
    ("falconvar.audio.output.base", ("TranscriptSink",)),
    ("falconvar.audio.output.document", ("TranscriptDocument", "build_document",
                                    "TRANSCRIPT_VERSION")),
    ("falconvar.audio.output.multi", ("MultiTranscriptSink",)),
    ("falconvar.audio.output.supabase", ("SupabaseTranscript",)),
    ("falconvar.audio.output", ("TranscriptDocument", "TranscriptSink",
                           "MultiTranscriptSink", "SupabaseTranscript",
                           "build_document")),
    ("falconvar.audio.reader", ("listen", "cut", "Result")),

    ("falconvar.embed.defaults", ("index", "embedder", "model", "describe")),
    ("falconvar.embed.units", ("Unit", "embedder_key", "collection_name", "text_hash")),
    ("falconvar.embed.embedders.base", ("Embedder",)),
    ("falconvar.embed.embedders.local", ("LocalEmbedder", "TextTooLong", "KNOWN")),
    ("falconvar.embed.embedders", ("Embedder", "build", "available")),
    ("falconvar.embed.index.base", ("Hit", "VectorIndex")),
    ("falconvar.embed.index.qdrant", ("QdrantIndex",)),
    ("falconvar.embed.index.pgvector", ("PgVectorIndex",)),
    ("falconvar.embed.index.multi", ("MultiIndex",)),
    ("falconvar.embed.index", ("Hit", "MultiIndex", "QdrantIndex", "PgVectorIndex",
                          "BACKENDS", "DEFAULT_QDRANT_PATH", "build")),
    ("falconvar.embed.indexer", ("index_units", "IndexResult")),

    ("falconvar.retrieve.search", ("Moment", "search", "to_moments")),
    ("falconvar.llm", ("complete", "client", "api_key", "LLMUnavailable")),
    ("falconvar.aggregate.base", ("Aggregator", "Context", "TIERS")),
    ("falconvar.aggregate.stats", ("StatsAggregator",)),
    ("falconvar.aggregate.speakers", ("SpeakerStatsAggregator",)),
    ("falconvar.aggregate.novelty", ("NoveltyAggregator",)),
    ("falconvar.aggregate.summary", ("SummaryAggregator",)),
    ("falconvar.aggregate.chapters", ("ChaptersAggregator",)),
    ("falconvar.aggregate.events", ("EventsAggregator",)),
    ("falconvar.aggregate.ner", ("NERAggregator",)),
    ("falconvar.aggregate.sentiment", ("SentimentAggregator",)),
    ("falconvar.aggregate.llm", ("chunk_lines", "pick_sources", "resolve_span")),
    ("falconvar.aggregate.output", ("AggregateDocuments", "AggregateSink",
                               "MultiAggregateSink", "SupabaseAggregates")),
    ("falconvar.aggregate.reader", ("aggregate", "context_for", "Result")),
    ("falconvar.aggregate", ("available", "build", "by_tier", "resolve_order", "TIERS")),
    ("falconvar.embed.summaries", ("index_summary", "search")),

    ("falconvar.orchestrate", ("process", "validate", "Options", "Outcome", "POLICIES")),

    # api/ is optional and sits outside falconvar, but a broken import there is a
    # broken deployment, and this is the only thing that checks anything.
    ("api.jobs", ("Runner", "Job", "progress")),
    ("api.service", ("ingest", "describe", "embed", "search", "videos",
                     "build_samplers", "available")),
    ("api.main", ("app", "runner", "safe_id")),

    ("falconvar.recovery.recreate", ("rebuild_sampled", "rebuild_decimated", "compare",
                                "targets_from", "Fetcher", "StoreWriter")),
    ("falconvar.recovery.supabase_manifest", ("fetch", "listing")),
    ("falconvar.recovery.supabase_description", ("fetch", "assemble", "fingerprint")),
]


def check_internal(module_name: str, names: tuple[str, ...]) -> Result:
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        return Result(module_name, False, error=f"{type(exc).__name__}: {exc}")
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        return Result(module_name, False, error=f"missing: {', '.join(missing)}")
    return Result(module_name, True, detail=f"{len(names)} names")


# Every command-line entry point, checked by actually invoking it. Importing a
# module proves nothing about whether it runs: a missing `import argparse` used
# only inside main() imports cleanly and fails at the first call.
ENTRYPOINTS = [
    "falconvar.video.ingest.driver",
    "falconvar.video.ingest.calibrate",
    "falconvar.video.describe.driver",
    "falconvar.audio.driver",
    "falconvar.driver",
    "falconvar.aggregate.driver",
    "falconvar.embed.driver",
    "falconvar.retrieve.driver",
    "falconvar.recovery.recreate",
    "falconvar.recovery.supabase_manifest",
    "falconvar.recovery.supabase_description",
    "falconvar.imports",
]


def check_entrypoints() -> list[Result]:
    """Run each CLI with --help in a subprocess and require a clean exit."""
    import subprocess

    out: list[Result] = []
    for module in ENTRYPOINTS:
        if module == "falconvar.imports":
            continue                       # would recurse
        try:
            proc = subprocess.run(
                [sys.executable, "-m", module, "--help"],
                capture_output=True, text=True, timeout=180,
            )
        except Exception as exc:
            out.append(Result(module, False, error=f"{type(exc).__name__}: {exc}"))
            continue
        if proc.returncode == 0:
            out.append(Result(module, True, detail="--help exits 0"))
        else:
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            out.append(Result(module, False,
                              error=tail[-1] if tail else f"exit {proc.returncode}"))
    return out


def check_recovery_standalone() -> Result:
    """recovery/ must not import the pipeline -- that independence is the claim."""
    import ast

    # Every file in recovery/, not just recreate.py: the recovery kit is
    # handed over whole, and one leaked import anywhere in it breaks the
    # claim that a manifest and a video are enough.
    sources = sorted((Path(__file__).resolve().parent / "recovery").glob("*.py"))
    leaked = set()
    for src in sources:
        tree = ast.parse(src.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("falconvar"):
                leaked.add(f"{src.name}: {node.module}")
            if isinstance(node, ast.Import):
                leaked |= {f"{src.name}: {a.name}"
                           for a in node.names if a.name.startswith("falconvar")}
    if leaked:
        return Result("recovery imports nothing from falconvar", False,
                      error=f"leaked: {', '.join(sorted(leaked))}")
    return Result("recovery imports nothing from falconvar", True,
                  detail=f"standalone ({len(sources)} files)")


#: describe/ may reach into ingest/ for exactly this, and nothing else.
DESCRIBE_MAY_IMPORT = {"falconvar.video.ingest.output": {"FrameStore"}}


def check_describe_boundary() -> Result:
    """describe/ talks to the manifest and the frame store. Not the pipeline.

    The manifest arrives as parsed JSON and the chunk stream as rows, so
    neither needs an import at all. The store is a real class and is imported;
    anything else would mean the describe stage had started depending on how
    ingest works rather than on what it produced.
    """
    import ast

    root = Path(__file__).resolve().parent / "video" / "describe"
    sources = sorted(root.rglob("*.py"))
    leaked = []
    for src in sources:
        tree = ast.parse(src.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if not module.startswith("falconvar.video.ingest"):
                    continue
                allowed = DESCRIBE_MAY_IMPORT.get(module, set())
                for alias in node.names:
                    if alias.name not in allowed:
                        leaked.append(f"{src.name}: {module}.{alias.name}")
            elif isinstance(node, ast.Import):
                leaked += [f"{src.name}: {a.name}" for a in node.names
                           if a.name.startswith("falconvar.video.ingest")]
    if leaked:
        return Result("describe touches only manifest + store", False,
                      error=f"leaked: {', '.join(sorted(leaked))}")
    return Result("describe touches only manifest + store", True,
                  detail=f"FrameStore only ({len(sources)} files)")


def check_registries() -> list[Result]:
    """The registries decide what the CLI can build, so a name that resolves to
    nothing is a break the import check above would not catch."""
    out: list[Result] = []
    try:
        from falconvar.video.ingest import chunker as chunker_mod
        from falconvar.video.ingest import samplers as samplers_mod
    except Exception as exc:
        return [Result("registries", False, error=f"{type(exc).__name__}: {exc}")]

    # Entries that cannot be built from nothing. `fixed` carries a grid decided
    # by another pass, so a no-argument build is not a shape it ever has -- the
    # check supplies a one-chunk timeline rather than reporting the absence of
    # an argument as a broken registry.
    from falconvar.timeline import uniform as _uniform_timeline

    needs: dict[str, dict] = {"fixed": {"timeline": _uniform_timeline(10.0, 10.0)}}

    for label, mod, heavy in (("samplers", samplers_mod, {"clip", "yolo", "objects", "text"}),
                              ("chunkers", chunker_mod, {"scene"})):
        for name in mod.available():
            if name in heavy:
                # Resolving these loads model weights; check only that the
                # registry knows how to reach them.
                out.append(Result(f"{label}:{name}", True, detail="registered (lazy)"))
                continue
            try:
                obj = mod.build(name, **needs.get(name, {}))
                out.append(Result(f"{label}:{name}", True, detail=type(obj).__name__))
            except Exception as exc:
                out.append(Result(f"{label}:{name}", False,
                                  error=f"{type(exc).__name__}: {exc}"))
    return out


def duplicate_distributions() -> list[str]:
    """Packages that provide the same import name -- the silent-shadowing case.

    opencv-python, opencv-contrib-python and opencv-python-headless all install
    a module called ``cv2``. Whichever wins is decided by install order, and
    nothing warns you.
    """
    families = {
        "cv2": ("opencv-python", "opencv-contrib-python", "opencv-python-headless",
                "opencv-contrib-python-headless"),
        "paddle": ("paddlepaddle", "paddlepaddle-gpu"),
    }
    warnings = []
    for module, names in families.items():
        found = []
        for n in names:
            try:
                found.append(f"{n}=={metadata.version(n)}")
            except metadata.PackageNotFoundError:
                pass
        if len(found) > 1:
            warnings.append(f"{module}: {', '.join(found)}")
    return warnings


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Import everything and report what loaded.")
    ap.add_argument("--verbose", "-v", action="store_true", help="full tracebacks")
    args = ap.parse_args()

    print(f"  python {sys.version.split()[0]}   {sys.prefix}")
    print()
    print("  EXTERNAL")
    failures: list[Result] = []
    for label, fn, optional in EXTERNAL:
        try:
            version, detail = fn()
            r = Result(label, True, version, detail, optional=optional)
        except Exception as exc:
            r = Result(label, False, error=f"{type(exc).__name__}: {exc}", optional=optional)
            if args.verbose:
                traceback.print_exc()
        mark = "ok  " if r.ok else ("SKIP" if r.optional else "FAIL")
        if r.ok:
            print(f"    {mark} {r.name:<16} {r.version:<12} {r.detail}")
        else:
            print(f"    {mark} {r.name:<16} {r.error[:88]}")
            if not r.optional:
                failures.append(r)

    print()
    print("  INTERNAL")
    for module_name, names in INTERNAL:
        r = check_internal(module_name, names)
        if r.ok:
            print(f"    ok   {r.name:<38} {r.detail}")
        else:
            print(f"    FAIL {r.name:<38} {r.error[:70]}")
            failures.append(r)

    print()
    print("  INVARIANTS")
    for r in (check_recovery_standalone(), check_describe_boundary()):
        print(f"    {'ok  ' if r.ok else 'FAIL'} {r.name:<38} {r.detail or r.error}")
        if not r.ok:
            failures.append(r)

    print()
    print("  ENTRY POINTS")
    for r in check_entrypoints():
        if r.ok:
            print(f"    ok   {r.name:<38} {r.detail}")
        else:
            print(f"    FAIL {r.name:<38} {r.error[:70]}")
            failures.append(r)

    print()
    print("  REGISTRIES")
    for r in check_registries():
        if r.ok:
            print(f"    ok   {r.name:<38} {r.detail}")
        else:
            print(f"    FAIL {r.name:<38} {r.error[:70]}")
            failures.append(r)

    dupes = duplicate_distributions()
    if dupes:
        print()
        print("  SHADOWING -- several packages provide the same import name:")
        for d in dupes:
            print(f"    ! {d}")
        print("    Which one wins depends on install order and nothing warns you.")

    print()
    if failures:
        print(f"  {len(failures)} FAILURE(S)")
        return 1
    print("  all imports OK" + ("  (with shadowing warnings above)" if dupes else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
