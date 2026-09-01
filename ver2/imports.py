"""Import everything and report what is actually loaded.

Installing anything into this environment can silently move a dependency the
pipeline relies on. Adding PaddleOCR downgraded numpy 2.4.4 -> 2.3.5 and put an
opencv-contrib 4.10 alongside the opencv-python 5.0 that was already there, so
`cv2.__version__` changed under the pipeline without a single line of it being
touched. Nothing failed loudly; it just became a different environment.

So this imports every external library and every internal module, exercises
each one just enough to prove it loaded a working binary rather than a stub,
and prints versions. Run it after any install.

    python -m ver2.imports
    python -m ver2.imports --verbose

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
from typing import Callable, Optional

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


def _check_paddleocr():
    from paddleocr import TextDetection
    TextDetection(model_name="PP-OCRv5_mobile_det")
    return metadata.version("paddleocr"), "TextDetection constructed"


EXTERNAL: list[tuple[str, Callable, bool]] = [
    ("numpy", _check_numpy, False),
    ("opencv (cv2)", _check_cv2, False),
    ("av (PyAV)", _check_av, False),
    ("torch", _check_torch, False),
    ("transformers", _check_transformers, False),
    ("ultralytics", _check_ultralytics, False),
    ("easyocr", _check_easyocr, False),
    ("scenedetect", _check_scenedetect, False),
    ("paddleocr", _check_paddleocr, True),
]


# --------------------------------------------------------------------------- #
# internal modules and the names other code imports from them
# --------------------------------------------------------------------------- #

INTERNAL: list[tuple[str, tuple[str, ...]]] = [
    ("ver2.ingest.source.types", ("Frame", "SourceInfo")),
    ("ver2.ingest.source.probe", ("probe", "UnusableSource", "MAX_PLAUSIBLE_FPS")),
    ("ver2.ingest.source.reader", ("read_frames", "ROTATIONS")),
    ("ver2.ingest.source.decimate", ("Decimator",)),
    ("ver2.ingest.source.fetch", ("FrameFetcher",)),
    ("ver2.ingest.source", ("Frame", "SourceInfo", "probe", "read_frames",
                            "Decimator", "FrameFetcher", "UnusableSource")),
    ("ver2.ingest.chunker.base", ("Chunker",)),
    ("ver2.ingest.chunker.uniform", ("UniformChunker",)),
    ("ver2.ingest.chunker.scene", ("SceneChunker",)),
    ("ver2.ingest.chunker", ("Chunker", "UniformChunker", "build", "available")),
    ("ver2.ingest.samplers.base", ("Sampler",)),
    ("ver2.ingest.samplers.uniform", ("UniformSampler",)),
    ("ver2.ingest.samplers.clip", ("ClipChangeSampler",)),
    ("ver2.ingest.samplers.change", ("DetectionChangeSampler", "PersonChangeSampler",
                                     "ObjectChangeSampler", "TextChangeSampler")),
    ("ver2.ingest.samplers.detectors", ("Detection", "ObjectDetector", "YoloPersonDetector",
                                        "OpenVocabDetector", "TextRegionDetector",
                                        "weight_path", "WEIGHTS_DIR")),
    ("ver2.ingest.samplers.embedders", ("FrameEmbedder", "CLIPEmbedder")),
    ("ver2.ingest.samplers.descriptors", ("RegionDescriptor", "CropEmbeddingDescriptor",
                                          "BoxGeometryDescriptor", "TextLayoutDescriptor")),
    ("ver2.ingest.samplers", ("Sampler", "UniformSampler", "build", "available")),
    ("ver2.ingest.output.manifest", ("ManifestWriter", "MANIFEST_VERSION")),
    ("ver2.ingest.output.store", ("FrameStore",)),
    ("ver2.ingest.output", ("ManifestWriter", "FrameStore", "MANIFEST_VERSION")),
    ("ver2.ingest.pipeline", ("ingest", "Chunk", "Result")),
    ("ver2.ingest.calibrate", ("analyse", "render", "replay", "collect", "Report", "Window")),
    ("ver2.ingest.driver", ("report", "main")),
    # recovery is deliberately standalone: it must import nothing from ver2,
    # so that a manifest plus this one file is enough to rebuild a store.
    ("ver2.recovery.recreate", ("rebuild_sampled", "rebuild_decimated", "compare",
                                "targets_from", "Fetcher", "StoreWriter")),
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
    "ver2.ingest.driver",
    "ver2.ingest.calibrate",
    "ver2.recovery.recreate",
    "ver2.imports",
]


def check_entrypoints() -> list[Result]:
    """Run each CLI with --help in a subprocess and require a clean exit."""
    import subprocess

    out: list[Result] = []
    for module in ENTRYPOINTS:
        if module == "ver2.imports":
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

    src = Path(__file__).resolve().parent / "recovery" / "recreate.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    leaked = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ver2"):
            leaked.add(node.module)
        if isinstance(node, ast.Import):
            leaked |= {a.name for a in node.names if a.name.startswith("ver2")}
    if leaked:
        return Result("recovery imports nothing from ver2", False,
                      error=f"leaked: {', '.join(sorted(leaked))}")
    return Result("recovery imports nothing from ver2", True, detail="standalone")


def check_registries() -> list[Result]:
    """The registries decide what the CLI can build, so a name that resolves to
    nothing is a break the import check above would not catch."""
    out: list[Result] = []
    try:
        from ver2.ingest import chunker as chunker_mod
        from ver2.ingest import samplers as samplers_mod
    except Exception as exc:
        return [Result("registries", False, error=f"{type(exc).__name__}: {exc}")]

    for label, mod, heavy in (("samplers", samplers_mod, {"clip", "yolo", "objects", "text"}),
                              ("chunkers", chunker_mod, {"scene"})):
        for name in mod.available():
            if name in heavy:
                # Resolving these loads model weights; check only that the
                # registry knows how to reach them.
                out.append(Result(f"{label}:{name}", True, detail="registered (lazy)"))
                continue
            try:
                obj = mod.build(name)
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
    r = check_recovery_standalone()
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
