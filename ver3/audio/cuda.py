"""Making the CUDA libraries findable on Windows.

CTranslate2 -- the runtime under faster-whisper -- asks Windows for
`cublas64_12.dll` by name, at the moment of the first encode rather than at
import. The DLL is installed: `nvidia-cublas-cu12` puts it in
`site-packages/nvidia/cublas/bin`. Windows will not find it there.

Three things conspire. Since Python 3.8 an extension module's dependencies are
not resolved from `PATH`; `os.add_dll_directory` is the replacement, but it
does not help here either, because the load happens lazily inside a C++
extension that has already been initialised. And the version has to match
exactly: this project's torch is built against CUDA 13 and ships
`cublas64_13.dll`, which is not a substitute -- the name is the contract.

Loading each library by absolute path puts it in the process's module table,
after which CTranslate2's by-name request resolves to the copy already
loaded. Call `enable()` before constructing any CUDA-backed audio model.

The failure this prevents is legible but misleading: `RuntimeError: Library
cublas64_12.dll is not found or cannot be loaded`, on a machine where the file
is present and CUDA works for everything else in the project.
"""

from __future__ import annotations

import ctypes
import site
import sys
from pathlib import Path

#: Loaded in dependency order -- cublas needs cublasLt, so it goes second.
NEEDED = ("cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll")

_done: list[str] | None = None


def enable() -> list[str]:
    """Preload the vendored CUDA libraries. Idempotent; a no-op off Windows."""
    global _done
    if _done is not None:
        return _done
    if sys.platform != "win32":
        _done = []
        return _done

    roots = [Path(p) / "nvidia" for p in site.getsitepackages()]
    roots += [Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"]
    loaded: list[str] = []
    for name in NEEDED:
        for root in roots:
            if not root.is_dir():
                continue
            for dll in sorted(root.glob(f"*/bin/{name}")):
                try:
                    ctypes.WinDLL(str(dll))
                except OSError:
                    continue          # a missing optional dep is not fatal here
                loaded.append(name)
                break
            if name in loaded:
                break
    _done = loaded
    return loaded
