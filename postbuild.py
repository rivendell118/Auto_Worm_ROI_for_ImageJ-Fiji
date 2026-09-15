"""Finish the distributable: synchronize MSVC CRT files, copy models and licenses."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dist" / "AutoWormImageJ"
MODEL_SOURCE = ROOT / "models"
LICENSE_SOURCE = ROOT / "licenses"
CRT_FILES = (
    "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll",
    "vcruntime140_threads.dll", "MSVCP140_ATOMIC_WAIT.dll", "concrt140.dll",
)


def main() -> int:
    errors = []
    internal = OUT / "_internal"
    torch_lib = internal / "torch" / "lib"
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    for destination in (internal, torch_lib):
        if not destination.is_dir():
            errors.append("missing " + str(destination))
            continue
        for name in CRT_FILES:
            source = system32 / name
            if source.is_file():
                shutil.copy2(source, destination / name)
                print("synced", name, "->", destination)

    model_out = OUT / "models"
    if model_out.exists():
        shutil.rmtree(model_out)
    shutil.copytree(MODEL_SOURCE, model_out)
    print("copied models ->", model_out)

    # The NVIDIA license texts and the third-party notices the package owes its
    # users. A build that silently ships without them is the failure this guards
    # against, so a missing source directory is an error, not a skip.
    if not LICENSE_SOURCE.is_dir():
        errors.append("missing " + str(LICENSE_SOURCE))
    else:
        license_out = OUT / "licenses"
        if license_out.exists():
            shutil.rmtree(license_out)
        shutil.copytree(LICENSE_SOURCE, license_out)
        print("copied licenses ->", license_out)

    if errors:
        print("postbuild errors:", "; ".join(errors), file=sys.stderr)
        return 1
    print("postbuild OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
