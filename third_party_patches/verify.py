#!/usr/bin/env python3
"""Check the exact patched HorizonGS runtime files without importing CUDA."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(horizon: Path) -> dict:
    directory = Path(__file__).resolve().parent
    receipt = json.loads((directory / "HorizonGS-runtime-provenance.json").read_text())
    patch = directory / receipt["patch"]
    if digest(patch.read_bytes()) != receipt["patch_sha256"]:
        raise ValueError("HorizonGS patch does not match the recorded SHA-256")
    checked = []
    for entry in receipt["files"]:
        file = horizon / entry["path"]
        if not file.is_file() or digest(file.read_bytes()) != entry["patched_sha256"]:
            raise ValueError(f"HorizonGS runtime file differs: {entry['path']}")
        ast.parse(file.read_text())
        checked.append(entry["path"])
    return {
        "status": "passed_source_checks",
        "upstream_commit": receipt["upstream_commit"],
        "verified_files": checked,
        "gpu_tested": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("horizon", type=Path, help="Patched HorizonGS checkout")
    args = parser.parse_args()
    print(json.dumps(verify(args.horizon), indent=2))
