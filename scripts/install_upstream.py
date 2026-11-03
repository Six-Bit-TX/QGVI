#!/usr/bin/env python3
"""Fetch original dependency repositories under their respective licenses."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "logs": ("https://github.com/RPL-CS-UCL/gs_localization.git", "163f7c82d9c8f9ddbc4e3c18a4cf73ba9f2fbced", "LoGS"),
    "horizon": ("https://github.com/InternRobotics/HorizonGS.git", "56f9c82a4e84796ba7c496429504dd4ab68127f5", "HorizonGS"),
}


def install(name):
    url, revision, folder = SOURCES[name]
    path = ROOT / "third_party" / folder
    if not path.exists():
        subprocess.run(["git", "clone", url, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", revision], check=True)
        subprocess.run(["git", "-C", str(path), "submodule", "update", "--init", "--recursive"], check=True)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RuntimeError(f"Existing {folder} revision differs; refusing to change your checkout")
    if name == "horizon":
        checker = ROOT / "third_party_patches/verify.py"
        verified = subprocess.run([sys.executable, str(checker), str(path)], capture_output=True)
        if verified.returncode:
            dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
            if dirty:
                raise RuntimeError("Existing HorizonGS has unknown changes; refusing to overwrite them")
            patch = ROOT / "third_party_patches/HorizonGS-runtime.patch"
            subprocess.run(["git", "-C", str(path), "apply", "--check", str(patch)], check=True)
            subprocess.run(["git", "-C", str(path), "apply", "--whitespace=nowarn", str(patch)], check=True)
            subprocess.run([sys.executable, str(checker), str(path)], check=True)
    print(f"{folder}: {revision}; retain its LICENSE and submodule licenses")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", action="store_true")
    parser.add_argument("--horizon", action="store_true")
    args = parser.parse_args()
    if not (args.logs or args.horizon):
        parser.error("Select --logs and/or --horizon")
    for name in SOURCES:
        if getattr(args, name):
            install(name)
