#!/usr/bin/env python3
"""Download byte-identical QGVI assets from a pinned Hugging Face commit.

No model is imported or run. Existing verified files are reused. Hugging Face's
cache resumes incomplete transfers; every completed file is SHA-256 checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "checkpoints" / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed during verification: {path.name}")
    return digest.hexdigest()


def safe_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if not name or "\\" in name or relative.is_absolute() or any(
        part in {"", ".", ".."} for part in name.split("/")
    ):
        raise ValueError(f"Unsafe asset path: {name!r}")
    root = root.resolve()
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Asset escapes destination: {name!r}")
    return path


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Unsupported checkpoint manifest")
    names = set()
    for record in manifest["files"]:
        name = record["path"]
        safe_path(Path.cwd(), name)
        if name in names:
            raise ValueError(f"Duplicate asset path: {name}")
        names.add(name)
        if type(record.get("size_bytes")) is not int or record["size_bytes"] < 0:
            raise ValueError(f"Invalid asset size: {name}")
        checksum = record.get("sha256")
        if checksum is not None and not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError(f"Invalid SHA-256: {name}")
    return manifest


def select_files(manifest: dict, scenes: list[str] | None, groups: list[str] | None) -> list[dict]:
    available_scenes = {r["scene"] for r in manifest["files"]}
    available_groups = {r["group"] for r in manifest["files"]}
    if scenes and set(scenes) - available_scenes:
        raise ValueError("Unknown scene: " + ", ".join(sorted(set(scenes) - available_scenes)))
    if groups and set(groups) - available_groups:
        raise ValueError("Unknown asset group: " + ", ".join(sorted(set(groups) - available_groups)))
    shared_roles = {"portable_scene_configuration", "reference_camera_metadata"}
    records = [r for r in manifest["files"] if (not scenes or r["scene"] in scenes)
               and (not groups or r["group"] in groups or r.get("role") in shared_roles)]
    if not records:
        raise ValueError("No assets match this selection")
    return records


def check_file(path: Path, record: dict) -> bool:
    if record.get("sha256") is None:
        raise ValueError("Asset has no qualified checksum: " + record["path"])
    return (path.is_file() and path.stat().st_size == record["size_bytes"]
            and sha256(path) == record["sha256"])


def publication(manifest: dict, repo_id: str | None, revision: str | None) -> dict:
    remote = dict(manifest.get("huggingface", {}))
    if repo_id:
        remote["repo_id"] = repo_id
    if revision:
        remote["revision"] = revision
    if not remote.get("repo_id") or not re.fullmatch(r"[0-9a-f]{40}", remote.get("revision") or ""):
        raise ValueError("A published repository and immutable 40-character commit are required; "
                         "supply --repo-id and --revision, or use the published manifest")
    if remote.get("repo_type", "model") not in {"model", "dataset"}:
        raise ValueError("Unsupported Hugging Face repository type")
    return remote


def download(manifest: dict, records: list[dict], destination: Path, *, repo_id=None,
             revision=None, endpoint=None, offline=False, retries=3, downloader=None) -> dict:
    remote = publication(manifest, repo_id, revision)
    available = remote.get("available_scenes")
    if available is not None and not repo_id and not revision:
        missing = {r["scene"] for r in records} - set(available)
        if missing:
            raise ValueError("These scenes have not completed remote verification: "
                             + ", ".join(sorted(missing)))
    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub to download checkpoints") from exc
        downloader = hf_hub_download
    completed = []
    for number, record in enumerate(records, 1):
        path = safe_path(destination, record["path"])
        if check_file(path, record):
            disposition = "verified_existing"
        else:
            # Never overwrite unrecognized user data. A transfer interrupted by
            # huggingface_hub remains in its .cache directory, not at this path.
            if path.exists():
                raise ValueError(f"Existing file differs from manifest: {path}; "
                                 "move it aside and retry")
            if offline:
                raise FileNotFoundError(f"Missing verified offline asset: {record['path']}")
            for attempt in range(retries + 1):
                try:
                    kwargs = dict(repo_id=remote["repo_id"], repo_type=remote.get("repo_type", "model"),
                                  revision=remote["revision"], filename=record["path"],
                                  local_dir=str(destination), local_files_only=offline)
                    kwargs["endpoint"] = endpoint or remote.get("endpoint", "https://huggingface.co")
                    downloader(**kwargs)
                    break
                except Exception as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if attempt == retries or status in {400, 401, 403, 404}:
                        raise
                    delay = min(30, 2 ** attempt)
                    print(json.dumps({"status": "retry", "file": record["path"],
                                      "attempt": attempt + 1, "wait_seconds": delay}), flush=True)
                    time.sleep(delay)
            if not check_file(path, record):
                raise ValueError(f"Downloaded asset failed SHA-256 verification: {record['path']}")
            disposition = "downloaded_and_verified"
        completed.append(record["path"])
        print(json.dumps({"status": disposition, "completed": number,
                          "total": len(records), "file": record["path"]}), flush=True)
    receipt = {"schema_version": 1, "repo_id": remote["repo_id"],
               "repo_type": remote.get("repo_type", "model"), "revision": remote["revision"],
               "files": completed, "all_sha256_verified": True}
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / "DOWNLOAD_RECEIPT.json"
    temporary = receipt_path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(receipt_path)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "download", "verify"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scene", action="append", help="Repeat to select multiple scenes; omit for all")
    parser.add_argument("--group", action="append", choices=["gaussian", "localization", "reference_images"])
    parser.add_argument("--destination", type=Path, default=ROOT / "assets")
    parser.add_argument("--repo-id")
    parser.add_argument("--revision", help="Immutable Hugging Face commit SHA, never main")
    parser.add_argument("--endpoint", help="Optional Hugging Face endpoint; authentication follows its SDK")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)
    if not 0 <= args.retries <= 10:
        parser.error("--retries must be between zero and ten")
    try:
        manifest = load_manifest(args.manifest)
        records = select_files(manifest, args.scene, args.group)
        if args.command == "list":
            counts = {}
            for record in records:
                key = record["scene"] + "/" + record["group"]
                counts.setdefault(key, {"files": 0, "bytes": 0})
                counts[key]["files"] += 1
                counts[key]["bytes"] += record["size_bytes"]
            print(json.dumps({"assets": counts, "total_files": len(records),
                              "total_bytes": sum(r["size_bytes"] for r in records),
                              "remote_revision_available": bool(manifest.get("huggingface", {}).get("revision")),
                              "private": manifest.get("huggingface", {}).get("private"),
                              "available_scenes": manifest.get("huggingface", {}).get("available_scenes", [])}, indent=2))
        elif args.command == "verify":
            failed = [r["path"] for r in records if not check_file(safe_path(args.destination, r["path"]), r)]
            print(json.dumps({"checked": len(records), "failed": failed, "ok": not failed}, indent=2))
            return int(bool(failed))
        else:
            download(manifest, records, args.destination, repo_id=args.repo_id, revision=args.revision,
                     endpoint=args.endpoint, offline=args.offline, retries=args.retries)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Checkpoint operation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
