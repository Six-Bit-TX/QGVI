#!/usr/bin/env python3
"""Copy the allowlisted research source without editing a single source byte.

Maintainer utility. Public users use ``--verify``; exporting again requires the
original research workspace. No checkpoints, credentials, logs, or result files
are discovered recursively or included implicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


GROUPS = {
    "agent": ("experiments/qgvi_a_full_eval_20261029_v1", [
        "controller.py", "protocol.py", "state.py", "early_exit.py", "tool_host.py",
        "io_utils.py", "providers.py", "speed_provider.py", "baselines.py",
        "retrieval_backend.py", "navigation_backend.py", "navigation_server.py",
        "source_geometry.py", "rgb_cache_import.py", "PROTOCOL.md",
        "test_retrieval_adapter.py", "test_source_geometry.py", "test_rgb_cache_import.py",
    ]),
    "provider_import_dependency": ("experiments/agentic_locator_effort_fast_20261025_v1", [
        "providers.py",
    ]),
    "optional_subscription_adapter": ("experiments/dual_model_review_20261020_v1", [
        "subscription_codex.py", "model_review.py",
    ]),
    "geometric_localizer": ("experiments/agentic_locator_comparison_20261025_v1", [
        "geometry_adapter.py",
    ]),
    "selective_rgb": ("experiments/cvpr2027_rgb/logs_selective_v2", [
        "catalog.py", "confidence.py", "calibration.py", "prepare.py",
        "run_initializer.py", "run_rgb.py", "score_policy.py",
        "test_confidence.py", "test_calibration.py", "test_initializer.py",
        "test_score_policy.py",
    ]),
    "caption_dependency": ("qgvi-rag/caption_rag", [
        "__init__.py", "pipeline.py", "retrieval.py", "vision.py", "credentials.py",
        "cli.py",
    ]),
    "caption_tests": ("qgvi-rag/tests", [
        "test_pipeline.py", "test_retrieval.py", "test_vision.py",
        "test_credentials.py", "test_caption_scheduling.py",
    ]),
    "gaussian_renderer": ("qgvi/qgvi", ["__init__.py", "multiscene_renderer.py"]),
    "gaussian_renderer_tests": ("qgvi/tests", [
        "test_multiscene_renderer.py", "test_multiscene_empty_render.py",
    ]),
    "matched_baseline_policy": ("experiments/qgvi_a_target_baselines_20261102_v1", [
        "policies.py", "test_policies.py", "PROTOCOL.md",
    ]),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selections():
    for group, (directory, names) in GROUPS.items():
        for name in names:
            yield group, Path(directory) / name


def verify(release: Path) -> dict:
    manifest = json.loads((release / "research/source_manifest.json").read_text())
    expected = {str(path) for _, path in selections()}
    if {row["source_path"] for row in manifest["files"]} != expected:
        raise ValueError("Manifest does not match the explicit source allowlist")
    for row in manifest["files"]:
        path = release / row["release_path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing file or unexpected symlink: {row['release_path']}")
        if sha(path) != row["sha256"] or path.stat().st_size != row["size_bytes"]:
            raise ValueError(f"Source bytes changed: {row['release_path']}")
    actual = {str(p.relative_to(release / "research/original"))
              for p in (release / "research/original").rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    if actual != expected:
        raise ValueError("Unexpected file in original-source snapshot")
    return {"status": "passed", "files": len(expected),
            "bytes": sum(row["size_bytes"] for row in manifest["files"]),
            "source_edits": 0}


def export(source: Path, release: Path) -> dict:
    source, release = source.resolve(strict=True), release.resolve()
    records = []
    for group, relative in selections():
        origin = source / relative
        if origin.is_symlink() or not origin.is_file():
            raise ValueError(f"Expected a regular source file: {relative}")
        before = sha(origin)
        destination = release / "research/original" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, destination)
        if sha(origin) != before or sha(destination) != before:
            raise ValueError(f"Source changed while exporting: {relative}")
        records.append({"group": group, "source_path": str(relative),
                        "release_path": str(destination.relative_to(release)),
                        "sha256": before, "size_bytes": destination.stat().st_size})
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source,
                              capture_output=True, text=True, check=True).stdout.strip()
    manifest = {
        "schema": "qgvi_original_source_snapshot_v1",
        "snapshot_date": "2026-11-03",
        "source_git_head": revision,
        "provenance": "Working-tree bytes, including uncommitted research changes; individual SHA-256 hashes are authoritative.",
        "algorithm_source_modified": False,
        "paths": "Repository-relative; no machine paths in this manifest.",
        "exclusions": ["credentials and environment files", "native session queues and transcripts",
                       "private evaluation labels", "model checkpoints and external datasets",
                       "third-party source code", "AlloMap admission and job orchestration"],
        "files": records,
    }
    target = release / "research/source_manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    return verify(release)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        help="Original research checkout; required only when exporting")
    parser.add_argument("--release-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify and args.source_root is None:
        parser.error("Export requires --source-root; use --verify for the release snapshot")
    result = verify(args.release_root) if args.verify else export(args.source_root, args.release_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
