#!/usr/bin/env python3
"""Verify exact source bytes and run isolated, offline original-code test suites."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "research/original"
SUITES = [
    ("selective_rgb", "experiments/cvpr2027_rgb/logs_selective_v2",
     ["test_confidence.py", "test_calibration.py", "test_initializer.py", "test_score_policy.py"]),
    ("agent_tool_adapters", "experiments/qgvi_a_full_eval_20261029_v1",
     ["test_retrieval_adapter.py", "test_source_geometry.py", "test_rgb_cache_import.py"]),
    ("gaussian_renderer_contract", "qgvi",
     ["tests/test_multiscene_renderer.py", "tests/test_multiscene_empty_render.py"]),
    ("caption_tools", "qgvi-rag",
     ["tests/test_pipeline.py", "tests/test_retrieval.py", "tests/test_vision.py",
      "tests/test_credentials.py", "tests/test_caption_scheduling.py"]),
    ("matched_baseline_policy", "experiments/qgvi_a_target_baselines_20261102_v1",
     ["test_policies.py"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", action="store_true", help="Run CPU fixture tests; no GPU or API work")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("qgvi_source_export", ROOT / "scripts/export_original.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    verified = module.verify(ROOT)
    for path in ORIGINAL.rglob("*.py"):
        ast.parse(path.read_bytes(), filename=str(path.relative_to(ORIGINAL)))
    print(json.dumps({"source_integrity": verified, "syntax": "passed"}))
    if not args.tests:
        return
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    optional_skips = []
    for name, directory, files in SUITES:
        work = ORIGINAL / directory
        files = list(files)
        if name == "gaussian_renderer_contract" and importlib.util.find_spec("torch") is None:
            files.remove("tests/test_multiscene_empty_render.py")
            optional_skips.append({"suite": name,
                                   "file": "tests/test_multiscene_empty_render.py",
                                   "reason": "Optional PyTorch installation is absent; original tensor-render tests are not run"})
            print(json.dumps({"optional_test_skip": optional_skips[-1]}), flush=True)
            # Other original renderer tests use pytest.importorskip themselves;
            # pytest reports those cases as skipped, not passed.
        environment["PYTHONPATH"] = str(work)
        print(f"Testing {name}", flush=True)
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files],
                       cwd=work, env=environment, check=True)
    print("Testing public_command_bridge", flush=True)
    environment["PYTHONPATH"] = str(ROOT)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "research/test_agent_bridge.py"], cwd=ROOT, env=environment, check=True)
    print(json.dumps({"status": "passed_available_tests" if optional_skips else "passed",
                      "optional_suite_skips": optional_skips,
                      "live_model_calls": 0, "gpu_jobs": 0,
                      "source_integrity": module.verify(ROOT)}))


if __name__ == "__main__":
    main()
