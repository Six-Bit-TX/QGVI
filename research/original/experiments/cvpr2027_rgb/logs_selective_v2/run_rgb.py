#!/usr/bin/env python3
"""Default RGB entrypoint: LoGS geometric initialization and pose-preserving gate.

Public confidence replay is CPU-only. GPU initialization respects the existing
AlloMap queue and holds both project GPU locks for the entire child execution.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from catalog import HERE, ROOT, canonical_protocol, read, sha, write_new

DEFAULT = HERE / "DEFAULT_METHOD.json"
SHARED_LOCK = ROOT / "experiments/cityloc_qgvi_v2/resources/gpu0.lock"
ALLOMAP_ROOT = Path("/home/xiongti1/slam/AlloMap/artifacts/user_resume_20261016_v1/deferred_validation_queue_v1/v8_adoption_20261018")
ALLOMAP_LOCK = Path("/home/xiongti1/slam/AlloMap/artifacts/cse_scaling_campaign_gpu.lock")
ALLOMAP_PLAN_SHA = "b6142fb65f0b1e94c1aeabaec14f14a94ec16f21cf47a2acd14e35fd24b4fd2f"


class AwaitingAdmission(RuntimeError):
    pass


def default_method():
    config = read(DEFAULT)
    if config["method_id"] != "logs_no_photo_v1" or config["photometric_refinement"] is not False:
        raise ValueError("Default recipe changed; create a new method version")
    return config


def check_queue_dependency(root=ALLOMAP_ROOT, expected_sha=ALLOMAP_PLAN_SHA):
    """Require the previously promised AlloMap work to finish, never overtake it."""
    root = Path(root)
    plan_path = root / "PLAN.json"
    if sha(plan_path) != expected_sha:
        raise ValueError("Prior AlloMap queue contract changed")
    complete_path = root / "run/COMPLETED.json"
    if not complete_path.exists():
        state = read(root / "run/STATUS.json")
        raise AwaitingAdmission("AlloMap prior reservation: " + state["status"])
    complete = read(complete_path)
    if (complete["status"] != "DEFERRED_VALIDATION_QUEUE_COMPLETE"
            or complete["plan"] != {"path": str(plan_path), "sha256": expected_sha}):
        raise ValueError("AlloMap completion does not bind the promised queue")
    expected_jobs = {j["id"]: j for j in read(plan_path)["jobs"] if j["status"] == "READY"}
    actual_jobs = complete["completed_jobs"]
    if (len(actual_jobs) != len(expected_jobs)
            or {j["id"] for j in actual_jobs} != set(expected_jobs)):
        raise ValueError("AlloMap completion omits or duplicates promised jobs")
    for job in complete["completed_jobs"]:
        receipt = job["terminal_receipt"]
        if (Path(receipt["path"]) != Path(expected_jobs[job["id"]]["output_root"]) / "deferred_terminal.json"
                or sha(receipt["path"]) != receipt["sha256"]):
            raise ValueError("AlloMap terminal receipt changed")
        terminal = read(receipt["path"])
        if (terminal["status"] != "DEFERRED_JOB_VALIDATED" or terminal["job_id"] != job["id"]
                or terminal["exit_code"] != 0):
            raise ValueError("AlloMap job was not validated")
    return {"path": str(complete_path), "sha256": sha(complete_path)}


@contextmanager
def gpu_admission():
    check_queue_dependency()
    handles = []
    try:
        for path in (SHARED_LOCK, ALLOMAP_LOCK):
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a+")
            handles.append(stream)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AwaitingAdmission("GPU project lease occupied") from exc
        receipt = check_queue_dependency()  # Recheck under both locks.
        yield receipt
    finally:
        for stream in reversed(handles):
            fcntl.flock(stream, fcntl.LOCK_UN)
            stream.close()


def initialize(plan, *, wait_for_gpu=False):
    default_method()
    # Delayed import keeps --help, dependency waiting and confidence GPU-free.
    while True:
        try:
            with gpu_admission():
                from run_initializer import run
                return run(plan)
        except AwaitingAdmission as exc:
            if not wait_for_gpu:
                raise
            print(json.dumps(dict(event="waiting_for_gpu_admission", reason=str(exc))), flush=True)
            time.sleep(15)


def _empty_correspondences():
    import numpy as np
    return dict(keypoints_colmap=np.empty((0, 2)), query_indices=np.empty(0, dtype=int),
                point3d_ids=np.empty(0, dtype=int), points3d=np.empty((0, 3)),
                reference_support=[], reference_names=[], raw_match_observation_count=0)


def replay_confidence(*, plan_path=None, legacy_scene=None):
    """Exact fixed-pose replay from verified caches, with no labels or PnP."""
    from catalog import public_queries, reference_assets, verify_plan
    from confidence import assemble_correspondences, extract_confidence
    if (plan_path is None) == (legacy_scene is None):
        raise ValueError("Specify one new plan or one original scene")
    if plan_path is not None:
        p = read(plan_path)
        verify_plan(p)
        assets, queries, scene = p["reference_assets"], p["queries"], p["scene"]
        directory = Path(p["output_dir"])
        summary = read(directory / "summary.json")
        if summary["plan_sha256"] != sha(plan_path) or summary["status"] != "complete":
            raise ValueError("Initializer did not complete this exact plan")
    else:
        scene = legacy_scene
        legacy = canonical_protocol(scene)
        p = read(legacy)
        assets = reference_assets(scene)
        queries = public_queries(ROOT / "qgvi/.staging/horizongs_eval_v1" / scene / "queries.json", "test")
        directory = legacy.parent / "initializer"
        summary = read(directory / "summary.json")
        if summary["status"] != "complete":
            raise ValueError("Legacy initializer incomplete")
    # These histories saved feature/prediction hashes; do not trust a mere flag.
    bound = summary.get("artifacts_sha256", {})
    if legacy_scene is not None:
        if summary["protocol_sha256"] != sha(canonical_protocol(scene)):
            raise ValueError("Legacy protocol changed")
        bound = {str(directory / "predictions.json"): summary["predictions_sha256"],
                 str(directory / "query_superpoint.h5"): summary["query_features_sha256"]}
    if not bound:
        raise ValueError("Initializer must bind its saved outputs")
    for path, expected in bound.items():
        if sha(path) != expected:
            raise ValueError("Initializer cache changed: " + path)
    predictions_path = directory / "predictions.json"
    if str(predictions_path) not in bound:
        raise ValueError("Predictions absent from producer checksum receipt")
    predictions = read(predictions_path)
    by_id = {row["query_id"]: row for row in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {q["id"] for q in queries}:
        raise ValueError("Prediction/query membership differs")
    sys.path.insert(0, assets["upstream"])
    import pycolmap
    native_rec = pycolmap.Reconstruction(assets["sfm_dir"])
    # The pycolmap properties materialize Python maps. Retain those maps once
    # for the batch; all coordinates and original hloc ordering are unchanged.
    rec = SimpleNamespace(images=native_rec.images, points3D=native_rec.points3D)
    ids = {image.name: ident for ident, image in rec.images.items()}
    results, started = [], time.time()
    for q in queries:
        pred = by_id[q["id"]]
        names = pred["retrieved_training_names"]
        corr = (assemble_correspondences(rec, q["id"] + ".png", names,
                                        directory / "query_superpoint.h5", directory / "matches.h5", name_to_id=ids)
                if names else _empty_correspondences())
        if corr["raw_match_observation_count"] != pred["num_matches"]:
            raise ValueError("Replayed original correspondence count differs")
        row = extract_confidence(pred, {**q, "scene": scene}, corr)
        if row["raw_c2w"] != pred.get("raw_c2w", pred.get("c2w")):
            raise ValueError("Confidence extraction modified the raw pose")
        results.append(row)
        if len(results) % 50 == 0 or len(results) == len(queries):
            print(json.dumps(dict(event="confidence_progress", scene=scene,
                                  completed=len(results), total=len(queries), elapsed_s=time.time()-started)), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show-default")
    init = sub.add_parser("initialize")
    init.add_argument("--plan", type=Path, required=True)
    init.add_argument("--wait-for-gpu", action="store_true")
    conf = sub.add_parser("confidence")
    choice = conf.add_mutually_exclusive_group(required=True)
    choice.add_argument("--plan", type=Path)
    choice.add_argument("--legacy-scene")
    conf.add_argument("--output", type=Path, required=True)
    gate = sub.add_parser("apply")
    gate.add_argument("--confidence", type=Path, required=True)
    gate.add_argument("--policy", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    default_method()
    if args.command == "show-default":
        print(json.dumps(default_method(), indent=2))
    elif args.command == "initialize":
        print(json.dumps(initialize(args.plan, wait_for_gpu=args.wait_for_gpu), indent=2))
    elif args.command == "confidence":
        if args.output.exists():
            raise FileExistsError(args.output)
        write_new(args.output, replay_confidence(plan_path=args.plan, legacy_scene=args.legacy_scene))
    else:
        from calibration import FrozenPolicy, apply_policy
        policy = FrozenPolicy.from_dict(read(args.policy))
        write_new(args.output, [apply_policy(r, policy) for r in read(args.confidence)])


if __name__ == "__main__":
    main()
