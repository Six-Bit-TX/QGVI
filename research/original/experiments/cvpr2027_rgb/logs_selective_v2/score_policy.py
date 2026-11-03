#!/usr/bin/env python3
"""CPU-only BlockA/SF calibration followed by frozen nine-scene test evaluation.

No private file is opened before complete public membership/hash admission is
persisted. Original private containers hold both splits; only the selected split
is joined, scored or supplied to calibration. Validation opens private containers
for BlockA and SF only. Test policies are never refitted.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

import calibration as cal
import catalog

HERE = Path(__file__).resolve().parent
EXPECTED_COUNT = {"validation": 490, "test": 3678}
REQUIRED_SOURCES = tuple(HERE / p for p in (
    "calibration.py", "test_calibration.py", "confidence.py", "score_policy.py", "test_score_policy.py"))


def cohort_scenes(split):
    if split not in EXPECTED_COUNT:
        raise ValueError("Unknown scoring split")
    return catalog.VALIDATION_SCENES if split == "validation" else catalog.SCENES


def calibration_scope():
    """Protocol scope, copied into admission, policy provenance and both reports."""
    return dict(validation_scenes=list(catalog.VALIDATION_SCENES),
                validation_query_count=EXPECTED_COUNT["validation"],
                test_scenes=list(catalog.SCENES), test_query_count=EXPECTED_COUNT["test"],
                fitting="one_global_policy_from_BlockA_and_SF_validation_only",
                other_scene_validation_labels_used=False, test_refitting=False)


def write_new(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def preflight(confidence_path, split):
    """All operations in this function are public-only, including hashing."""
    confidence_path = Path(confidence_path).resolve()
    scenes = cohort_scenes(split)
    receipt_path = confidence_path.with_name("CONFIDENCE_RECEIPT.json")
    receipt = catalog.read(receipt_path)
    if (receipt.get("status") != "complete" or receipt.get("split") != split
            or receipt.get("query_count") != EXPECTED_COUNT[split]
            or receipt.get("scene_names") != list(scenes)
            or receipt.get("confidence") != {"path": str(confidence_path), "sha256": catalog.sha(confidence_path)}):
        raise ValueError("Confidence receipt is incomplete or belongs to a different cohort")
    sources = receipt["source_sha256"]
    if any(str(p) not in sources for p in REQUIRED_SOURCES):
        raise ValueError("Missing frozen calibration/scoring source binding")
    for path, digest in sources.items():
        if "ground_truth_private" in str(path):
            raise ValueError("Private files are forbidden in public source bindings")
        if catalog.sha(path) != digest:
            raise ValueError("Frozen source changed: " + path)
    records = catalog.read(confidence_path)
    if not isinstance(records, list) or len(records) != EXPECTED_COUNT[split]:
        raise ValueError("Confidence cohort is not complete")
    identities = []
    for record in records:
        cal._validate_public(record)
        if (record.get("split") != split or record.get("scene") not in scenes
                or any(key in record for key in ("labels", "is_in_map", "translation_error_m", "rotation_error_deg"))
                or record.get("pose_modified") is not False or record.get("pnp_calls") != 0
                or record.get("photometric_refinement") is not False
                or record.get("reprojection_threshold_px") != 12.0):
            raise ValueError("Confidence record violates public fixed-pose extraction contract")
        identities.append((record["scene"], record["query_id"]))
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate confidence query")
    public, protocols, private_paths, scales = {}, {}, {}, {}
    expected_protocol_paths = {str(catalog.canonical_protocol(scene)) for scene in catalog.SCENES}
    if set(receipt["protocol_sha256"]) != expected_protocol_paths:
        raise ValueError("Receipt does not bind exactly nine historical protocols")
    for scene in catalog.SCENES:
        path = catalog.canonical_protocol(scene)
        if catalog.sha(path) != receipt["protocol_sha256"][str(path)]:
            raise ValueError("Historical protocol changed")
        # Every source protocol stays bound, while only selected scenes supply
        # public membership and private paths for this operation.
        if scene not in scenes:
            continue
        protocol = catalog.read(path)
        source = protocol["public_query_source"]
        if protocol["scene"] != scene or catalog.sha(source["path"]) != source["sha256"]:
            raise ValueError("Frozen public manifest changed")
        public[scene] = catalog.public_queries(source["path"], split, verify_pixels=False)
        expected = {(scene, row["id"]) for row in public[scene]}
        actual = {identity for identity in identities if identity[0] == scene}
        if expected != actual:
            raise ValueError("Confidence/public query membership mismatch for " + scene)
        scale = protocol["meters_per_model_unit"]
        if not cal._finite_number(scale) or scale <= 0:
            raise ValueError("Invalid historical metre scale")
        scales[scene] = {"meters_per_model_unit": scale,
                         "status": protocol.get("scale_status", "source_verified")}
        protocols[scene] = {"path": str(path), "sha256": catalog.sha(path), "public_source": source}
        private_paths[scene] = Path(source["path"]).with_name("ground_truth_private.json")
    audit = dict(status="passed", split=split, query_count=len(records), scene_names=list(scenes),
                 confidence=receipt["confidence"], receipt_sha256=catalog.sha(receipt_path),
                 source_sha256=sources, protocols=protocols, scales=scales, private_truth_read=False,
                 complete_public_membership_verified=True, calibration_scope=calibration_scope())
    return records, public, private_paths, scales, audit


def proper_pose_error(raw, truth, scale):
    """Same projected-SO(3)/atan2 diagnostic as the verified component scorer."""
    if not cal._valid_pose(raw) or not cal._valid_pose(truth):
        raise ValueError("Pose error requires finite 4x4 poses")
    a, b = np.asarray(raw, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    def so3(rotation):
        u, _, vt = np.linalg.svd(rotation)
        return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    relative = so3(a[:3, :3]).T @ so3(b[:3, :3])
    skew = np.array([relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0],
                     relative[1, 0] - relative[0, 1]])
    angle = math.degrees(math.atan2(.5 * np.linalg.norm(skew), np.clip((np.trace(relative)-1)/2, -1, 1)))
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * scale), float(angle)


def run(confidence_path, split, output, policy_path=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve existing scoring artifacts")
    if (split == "test") != (policy_path is not None):
        raise ValueError("Test requires --policy; validation forbids an input policy")
    policy = None
    if policy_path:
        policy_path = Path(policy_path)
        saved = catalog.read(policy_path)
        policy = cal.FrozenPolicy.from_dict(saved)
        report_path = policy_path.with_name("REPORT.json")
        validation = catalog.read(report_path)
        completed = catalog.read(policy_path.with_name("COMPLETED.json"))
        validation_audit = validation.get("pre_private_audit", {})
        if (policy.validation_scenes != tuple(sorted(catalog.VALIDATION_SCENES))
                or policy.validation_record_count != EXPECTED_COUNT["validation"]
                or validation.get("scoring_split") != "validation" or validation.get("status") != "qualified"
                or validation.get("calibration_scope") != calibration_scope()
                or validation_audit.get("calibration_scope") != calibration_scope()
                or validation_audit.get("split") != "validation"
                or validation_audit.get("query_count") != EXPECTED_COUNT["validation"]
                or validation_audit.get("scene_names") != list(catalog.VALIDATION_SCENES)
                or validation_audit.get("complete_public_membership_verified") is not True
                or set(validation.get("scenes", {})) != set(catalog.VALIDATION_SCENES)
                or sum(validation.get("scenes", {}).values()) != EXPECTED_COUNT["validation"]
                or validation.get("validation_cohort_sha256") != policy.validation_cohort_sha256
                or validation.get("policy") != saved or completed.get("status") != "complete"
                or completed.get("calibration_scope") != calibration_scope()
                or completed.get("report_sha256") != catalog.sha(report_path)):
            raise ValueError("Policy lacks completed BlockA/SF-only validation provenance")
    records, public, private_paths, scales, audit = preflight(confidence_path, split)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "PRE_PRIVATE_AUDIT.json", audit)
    # First private access. Containers combine splits; do not score other-split rows.
    truths, truth_hashes = {}, {}
    for scene, path in private_paths.items():
        payload = catalog.read(path)
        rows = [r for r in payload["frames"] if r["split"] == split]
        truths[scene] = {r["id"]: r for r in rows}
        if len(truths[scene]) != len(rows) or set(truths[scene]) != {q["id"] for q in public[scene]}:
            raise ValueError("Private/public selected-split membership mismatch")
        truth_hashes[scene] = catalog.sha(path)
    labeled = deepcopy(records)
    for record in labeled:
        target = truths[record["scene"]][record["query_id"]]
        positive = target["is_in_map"]
        if type(positive) is not bool:
            raise ValueError("Private membership must be Boolean")
        errors = (None, None)
        if positive and record["raw_c2w"] is not None:
            errors = proper_pose_error(record["raw_c2w"], target["c2w"],
                                       scales[record["scene"]]["meters_per_model_unit"])
        record["labels"] = dict(is_in_map=positive, translation_error_m=errors[0], rotation_error_deg=errors[1])
    report = (cal.fit_validation(labeled, expected_scenes=catalog.VALIDATION_SCENES) if split == "validation"
              else cal.evaluate_test(labeled, policy, expected_scenes=catalog.SCENES))
    report.update(completed_local=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  pre_private_audit=audit, private_container_sha256=truth_hashes,
                  private_container_scope="combined validation/test containers; only selected-split labels scored or fitted",
                  scoring_split=split, source_scale_status=scales, calibration_scope=calibration_scope(),
                  rotation_metric="proper SO(3) projection then atan2 angle in degrees",
                  cohort_status="historically_exposed_development_cohort",
                  production_default_promoted=False)
    write_new(output / "REPORT.json", report)
    if split == "validation" and report["policy"] is not None:
        write_new(output / "POLICY.json", report["policy"])
    write_new(output / "COMPLETED.json", dict(status="complete", scoring_split=split,
              report_sha256=catalog.sha(output / "REPORT.json"), private_truth_read=True,
              calibration_scope=calibration_scope(), production_default_promoted=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=tuple(EXPECTED_COUNT))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    run(args.confidence, args.split, args.output, args.policy)
