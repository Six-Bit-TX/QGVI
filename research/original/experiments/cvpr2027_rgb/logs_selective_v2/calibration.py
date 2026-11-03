"""Acceptance-only LoGS calibration; never changes a candidate camera pose.

This module does no IO and never loads ground truth. The caller supplies labels
only after public confidence extraction is complete. Fitting accepts validation
records exclusively; held-out evaluation accepts an already frozen policy.
Wilson intervals describe finite cohorts, not a guarantee after model selection.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
import math
from typing import Any, Iterable, Mapping, Sequence


METHOD_ID = "logs_no_photo_v1"
FEATURE_SCHEMA = "logs_confidence_v1"
POLICY_SCHEMA = "logs_acceptance_policy_v1"
GRID = {
    "min_unique_inlier_queries": (0, 4, 8, 12, 20, 40, 80),
    "min_inlier_fraction": (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7),
    "min_spatial_coverage": (0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
}
POSE_THRESHOLDS = {"1m_2deg": (1.0, 2.0), "5m_5deg": (5.0, 5.0)}
SELECTION_RULE = (
    "preserve all raw-correct validation poses at both inclusive thresholds; "
    "maximize out-of-map rejections; maximize retained in-map queries; "
    "minimize active gates; minimize threshold tuple in declared grid order"
)


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


GRID_SHA256 = _digest(GRID)


@dataclass(frozen=True)
class FrozenPolicy:
    """An immutable validation-fitted threshold policy with recipe binding."""

    min_unique_inlier_queries: int
    min_inlier_fraction: float
    min_spatial_coverage: float
    validation_cohort_sha256: str
    validation_record_count: int
    validation_scenes: tuple[str, ...]
    method_id: str = METHOD_ID
    feature_schema: str = FEATURE_SCHEMA
    schema: str = POLICY_SCHEMA
    fitted_split: str = "validation"
    grid_sha256: str = GRID_SHA256
    require_pnp: bool = True

    def __post_init__(self) -> None:
        if (self.method_id != METHOD_ID or self.feature_schema != FEATURE_SCHEMA
                or self.schema != POLICY_SCHEMA or self.fitted_split != "validation"
                or self.grid_sha256 != GRID_SHA256 or self.require_pnp is not True):
            raise ValueError("Policy recipe, feature schema, split or grid mismatch")
        for name, values in GRID.items():
            threshold = getattr(self, name)
            if isinstance(threshold, bool) or threshold not in values:
                raise ValueError(f"Threshold {name} is outside the predeclared grid")
        if (len(self.validation_cohort_sha256) != 64
                or any(ch not in "0123456789abcdef" for ch in self.validation_cohort_sha256)
                or not isinstance(self.validation_record_count, int)
                or isinstance(self.validation_record_count, bool)
                or self.validation_record_count < 1
                or not isinstance(self.validation_scenes, tuple)
                or not self.validation_scenes
                or tuple(sorted(set(self.validation_scenes))) != self.validation_scenes):
            raise ValueError("Missing or malformed validation provenance")

    def to_dict(self) -> dict[str, Any]:
        payload = {name: getattr(self, name) for name in self.__dataclass_fields__}
        payload["validation_scenes"] = list(self.validation_scenes)
        return {**payload, "sha256": _digest(payload)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenPolicy":
        payload = dict(value)
        supplied_digest = payload.pop("sha256", None)
        if supplied_digest != _digest(payload):
            raise ValueError("Policy checksum mismatch")
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("Policy fields differ from the frozen schema")
        payload["validation_scenes"] = tuple(payload["validation_scenes"])
        return cls(**payload)


def _finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _valid_pose(pose: Any) -> bool:
    return (isinstance(pose, (list, tuple)) and len(pose) == 4
            and all(isinstance(row, (list, tuple)) and len(row) == 4
                    and all(_finite_number(value) for value in row) for row in pose))


def _validate_public(record: Mapping[str, Any]) -> None:
    if record.get("method_id") != METHOD_ID or record.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("Confidence record belongs to a different method or feature schema")
    if not isinstance(record.get("query_id"), str) or not record["query_id"]:
        raise ValueError("query_id must be a nonempty string")
    if not isinstance(record.get("scene"), str) or not record["scene"]:
        raise ValueError("scene must be a nonempty string")
    features = record["features"]
    if not isinstance(features.get("pnp_success"), bool):
        raise ValueError("pnp_success must be Boolean")
    count = features.get("unique_inlier_queries")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("unique_inlier_queries must be a nonnegative integer")
    for name in ("inlier_fraction", "spatial_coverage"):
        if not _finite_number(features.get(name)) or not 0 <= features[name] <= 1:
            raise ValueError(f"{name} must be finite and lie in [0, 1]")
    if features["pnp_success"] and (record.get("initializer_status") != "pnp"
                                    or not _valid_pose(record.get("raw_c2w"))):
        raise ValueError("Successful PnP must have pnp status and a finite 4x4 raw pose")


def _accept(record: Mapping[str, Any], thresholds: Sequence[float]) -> bool:
    f = record["features"]
    return bool(f["pnp_success"] and f["unique_inlier_queries"] >= thresholds[0]
                and f["inlier_fraction"] >= thresholds[1]
                and f["spatial_coverage"] >= thresholds[2])


def apply_policy(record: Mapping[str, Any], policy: FrozenPolicy) -> dict[str, Any]:
    """Return a public-only decision. Abstention is None, never a fallback pose.

    raw_c2w is copied without arithmetic, rounding, normalization or conversion;
    accepted_c2w is the exact same value for acceptance. Private labels and their
    derived errors are deliberately absent from this inference output.
    """
    if not isinstance(policy, FrozenPolicy):
        raise TypeError("Use a frozen validation policy, not an unfitted threshold mapping")
    _validate_public(record)
    thresholds = (policy.min_unique_inlier_queries, policy.min_inlier_fraction,
                  policy.min_spatial_coverage)
    accepted = _accept(record, thresholds)
    reasons = []
    if not record["features"]["pnp_success"]:
        reasons.append("pnp_not_verified")
    for key, minimum in zip(("unique_inlier_queries", "inlier_fraction", "spatial_coverage"), thresholds):
        if record["features"][key] < minimum:
            reasons.append(f"below_min_{key}")
    return {
        "query_id": record["query_id"], "scene": record["scene"],
        "split": record.get("split"), "method_id": METHOD_ID,
        "feature_schema": FEATURE_SCHEMA,
        "initializer_status": record.get("initializer_status"),
        "raw_c2w": deepcopy(record.get("raw_c2w")),
        "accepted_c2w": deepcopy(record.get("raw_c2w")) if accepted else None,
        "accepted": accepted, "abstained": not accepted, "reasons": reasons,
        "policy_sha256": policy.to_dict()["sha256"],
    }


def _validate_labeled_cohort(records: Sequence[Mapping[str, Any]], split: str,
                             expected_scenes: Iterable[str] | None) -> None:
    if not records:
        raise ValueError("Cohort is empty")
    # Check the whole split boundary before looking at any labels.
    if any(record.get("split") != split for record in records):
        raise ValueError(f"Only {split} records are allowed in this operation")
    identities = [(record.get("scene"), record.get("query_id")) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate scene/query identity in cohort")
    actual_scenes = {record.get("scene") for record in records}
    if expected_scenes is not None and actual_scenes != set(expected_scenes):
        raise ValueError("Cohort scene set does not match expected_scenes")
    for record in records:
        _validate_public(record)
        labels = record["labels"]
        if not isinstance(labels.get("is_in_map"), bool):
            raise ValueError("is_in_map label must be Boolean")
        if labels["is_in_map"]:
            values = (labels.get("translation_error_m"), labels.get("rotation_error_deg"))
            if values != (None, None) and not all(_finite_number(v) and v >= 0 for v in values):
                raise ValueError("Positive pose errors must be finite/nonnegative, or both None")
            if values != (None, None) and not _valid_pose(record.get("raw_c2w")):
                raise ValueError("Pose errors require a finite raw camera pose")
            if values == (None, None) and record.get("raw_c2w") is not None:
                raise ValueError("Existing in-map poses must have complete error labels")


def _correct(record: Mapping[str, Any], threshold: Sequence[float]) -> bool:
    label = record["labels"]
    return bool(label["is_in_map"] and label.get("translation_error_m") is not None
                and label.get("rotation_error_deg") is not None
                and label["translation_error_m"] <= threshold[0]
                and label["rotation_error_deg"] <= threshold[1])


def proportion(numerator: int, denominator: int) -> dict[str, Any]:
    """Wilson 95% interval (z=1.959963984540054); null for empty denominator."""
    if (not isinstance(numerator, int) or not isinstance(denominator, int)
            or isinstance(numerator, bool) or isinstance(denominator, bool)
            or denominator < 0 or not 0 <= numerator <= denominator):
        raise ValueError("Invalid binomial count")
    if not denominator:
        return {"numerator": numerator, "denominator": denominator,
                "rate": None, "wilson_95": None}
    z = 1.959963984540054
    p = numerator / denominator
    a = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / a
    half = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator ** 2)) / a
    return {"numerator": numerator, "denominator": denominator, "rate": p,
            "wilson_95": [max(0.0, center - half), min(1.0, center + half)]}


def _metrics(records: Sequence[Mapping[str, Any]], decisions: Sequence[bool]) -> dict[str, Any]:
    positives = sum(r["labels"]["is_in_map"] for r in records)
    negatives = len(records) - positives
    accepted = sum(decisions)
    accepted_positive = sum(a and r["labels"]["is_in_map"] for r, a in zip(records, decisions))
    accepted_negative = accepted - accepted_positive
    result: dict[str, Any] = {
        "counts": {"queries": len(records), "in_map": positives, "out_of_map": negatives,
                   "accepted": accepted, "accepted_in_map": accepted_positive,
                   "accepted_out_of_map": accepted_negative,
                   "rejected_in_map": positives - accepted_positive,
                   "rejected_out_of_map": negatives - accepted_negative},
        "classification_false_positive_rate": proportion(accepted_negative, negatives),
        "classification_false_negative_rate": proportion(positives - accepted_positive, positives),
        "out_of_map_rejection_rate": proportion(negatives - accepted_negative, negatives),
        "classification_precision": proportion(accepted_positive, accepted),
        "pose": {},
    }
    for name, threshold in POSE_THRESHOLDS.items():
        raw_correct = sum(_correct(r, threshold) for r in records)
        retained_correct = sum(a and _correct(r, threshold) for r, a in zip(records, decisions))
        result["pose"][name] = {
            "baseline_raw_correct": raw_correct, "retained_correct": retained_correct,
            "lost_correct": raw_correct - retained_correct,
            "wrong_accepted_in_map": accepted_positive - retained_correct,
            "baseline_raw_success_all_in_map": proportion(raw_correct, positives),
            "pose_success_all_in_map": proportion(retained_correct, positives),
            "correct_pose_retention": proportion(retained_correct, raw_correct),
            # An accepted negative is not a correct localization and stays in denominator.
            "accepted_pose_precision": proportion(retained_correct, accepted),
        }
    return result


def fit_validation(records: Sequence[Mapping[str, Any]], *,
                   expected_scenes: Iterable[str] | None = None) -> dict[str, Any]:
    """Select one global policy from 343 predeclared candidates on validation only.

    The production caller must supply expected_scenes and independently verify
    exact frozen query IDs/completeness. No scene-specific fitting, test refitting,
    GT-selected candidate poses, or silent relaxation of constraints is provided.
    """
    records = list(records)
    _validate_labeled_cohort(records, "validation", expected_scenes)
    ordered = sorted(records, key=lambda r: (r["scene"], r["query_id"]))
    # Bind only fields used for acceptance and labels used for selection, and the
    # original pose whose correctness is being preserved; unrelated metadata is ignored.
    bound_records = [{
        "query_id": r["query_id"], "scene": r["scene"], "split": r["split"],
        "method_id": r["method_id"], "feature_schema": r["feature_schema"],
        "raw_c2w": r.get("raw_c2w"),
        "features": {k: r["features"][k] for k in
                     ("pnp_success", "unique_inlier_queries", "inlier_fraction", "spatial_coverage")},
        "labels": r["labels"],
    } for r in ordered]
    cohort_digest = _digest(bound_records)
    baseline = _metrics(records, [_valid_pose(r.get("raw_c2w")) for r in records])
    common = {
        "schema": "logs_validation_calibration_v1", "method_id": METHOD_ID,
        "feature_schema": FEATURE_SCHEMA, "grid": GRID, "grid_sha256": GRID_SHA256,
        "candidate_count": math.prod(len(values) for values in GRID.values()),
        "selection_rule": SELECTION_RULE, "validation_cohort_sha256": cohort_digest,
        "scenes": dict(sorted(Counter(r["scene"] for r in records).items())),
        "baseline": baseline, "interval_caveat":
        "Wilson intervals are descriptive; validation-selected results are not held-out guarantees.",
    }
    if (not baseline["counts"]["in_map"] or not baseline["counts"]["out_of_map"]
            or any(not baseline["pose"][name]["baseline_raw_correct"] for name in POSE_THRESHOLDS)):
        return {**common, "status": "unqualified", "policy": None,
                "reason": "insufficient_positive_negative_or_correct_pose_validation_support",
                "demonstrated_rejection_gain": False}
    must_retain = [any(_correct(r, threshold) for threshold in POSE_THRESHOLDS.values()) for r in records]
    feasible = []
    for thresholds in product(*GRID.values()):
        decisions = [_accept(r, thresholds) for r in records]
        if any(correct and not accepted for correct, accepted in zip(must_retain, decisions)):
            continue
        rejected_negative = sum(not a and not r["labels"]["is_in_map"] for r, a in zip(records, decisions))
        retained_positive = sum(a and r["labels"]["is_in_map"] for r, a in zip(records, decisions))
        active_gates = sum(t > 0 for t in thresholds)
        rank = (-rejected_negative, -retained_positive, active_gates, *thresholds)
        feasible.append((rank, thresholds, decisions))
    if not feasible:
        return {**common, "status": "unqualified", "policy": None,
                "feasible_candidate_count": 0,
                "reason": "no_policy_preserves_all_correct_validation_poses",
                "demonstrated_rejection_gain": False}
    _, thresholds, decisions = min(feasible, key=lambda item: item[0])
    policy = FrozenPolicy(*thresholds, cohort_digest, len(records), tuple(sorted(common["scenes"])))
    metrics = _metrics(records, decisions)
    return {**common, "status": "qualified", "policy": policy.to_dict(),
            "feasible_candidate_count": len(feasible), "validation_metrics": metrics,
            "demonstrated_rejection_gain": (metrics["counts"]["accepted_out_of_map"]
                                            < baseline["counts"]["accepted_out_of_map"]),
            "qualification_scope": "validation correct-pose retention only; test generalization unproven"}


def evaluate_test(records: Sequence[Mapping[str, Any]], policy: FrozenPolicy, *,
                  expected_scenes: Iterable[str] | None = None) -> dict[str, Any]:
    """Score held-out records with a frozen validation policy; never select or refit."""
    if not isinstance(policy, FrozenPolicy):
        raise TypeError("evaluate_test requires an already frozen validation policy")
    records = list(records)
    _validate_labeled_cohort(records, "test", expected_scenes)
    decisions = [apply_policy(record, policy) for record in records]
    accepted = [d["accepted"] for d in decisions]
    metrics = _metrics(records, accepted)
    by_scene = {}
    for scene in sorted({r["scene"] for r in records}):
        pairs = [(r, a) for r, a in zip(records, accepted) if r["scene"] == scene]
        by_scene[scene] = _metrics([r for r, _ in pairs], [a for _, a in pairs])
    lost = {name: metrics["pose"][name]["lost_correct"] for name in POSE_THRESHOLDS}
    return {
        "schema": "logs_frozen_test_evaluation_v1", "policy": policy.to_dict(),
        "metrics": metrics, "by_scene": by_scene, "decisions": decisions,
        "zero_correct_pose_loss_on_this_test_cohort": not any(lost.values()),
        "lost_correct_test_poses": lost, "policy_refitted": False,
        "cohort_status": "historically_exposed_development_test_cohort",
        "interval_caveat": "Wilson intervals describe this finite cohort; zero observed errors does not imply zero population risk.",
    }
