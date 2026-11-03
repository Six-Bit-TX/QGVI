"""CPU contract tests: disjoint fitting, actual abstention and metric denominators."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import importlib.util
import json
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).with_name("calibration.py")
SPEC = importlib.util.spec_from_file_location("logs_selective_calibration", MODULE_PATH)
cal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cal
SPEC.loader.exec_module(cal)


def record(query_id, *, positive=True, error=(0.1, 0.1), pnp=True,
           count=20, fraction=0.5, coverage=0.05, scene="small", split="validation"):
    # Includes signed zero and subnormal translation to detect pose rewriting.
    pose = [[1., -0.0, 0., 5e-324], [0., 1., 0., -0.0],
            [0., 0., 1., 0.12345678901234568], [0., 0., 0., 1.]]
    return {
        "query_id": query_id, "scene": scene, "split": split,
        "method_id": cal.METHOD_ID, "feature_schema": cal.FEATURE_SCHEMA,
        "raw_c2w": pose, "initializer_status": "pnp" if pnp else "nearest_reference_fallback",
        "features": {"pnp_success": pnp, "unique_inlier_queries": count,
                     "inlier_fraction": fraction, "spatial_coverage": coverage},
        "labels": {"is_in_map": positive, "translation_error_m": error[0] if positive else None,
                   "rotation_error_deg": error[1] if positive else None},
    }


def calibration_cohort():
    return [record("correct"), record("wrong", error=(20, 20), count=40),
            record("outside", positive=False, count=3),
            record("outside_fallback", positive=False, pnp=False)]


def fit_policy(records=None):
    result = cal.fit_validation(records or calibration_cohort())
    assert result["status"] == "qualified", result
    return cal.FrozenPolicy.from_dict(result["policy"])


class CalibrationTests(unittest.TestCase):
    def test_predeclared_grid_exactly_343_and_deterministic(self):
        records = calibration_cohort()
        first = cal.fit_validation(records)
        second = cal.fit_validation(list(reversed(records)))
        self.assertEqual(first["candidate_count"], 343)
        self.assertEqual(first["policy"], second["policy"])
        self.assertEqual(first["policy"]["min_unique_inlier_queries"], 4)
        self.assertEqual(first["policy"]["min_inlier_fraction"], 0)
        self.assertEqual(first["policy"]["min_spatial_coverage"], 0)
        self.assertTrue(first["demonstrated_rejection_gain"])
        self.assertEqual(first["validation_metrics"]["counts"]["rejected_out_of_map"], 2)

    def test_fit_refuses_any_test_record_before_private_label_access(self):
        class ForbiddenLabels(dict):
            def __getitem__(self, key):
                if key == "labels":
                    raise AssertionError("Private test labels were inspected")
                return super().__getitem__(key)
        test_record = ForbiddenLabels(record("test", split="test"))
        with self.assertRaisesRegex(ValueError, "Only validation"):
            cal.fit_validation([*calibration_cohort(), test_record])

    def test_no_constraint_relaxation_for_correct_fallback(self):
        records = calibration_cohort()
        records.append(record("correct_fallback", pnp=False))
        result = cal.fit_validation(records)
        self.assertEqual(result["status"], "unqualified")
        self.assertIsNone(result["policy"])
        self.assertEqual(result["feasible_candidate_count"], 0)
        self.assertEqual(result["reason"], "no_policy_preserves_all_correct_validation_poses")

    def test_both_pose_thresholds_must_be_preserved(self):
        records = [record("strict_correct", count=40),
                   record("coarse_only_correct", error=(4, 4), count=3),
                   record("outside", positive=False, count=3)]
        result = cal.fit_validation(records)
        self.assertEqual(result["policy"]["min_unique_inlier_queries"], 0)
        for metric in result["validation_metrics"]["pose"].values():
            self.assertEqual(metric["lost_correct"], 0)
        self.assertFalse(result["demonstrated_rejection_gain"])

    def test_inclusive_threshold_edges(self):
        records = [record("at_strict", error=(1., 2.)),
                   record("at_coarse", error=(5., 5.)),
                   record("outside", positive=False, pnp=False)]
        result = cal.fit_validation(records)
        metrics = result["validation_metrics"]["pose"]
        self.assertEqual(metrics["1m_2deg"]["baseline_raw_correct"], 1)
        self.assertEqual(metrics["5m_5deg"]["baseline_raw_correct"], 2)

    def test_lack_of_negative_or_correct_pose_support_is_unqualified(self):
        for records in ([record("only_positive")],
                        [record("wrong", error=(20, 20)), record("negative", positive=False)]):
            with self.subTest(records=records):
                result = cal.fit_validation(records)
                self.assertEqual(result["status"], "unqualified")
                self.assertIsNone(result["policy"])

    def test_public_apply_never_accesses_private_labels(self):
        class LabelTrap(dict):
            def __getitem__(self, key):
                if key == "labels":
                    raise AssertionError("Inference accessed labels")
                return super().__getitem__(key)
        output = cal.apply_policy(LabelTrap(record("public")), fit_policy())
        self.assertTrue(output["accepted"])
        self.assertNotIn("labels", output)

    def test_raw_pose_and_retained_pose_exact_json_preservation(self):
        policy = fit_policy()
        original = record("retained")
        before = json.dumps(original, sort_keys=True)
        result = cal.apply_policy(original, policy)
        raw = json.dumps(original["raw_c2w"])
        self.assertEqual(json.dumps(result["raw_c2w"]), raw)
        self.assertEqual(json.dumps(result["accepted_c2w"]), raw)
        self.assertEqual(json.dumps(original, sort_keys=True), before)
        result["accepted_c2w"][0][3] = 5
        self.assertEqual(json.dumps(original, sort_keys=True), before)
        self.assertEqual(json.dumps(result["raw_c2w"]), raw)

    def test_rejection_is_real_abstention_and_retains_raw_fallback(self):
        policy = fit_policy()
        for value in (record("low", count=3), record("fallback", pnp=False, count=999)):
            output = cal.apply_policy(value, policy)
            self.assertFalse(output["accepted"])
            self.assertTrue(output["abstained"])
            self.assertIsNone(output["accepted_c2w"])
            self.assertEqual(output["raw_c2w"], value["raw_c2w"])

    def test_frozen_policy_round_trip_and_fitting_split_binding(self):
        policy = fit_policy()
        encoded = policy.to_dict()
        self.assertEqual(cal.FrozenPolicy.from_dict(json.loads(json.dumps(encoded))), policy)
        with self.assertRaises(FrozenInstanceError):
            policy.min_unique_inlier_queries = 0
        encoded["min_unique_inlier_queries"] = 0
        with self.assertRaisesRegex(ValueError, "checksum"):
            cal.FrozenPolicy.from_dict(encoded)
        clean = policy.to_dict()
        clean.pop("sha256")
        clean["fitted_split"] = "test"
        clean["sha256"] = cal._digest(clean)
        with self.assertRaisesRegex(ValueError, "split or grid mismatch"):
            cal.FrozenPolicy.from_dict(clean)

    def test_method_and_feature_schema_bound_for_apply_and_fit(self):
        for key in ("method_id", "feature_schema"):
            value = record("changed")
            value[key] = "a_different_recipe"
            with self.assertRaisesRegex(ValueError, "different method"):
                cal.apply_policy(value, fit_policy())
            with self.assertRaisesRegex(ValueError, "different method"):
                cal.fit_validation([value])

    def test_expected_scenes_and_unique_ids_required(self):
        with self.assertRaisesRegex(ValueError, "scene set"):
            cal.fit_validation(calibration_cohort(), expected_scenes=["small", "road"])
        records = calibration_cohort()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            cal.fit_validation(records + [deepcopy(records[0])])

    def test_validation_must_not_omit_scene_name(self):
        records = calibration_cohort()
        records[0]["scene"] = ""
        with self.assertRaisesRegex(ValueError, "scene must"):
            cal.fit_validation(records)

    def test_test_metrics_include_negative_accepts_and_rejected_corrects(self):
        policy = fit_policy()
        records = [record("correct", split="test"),
                   record("wrong", split="test", error=(20, 20)),
                   record("lost_correct", split="test", count=3),
                   record("accepted_outside", split="test", positive=False, count=80),
                   record("rejected_outside", split="test", positive=False, pnp=False)]
        snapshot = policy.to_dict()
        report = cal.evaluate_test(records, policy, expected_scenes=["small"])
        metrics = report["metrics"]
        self.assertEqual(metrics["classification_false_positive_rate"]["rate"], 1/2)
        self.assertEqual(metrics["classification_false_negative_rate"]["rate"], 1/3)
        self.assertEqual(metrics["classification_precision"]["rate"], 2/3)
        pose = metrics["pose"]["1m_2deg"]
        self.assertEqual(pose["accepted_pose_precision"]["rate"], 1/3)
        self.assertEqual(pose["pose_success_all_in_map"]["rate"], 1/3)
        self.assertEqual(pose["correct_pose_retention"]["rate"], 1/2)
        self.assertEqual(pose["wrong_accepted_in_map"], 1)
        self.assertEqual(pose["lost_correct"], 1)
        self.assertFalse(report["zero_correct_pose_loss_on_this_test_cohort"])
        self.assertFalse(report["policy_refitted"])
        self.assertEqual(policy.to_dict(), snapshot)
        self.assertEqual(report["by_scene"]["small"], metrics)

    def test_test_cannot_fit_or_take_unfrozen_threshold_dict(self):
        with self.assertRaisesRegex(TypeError, "already frozen"):
            cal.evaluate_test([record("test", split="test")], {"count": 1})
        with self.assertRaisesRegex(ValueError, "Only test"):
            cal.evaluate_test(calibration_cohort(), fit_policy())
        with self.assertRaisesRegex(TypeError, "frozen validation"):
            cal.apply_policy(record("test"), {"count": 1})

    def test_no_pose_and_zero_denominator_are_not_perfect_precision(self):
        value = record("missing", pnp=False, split="test", error=(None, None))
        value["raw_c2w"] = None
        report = cal.evaluate_test([value], fit_policy())
        self.assertIsNone(report["metrics"]["pose"]["1m_2deg"]["accepted_pose_precision"]["rate"])
        self.assertIsNone(report["metrics"]["classification_false_positive_rate"]["rate"])
        self.assertEqual(report["metrics"]["classification_false_negative_rate"]["rate"], 1)
        self.assertIsNone(report["decisions"][0]["raw_c2w"])
        self.assertIsNone(report["decisions"][0]["accepted_c2w"])

    def test_malformed_features_and_false_success_fail_closed(self):
        for key, replacement in (("unique_inlier_queries", True),
                                 ("inlier_fraction", float("nan")),
                                 ("spatial_coverage", -0.1),
                                 ("pnp_success", "true")):
            value = record("invalid")
            value["features"][key] = replacement
            with self.assertRaises(ValueError):
                cal.apply_policy(value, fit_policy())
        value = record("false_success")
        value["initializer_status"] = "fallback"
        with self.assertRaisesRegex(ValueError, "pnp status"):
            cal.apply_policy(value, fit_policy())

    def test_unscored_existing_positive_pose_cannot_evade_preservation_constraint(self):
        records = calibration_cohort()
        records.append(record("missing_errors", error=(None, None)))
        with self.assertRaisesRegex(ValueError, "complete error labels"):
            cal.fit_validation(records)

    def test_wilson_intervals_show_finite_sample_uncertainty(self):
        zero = cal.proportion(0, 10)
        self.assertEqual(zero["rate"], 0)
        self.assertAlmostEqual(zero["wilson_95"][1], 0.2775327998628892)
        perfect = cal.proportion(10, 10)
        self.assertLess(perfect["wilson_95"][0], 1)
        self.assertIsNone(cal.proportion(0, 0)["wilson_95"])
        with self.assertRaises(ValueError):
            cal.proportion(2, 1)


if __name__ == "__main__":
    unittest.main()
