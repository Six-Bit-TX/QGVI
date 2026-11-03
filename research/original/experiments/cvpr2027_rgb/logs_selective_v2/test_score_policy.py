"""Synthetic-only CPU admission and label-join tests; no actual GT is read."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score_policy as score
import calibration as cal


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def pose(x=0., angle=0.):
    angle = math.radians(angle)
    return [[math.cos(angle), -math.sin(angle), 0., x],
            [math.sin(angle), math.cos(angle), 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]


class ScorePolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.protocols = {}
        self.records = {"validation": [], "test": []}
        self.all_scene_validation = []
        for scene in score.catalog.SCENES:
            public, private = [], []
            for split in self.records:
                for positive in (True, False):
                    qid = scene + "_" + split + ("_positive" if positive else "_negative")
                    public.append(dict(id=qid, image_path=str(self.root / scene / "images" / (qid + ".png")),
                                       image_sha256="0" * 64, K=[[10, 0, 5], [0, 10, 5], [0, 0, 1]],
                                       width=10, height=10, split=split))
                    private.append(dict(id=qid, split=split, is_in_map=positive,
                                        c2w=pose() if positive else None))
                    record = dict(query_id=qid, scene=scene, split=split,
                        method_id=cal.METHOD_ID, feature_schema=cal.FEATURE_SCHEMA,
                        raw_c2w=pose(), initializer_status="pnp", pose_modified=False, pnp_calls=0,
                        photometric_refinement=False, reprojection_threshold_px=12.,
                        features=dict(pnp_success=True, unique_inlier_queries=20 if positive else 3,
                                      inlier_fraction=.5, spatial_coverage=.05))
                    if split == "validation":
                        self.all_scene_validation.append(record)
                    if scene in score.cohort_scenes(split):
                        self.records[split].append(record)
            manifest = self.root / scene / "queries.json"
            write(manifest, {"frames": public})
            write(manifest.with_name("ground_truth_private.json"), {"frames": private})
            protocol = self.root / scene / "protocol.json"
            write(protocol, dict(scene=scene, meters_per_model_unit=25.,
                                 public_query_source={"path": str(manifest), "sha256": score.catalog.sha(manifest)}))
            self.protocols[scene] = protocol
        self.addCleanup(patch.stopall)
        patch.object(score.catalog, "canonical_protocol", lambda scene: self.protocols[scene]).start()
        patch.object(score, "EXPECTED_COUNT", {split: len(rows) for split, rows in self.records.items()}).start()
        for split in self.records:
            self.save_confidence(split)

    def confidence_path(self, split):
        return self.root / "confidence" / split / "CONFIDENCE.json"

    def save_confidence(self, split):
        path = self.confidence_path(split)
        write(path, self.records[split])
        receipt = dict(status="complete", split=split, query_count=len(self.records[split]),
                       scene_names=list(score.cohort_scenes(split)),
                       confidence=dict(path=str(path), sha256=score.catalog.sha(path)),
                       source_sha256={str(p): score.catalog.sha(p) for p in score.REQUIRED_SOURCES},
                       protocol_sha256={str(p): score.catalog.sha(p) for p in self.protocols.values()})
        write(path.with_name("CONFIDENCE_RECEIPT.json"), receipt)

    def test_complete_admission_is_durable_before_private_first_open(self):
        output = self.root / "validation_report"
        original_read = score.catalog.read
        private_opens = []
        def monitored_read(path):
            if Path(path).name == "ground_truth_private.json":
                gate = json.loads((output / "PRE_PRIVATE_AUDIT.json").read_text())
                self.assertTrue(gate["complete_public_membership_verified"])
                self.assertFalse(gate["private_truth_read"])
                private_opens.append(str(path))
            return original_read(path)
        with patch.object(score.catalog, "read", monitored_read):
            result = score.run(self.confidence_path("validation"), "validation", output)
        expected_private = {str(self.root / scene / "ground_truth_private.json")
                            for scene in score.catalog.VALIDATION_SCENES}
        self.assertEqual(set(private_opens), expected_private)
        self.assertEqual(len(private_opens), 2)
        self.assertEqual(result["status"], "qualified")
        self.assertTrue((output / "POLICY.json").is_file())
        self.assertFalse(result["production_default_promoted"])
        self.assertEqual(result["validation_metrics"]["counts"]["rejected_out_of_map"], 2)
        self.assertEqual(result["calibration_scope"], score.calibration_scope())
        self.assertEqual(set(result["private_container_sha256"]), {"block_a", "sf"})

    def test_public_membership_or_receipt_failures_never_open_private(self):
        original = deepcopy(self.records["validation"])
        original_read = score.catalog.read
        for mutation in ("missing", "duplicate", "foreign_id", "foreign_scene", "foreign_split", "private_field", "recipe", "protocol", "source"):
            with self.subTest(mutation=mutation):
                self.records["validation"] = deepcopy(original)
                rows = self.records["validation"]
                if mutation == "missing": rows.pop()
                if mutation == "duplicate": rows[1] = deepcopy(rows[0])
                if mutation == "foreign_id": rows[0]["query_id"] = "foreign"
                if mutation == "foreign_scene": rows[0]["scene"] = "road"
                if mutation == "foreign_split": rows[0]["split"] = "test"
                if mutation == "private_field": rows[0]["labels"] = {"is_in_map": True}
                if mutation == "recipe": rows[0]["photometric_refinement"] = True
                self.save_confidence("validation")
                receipt_path = self.confidence_path("validation").with_name("CONFIDENCE_RECEIPT.json")
                receipt = original_read(receipt_path)
                if mutation in ("protocol", "source"):
                    key = "protocol_sha256" if mutation == "protocol" else "source_sha256"
                    receipt[key][next(iter(receipt[key]))] = "0" * 64
                    write(receipt_path, receipt)
                def forbid_private(path):
                    self.assertNotEqual(Path(path).name, "ground_truth_private.json")
                    return original_read(path)
                with patch.object(score.catalog, "read", forbid_private):
                    with self.assertRaises(ValueError):
                        score.run(self.confidence_path("validation"), "validation", self.root / mutation)

    def test_validation_fit_cannot_receive_any_test_label(self):
        original_fit = cal.fit_validation
        def fit_spy(records, **kwargs):
            self.assertEqual({r["split"] for r in records}, {"validation"})
            self.assertEqual({r["scene"] for r in records}, {"block_a", "sf"})
            self.assertEqual(kwargs["expected_scenes"], ("block_a", "sf"))
            self.assertEqual(len(records), 4)
            return original_fit(records, **kwargs)
        with patch.object(cal, "fit_validation", fit_spy):
            score.run(self.confidence_path("validation"), "validation", self.root / "validation")

    def test_full_nine_scene_validation_is_rejected_before_private_access(self):
        self.records["validation"] = deepcopy(self.all_scene_validation)
        self.save_confidence("validation")
        receipt_path = self.confidence_path("validation").with_name("CONFIDENCE_RECEIPT.json")
        receipt = score.catalog.read(receipt_path)
        receipt["scene_names"] = list(score.catalog.SCENES)
        write(receipt_path, receipt)
        original_read = score.catalog.read
        def forbid_private(path):
            self.assertNotEqual(Path(path).name, "ground_truth_private.json")
            return original_read(path)
        # Even if a future caller mistakenly permits the larger count, its
        # scene scope cannot qualify as the requested BlockA/SF validation.
        with patch.object(score, "EXPECTED_COUNT", {"validation": 18, "test": 18}), \
                patch.object(score.catalog, "read", forbid_private):
            with self.assertRaisesRegex(ValueError, "different cohort"):
                score.run(self.confidence_path("validation"), "validation", self.root / "nine")

    def test_missing_or_substituted_validation_scene_is_rejected_before_private_access(self):
        original_read = score.catalog.read
        def forbid_private(path):
            self.assertNotEqual(Path(path).name, "ground_truth_private.json")
            return original_read(path)
        for replacement in (None, "road"):
            with self.subTest(replacement=replacement):
                wanted = {"block_a"} | ({replacement} if replacement else set())
                self.records["validation"] = [deepcopy(r) for r in self.all_scene_validation if r["scene"] in wanted]
                self.save_confidence("validation")
                with patch.object(score.catalog, "read", forbid_private):
                    with self.assertRaises(ValueError):
                        score.run(self.confidence_path("validation"), "validation", self.root / ("scope_" + str(replacement)))

    def test_test_uses_frozen_policy_and_exposes_loss_without_refit(self):
        validation_dir = self.root / "validation"
        initial = score.run(self.confidence_path("validation"), "validation", validation_dir)
        frozen_bytes = (validation_dir / "POLICY.json").read_bytes()
        self.records["test"][0]["features"]["unique_inlier_queries"] = 3
        self.save_confidence("test")
        with patch.object(cal, "fit_validation", side_effect=AssertionError("Test refit forbidden")):
            result = score.run(self.confidence_path("test"), "test", self.root / "test", validation_dir / "POLICY.json")
        self.assertEqual(result["policy"], initial["policy"])
        self.assertEqual((validation_dir / "POLICY.json").read_bytes(), frozen_bytes)
        self.assertFalse(result["zero_correct_pose_loss_on_this_test_cohort"])
        self.assertEqual(result["lost_correct_test_poses"]["1m_2deg"], 1)
        self.assertFalse(result["production_default_promoted"])
        self.assertEqual(set(result["private_container_sha256"]), set(score.catalog.SCENES))
        self.assertEqual(result["pre_private_audit"]["scene_names"], list(score.catalog.SCENES))
        self.assertEqual(result["calibration_scope"]["validation_scenes"], ["block_a", "sf"])

    def test_test_requires_complete_validation_provenance(self):
        directory = self.root / "validation"
        score.run(self.confidence_path("validation"), "validation", directory)
        completed = json.loads((directory / "COMPLETED.json").read_text())
        completed["report_sha256"] = "0" * 64
        write(directory / "COMPLETED.json", completed)
        with self.assertRaisesRegex(ValueError, "validation provenance"):
            score.run(self.confidence_path("test"), "test", self.root / "test", directory / "POLICY.json")

    def test_wrong_validation_scope_never_admits_test_private_access(self):
        directory = self.root / "validation"
        score.run(self.confidence_path("validation"), "validation", directory)
        original_policy = score.catalog.read(directory / "POLICY.json")
        original_report = score.catalog.read(directory / "REPORT.json")
        original_completed = score.catalog.read(directory / "COMPLETED.json")
        original_read = score.catalog.read
        def forbid_private(path):
            self.assertNotEqual(Path(path).name, "ground_truth_private.json")
            return original_read(path)
        for mutation in ("nine_scene_policy", "foreign_scene_policy", "wrong_count", "missing_scope",
                         "report_scenes", "audit_scenes", "cohort_digest", "completed_scope"):
            with self.subTest(mutation=mutation):
                policy = cal.FrozenPolicy.from_dict(original_policy)
                report, completed = deepcopy(original_report), deepcopy(original_completed)
                if mutation == "nine_scene_policy":
                    policy = replace(policy, validation_scenes=tuple(sorted(score.catalog.SCENES)),
                                     validation_record_count=18)
                if mutation == "foreign_scene_policy":
                    policy = replace(policy, validation_scenes=("block_a", "road"))
                if mutation == "wrong_count":
                    policy = replace(policy, validation_record_count=18)
                if mutation == "missing_scope": report.pop("calibration_scope")
                if mutation == "report_scenes": report["scenes"] = {"block_a": 2, "road": 2}
                if mutation == "audit_scenes": report["pre_private_audit"]["scene_names"] = ["block_a", "road"]
                if mutation == "cohort_digest": report["validation_cohort_sha256"] = "0" * 64
                if mutation == "completed_scope": completed.pop("calibration_scope")
                report["policy"] = policy.to_dict()
                write(directory / "POLICY.json", report["policy"])
                write(directory / "REPORT.json", report)
                completed["report_sha256"] = score.catalog.sha(directory / "REPORT.json")
                write(directory / "COMPLETED.json", completed)
                with patch.object(score.catalog, "read", forbid_private):
                    with self.assertRaisesRegex(ValueError, "BlockA/SF-only validation provenance"):
                        score.run(self.confidence_path("test"), "test", self.root / mutation, directory / "POLICY.json")

    def test_output_is_exclusive_and_test_policy_argument_is_required(self):
        output = self.root / "exists"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            score.run(self.confidence_path("validation"), "validation", output)
        with self.assertRaisesRegex(ValueError, "Test requires"):
            score.run(self.confidence_path("test"), "test", self.root / "test")

    def test_duplicate_private_join_fails_after_public_admission(self):
        path = self.root / score.catalog.VALIDATION_SCENES[0] / "ground_truth_private.json"
        value = json.loads(path.read_text())
        value["frames"].append(deepcopy(value["frames"][0]))
        write(path, value)
        output = self.root / "validation"
        with self.assertRaisesRegex(ValueError, "Private/public"):
            score.run(self.confidence_path("validation"), "validation", output)
        self.assertTrue((output / "PRE_PRIVATE_AUDIT.json").is_file())
        self.assertFalse((output / "COMPLETED.json").exists())

    def test_metric_uses_model_scale_and_proper_rotation(self):
        translation, rotation = score.proper_pose_error(pose(1, 60), pose(), 25.)
        self.assertAlmostEqual(translation, 25.)
        self.assertAlmostEqual(rotation, 60.)
        almost_rotation = pose()
        almost_rotation[0][0] += 1e-7
        self.assertAlmostEqual(score.proper_pose_error(almost_rotation, pose(), 1.)[1], 0.)
        with self.assertRaisesRegex(ValueError, "finite 4x4"):
            score.proper_pose_error(None, pose(), 1.)


if __name__ == "__main__":
    unittest.main()
