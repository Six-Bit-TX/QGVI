"""CPU tests for pose-preserving, ambiguity-aware public confidence evidence."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

_SPEC = importlib.util.spec_from_file_location("selective_confidence", Path(__file__).with_name("confidence.py"))
confidence = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(confidence)


def fixture(xy=None, xyz=None, qi=None, pids=None, support=None):
    xy = np.asarray(xy if xy is not None else [[10., 10.], [90., 10.], [90., 90.], [10., 90.]], float).reshape(-1, 2)
    n = len(xy)
    return dict(
        keypoints_colmap=xy,
        points3d=np.asarray(xyz if xyz is not None else np.c_[xy, np.ones(n)], float).reshape(-1, 3),
        query_indices=np.asarray(qi if qi is not None else np.arange(n), np.int64),
        point3d_ids=np.asarray(pids if pids is not None else np.arange(n), np.int64),
        reference_support=support if support is not None else [(0, 1)] * n,
        reference_names=["ref0", "ref1"], raw_match_observation_count=2 * n,
    )


class ConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.query = dict(id="public", K=np.eye(3).tolist(), width=100, height=100, split="validation", scene="toy")
        self.pred = dict(query_id="public", c2w=np.eye(4).tolist(), status="pnp")

    def score(self, corr, **kwargs):
        return confidence.extract_confidence(self.pred, self.query, corr, **kwargs)

    def test_fixed_pose_support_coverage_and_json_safe(self):
        result = self.score(fixture())
        f = result["features"]
        self.assertTrue(f["pnp_success"])
        self.assertEqual(f["unique_inlier_queries"], 4)
        self.assertEqual(f["unique_inlier_landmarks"], 4)
        self.assertEqual(f["inlier_fraction"], 1)
        self.assertAlmostEqual(f["spatial_coverage"], .64)
        self.assertEqual(f["inlier_occupied_4x4_grid_fraction"], .25)
        self.assertEqual(f["median_inlier_reprojection_px"], 0)
        self.assertEqual(f["supporting_reference_count"], 2)
        self.assertEqual(f["max_reference_unique_inlier_queries"], 4)
        self.assertEqual(f["second_reference_unique_inlier_queries"], 4)
        self.assertEqual(f["multi_reference_inlier_query_fraction"], 1)
        self.assertEqual(result["method_id"], "logs_no_photo_v1")
        self.assertEqual(result["feature_schema"], "logs_confidence_v1")
        self.assertNotIn("accepted", result)
        json.dumps(result, allow_nan=False)

    def test_competing_landmarks_do_not_inflate_unique_support(self):
        c = fixture(xy=[[10, 10], [10, 10], [20, 20]], xyz=[[11, 10, 1], [10, 10, 1], [20, 20, 1]],
                    qi=[0, 0, 1], pids=[5, 6, 7], support=[(0,), (1,), (0, 1)])
        f = self.score(c)["features"]
        self.assertEqual(f["deduplicated_pair_count"], 3)
        self.assertEqual(f["unique_inlier_queries"], 2)
        self.assertEqual(f["unique_inlier_landmarks"], 2)
        self.assertEqual(f["ambiguous_query_count"], 1)
        self.assertEqual(f["inlier_fraction"], 1)
        self.assertEqual(f["median_inlier_reprojection_px"], 0)
        self.assertEqual(f["second_reference_unique_inlier_queries"], 1)
        self.assertEqual(f["multi_reference_inlier_query_fraction"], .5)
        # Permuting pairs cannot change tie-breaking or output evidence.
        order = [2, 1, 0]
        c2 = dict(c)
        for key in ("keypoints_colmap", "query_indices", "point3d_ids", "points3d"):
            c2[key] = c[key][order]
        c2["reference_support"] = [c["reference_support"][i] for i in order]
        self.assertEqual(f, self.score(c2)["features"])

    def test_duplicate_reference_observations_and_shared_landmark(self):
        c = fixture(xy=[[10, 10], [10, 10]], qi=[0, 1], pids=[5, 5], support=[(0, 0, 1), (0, 1, 1)])
        f = self.score(c)["features"]
        self.assertEqual(f["unique_inlier_queries"], 2)
        self.assertEqual(f["unique_inlier_landmarks"], 1)
        self.assertEqual(f["max_reference_unique_inlier_queries"], 2)
        c["query_indices"][:] = 0
        with self.assertRaisesRegex(ValueError, "deduplicated"):
            self.score(c)

    def test_negative_depth_and_outlier_remain_in_fraction_denominator(self):
        c = fixture(xy=[[10, 10], [10, 10], [50, 50]],
                    xyz=[[10, 10, 1], [-10, -10, -1], [100, 50, 1]])
        f = self.score(c)["features"]
        self.assertEqual(f["unique_inlier_queries"], 1)
        self.assertAlmostEqual(f["inlier_fraction"], 1 / 3)
        self.assertAlmostEqual(f["positive_depth_pair_fraction"], 2 / 3)
        self.assertEqual(f["spatial_coverage"], 0)

    def test_reprojection_k_colmap_offset_and_exact_raw_pose_preservation(self):
        theta = .17
        raw = np.eye(4)
        raw[:3, :3] = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        raw[:3, 3] = [4.1, -2.3, .75]
        self.pred["raw_c2w"] = raw.tolist()
        # Prefer explicitly preserved raw pose, not a different c2w field.
        K = np.array([[80., 0, 30.5], [0, 90., 40.5], [0, 0, 1.]])
        self.query["K"] = K.tolist()
        camera = np.array([[0., 0., 2.], [.25, -.1, 2.]])
        world = camera @ raw[:3, :3].T + raw[:3, 3]
        uv = camera @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        before = copy.deepcopy(self.pred)
        result = self.score(fixture(xy=uv, xyz=world))
        self.assertLess(result["features"]["p90_inlier_reprojection_px"], 1e-10)
        self.assertEqual(result["raw_c2w"], before["raw_c2w"])
        self.assertEqual(self.pred, before)
        self.assertEqual(np.asarray(result["raw_c2w"]).tobytes(), raw.tobytes())
        result["raw_c2w"][0][0] = 7
        self.assertEqual(self.pred, before)
        self.assertFalse(result["pose_modified"])

    def test_fallback_is_evidence_not_an_acceptance_decision(self):
        self.pred["status"] = "retrieval_top1_fallback"
        result = self.score(fixture())
        self.assertFalse(result["features"]["pnp_success"])
        self.assertEqual(result["features"]["unique_inlier_queries"], 4)
        self.assertEqual(result["raw_c2w"], self.pred["c2w"])
        self.assertNotIn("accepted", result)

    def test_empty_correspondences_and_missing_pose(self):
        self.pred.update(c2w=None, status="no_retrieved_reference")
        result = self.score(fixture(xy=[]))
        self.assertEqual(result["features"]["inlier_fraction"], 0)
        self.assertIsNone(result["features"]["median_inlier_reprojection_px"])
        self.assertEqual(result["features"]["supporting_reference_count"], 0)
        self.assertIsNone(result["raw_c2w"])
        json.dumps(result, allow_nan=False)

    def test_malformed_or_nonfinite_inputs_are_not_silently_confident(self):
        c = fixture()
        c["points3d"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid correspondence"):
            self.score(c)
        self.pred["c2w"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite 4x4"):
            self.score(fixture())

    def test_capture_uses_exact_assembler_without_pnp(self):
        rec = SimpleNamespace(images={4: SimpleNamespace(name="r0"), 7: SimpleNamespace(name="r1")},
                              points3D={9: SimpleNamespace(xyz=[1, 2, 3]), 10: SimpleNamespace(xyz=[4, 5, 6])})
        def original_assembly(localizer, name, camera, dbids, features, matches):
            self.assertEqual((name, camera, dbids, features, matches), ("q", None, [4, 7], "f", "m"))
            self.assertIs(localizer.reconstruction, rec)
            all_xy = np.array([[1.5, 2.5], [8.5, 9.5]])
            ret = localizer.localize(all_xy, [0, 0, 1], [9, 10, 10], camera)
            self.assertIsNone(ret)
            return ret, dict(keypoints_query=all_xy[[0, 0, 1]], points3D_ids=[9, 10, 10], num_matches=7,
                             keypoint_index_to_db=([0, 0, 1], [(9, [0, 1]), (10, [1]), (10, [0, 1])]))
        c = confidence.assemble_correspondences(rec, "q", ["r0", "r1"], "f", "m", pose_from_cluster=original_assembly)
        self.assertEqual(c["raw_match_observation_count"], 7)
        self.assertEqual(c["query_indices"].tolist(), [0, 0, 1])
        self.assertEqual(c["point3d_ids"].tolist(), [9, 10, 10])
        self.assertEqual(c["reference_support"], [(0, 1), (1,), (0, 1)])
        self.assertEqual(c["pnp_calls"], 0)


if __name__ == "__main__":
    unittest.main()
