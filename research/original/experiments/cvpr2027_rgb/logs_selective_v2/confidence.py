"""Public, fixed-pose evidence for the LoGS initializer.

No label access, acceptance threshold selection, optimization, rendering or
photometric dependencies. The released hloc correspondence assembler is called
with a capture-only localizer: PnP is never rerun. Confidence counts each query
keypoint once, selecting its lowest-residual positive-depth landmark; repeated
reference observations and competing landmarks cannot inflate that count.
"""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np


def assemble_correspondences(
    reconstruction, query_name, db_names, features_path, matches_path, *,
    name_to_id=None, pose_from_cluster: Callable | None = None,
):
    """Run exact released hloc assembly without running its pose solver.

    Callers put the frozen LoGS checkout on sys.path before the default import.
    Dependency injection permits CPU tests of ordering and duplicate handling.
    Returned keypoints already include hloc's COLMAP +0.5 pixel offset.
    """
    if pose_from_cluster is None:
        from hloc.localize_sfm import pose_from_cluster
    if name_to_id is None:
        name_to_id = {image.name: ident for ident, image in reconstruction.images.items()}
    db_names = list(db_names)
    if len(set(db_names)) != len(db_names):
        raise ValueError("Retrieved reference names must be unique")
    db_ids = [name_to_id[name] for name in db_names]

    class CaptureOnly:
        def __init__(self):
            self.reconstruction = reconstruction
            self.captured = None

        def localize(self, points2D_all, points2D_idxs, points3D_id, query_camera, **kwargs):
            if self.captured is not None:
                raise ValueError("Unexpected repeated pose solver call")
            self.captured = (
                np.asarray(points2D_all)[points2D_idxs].copy(),
                np.asarray(points2D_idxs, dtype=np.int64),
                np.asarray(points3D_id, dtype=np.int64),
            )
            return None

    capture = CaptureOnly()
    ret, log = pose_from_cluster(
        capture, query_name, None, db_ids, features_path, matches_path,
    )
    if ret is not None or capture.captured is None:
        raise ValueError("Capture-only assembly unexpectedly produced a pose")
    xy, qi, pids = capture.captured
    logged_qi, logged_support = log["keypoint_index_to_db"]
    if list(logged_qi) != qi.tolist() or list(log["points3D_ids"]) != pids.tolist():
        raise ValueError("hloc captured and logged correspondence order disagree")
    if not np.array_equal(np.asarray(log["keypoints_query"]), xy):
        raise ValueError("hloc logged keypoints disagree with solver inputs")
    support = []
    for pid, (logged_pid, refs) in zip(pids, logged_support):
        if int(pid) != int(logged_pid):
            raise ValueError("hloc landmark/reference association disagreement")
        support.append(tuple(sorted({int(i) for i in refs})))
    if len(support) != len(pids):
        raise ValueError("Incomplete reference support")
    xyz = np.asarray([reconstruction.points3D[int(i)].xyz for i in pids], dtype=np.float64).reshape(-1, 3)
    return dict(
        keypoints_colmap=xy.reshape(-1, 2), query_indices=qi,
        point3d_ids=pids, points3d=xyz, reference_support=support,
        reference_names=db_names, raw_match_observation_count=int(log["num_matches"]),
        assembly="released_hloc.pose_from_cluster_capture_only",
        pnp_calls=0,
    )


def _hull_area(points):
    """Convex hull area without OpenCV/torch imports; deterministic float64."""
    points = sorted(set(map(tuple, np.asarray(points, dtype=np.float64).tolist())))
    if len(points) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.asarray(lower[:-1] + upper[:-1])
    return float(abs(np.sum(hull[:, 0] * np.roll(hull[:, 1], -1)
                            - hull[:, 1] * np.roll(hull[:, 0], -1))) * .5)


def extract_confidence(prediction, query, correspondences, *, reprojection_threshold_px=12.0):
    """Return JSON-safe evidence and an exact copy of the input raw pose.

    `inlier_fraction` denominator is ALL unique matched query keypoints, including
    those with no positive-depth projection. At most one landmark contributes
    per query keypoint; unique landmark support is reported separately. This is
    intentionally distinct from the saved num_inliers/num_matches statistic.
    No record is accepted or rejected here.
    """
    if not np.isfinite(reprojection_threshold_px) or reprojection_threshold_px <= 0:
        raise ValueError("Reprojection threshold must be finite and positive")
    raw_pose = copy.deepcopy(prediction.get("raw_c2w", prediction.get("c2w")))
    if isinstance(raw_pose, np.ndarray):
        raw_pose = raw_pose.tolist()
    pose = None if raw_pose is None else np.asarray(raw_pose, dtype=np.float64)
    if pose is not None:
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Raw pose must be a finite 4x4 matrix or None")
        if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
            raise ValueError("Raw pose has an invalid homogeneous last row")
        if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0):
            raise ValueError("Raw pose must contain an orthonormal rotation")
        if not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-6, rtol=0):
            raise ValueError("Raw pose rotation must be proper")
    K = np.asarray(query["K"], dtype=np.float64)
    width, height = float(query["width"]), float(query["height"])
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise ValueError("Invalid public camera calibration")
    xy = np.asarray(correspondences["keypoints_colmap"], dtype=np.float64)
    qi = np.asarray(correspondences["query_indices"], dtype=np.int64)
    pids = np.asarray(correspondences["point3d_ids"], dtype=np.int64)
    xyz = np.asarray(correspondences["points3d"], dtype=np.float64)
    refs = correspondences["reference_names"]
    support = correspondences["reference_support"]
    n = len(qi)
    if xy.shape != (n, 2) or xyz.shape != (n, 3) or pids.shape != (n,) or len(support) != n:
        raise ValueError("Malformed correspondence arrays")
    if not np.isfinite(xy).all() or not np.isfinite(xyz).all() or (qi < 0).any() or (pids < 0).any():
        raise ValueError("Invalid correspondence values")
    if len(set(zip(qi.tolist(), pids.tolist()))) != n:
        raise ValueError("Expected hloc-deduplicated query/landmark pairs")
    for s in support:
        if not s or any(not isinstance(i, (int, np.integer)) or not 0 <= int(i) < len(refs) for i in s):
            raise ValueError("Invalid reference support")
    # A query index must always denote precisely the same observed image point.
    unique_qi, query_counts = np.unique(qi, return_counts=True)
    query_order = np.argsort(qi, kind="stable")
    same_query = np.diff(qi[query_order]) == 0
    if np.any(np.any(np.diff(xy[query_order], axis=0) != 0, axis=1) & same_query):
        raise ValueError("A query keypoint has inconsistent coordinates")

    positive = np.zeros(n, dtype=bool)
    residuals = np.full(n, np.inf)
    if pose is not None and n:
        # Exact inverse of the saved affine transform, no pose orthogonalization.
        inverse = np.linalg.inv(pose)
        camera_xyz = xyz @ inverse[:3, :3].T + inverse[:3, 3]
        homogeneous = camera_xyz @ K.T
        positive = (camera_xyz[:, 2] > 0) & np.isfinite(homogeneous).all(axis=1)
        projectable = positive & (np.abs(homogeneous[:, 2]) > 0)
        projected = homogeneous[projectable, :2] / homogeneous[projectable, 2:3]
        residuals[projectable] = np.linalg.norm(projected - xy[projectable], axis=1)
    candidates = np.flatnonzero(positive & np.isfinite(residuals))
    # Group by query, then residual, then stable landmark identity. Input order
    # cannot change confidence when ambiguous matches have identical errors.
    order = np.lexsort((pids[candidates], residuals[candidates], qi[candidates]))
    candidates = candidates[order]
    first = np.r_[True, np.diff(qi[candidates]) != 0] if len(candidates) else np.zeros(0, bool)
    selected = candidates[first]
    inliers = selected[residuals[selected] <= reprojection_threshold_px]
    unique_queries = int(len(unique_qi))
    inlier_xy = xy[inliers]
    per_ref = [set() for _ in refs]
    for i in inliers:
        for ref in set(support[int(i)]):
            per_ref[int(ref)].add(int(qi[int(i)]))
    inside = ((inlier_xy[:, 0] >= 0) & (inlier_xy[:, 0] <= width)
              & (inlier_xy[:, 1] >= 0) & (inlier_xy[:, 1] <= height))
    # Out-of-image observations cannot increase image coverage.
    covered_xy = inlier_xy[inside]
    bins = np.floor(covered_xy / [width, height] * 4).astype(np.int64)
    bins = np.clip(bins, 0, 3)
    counts = [len(s) for s in per_ref]
    status = prediction.get("status", prediction.get("initializer_status"))
    features = dict(
        pnp_success=bool(status == "pnp" and pose is not None),
        raw_match_observation_count=int(correspondences["raw_match_observation_count"]),
        deduplicated_pair_count=n,
        unique_matched_queries=unique_queries,
        unique_matched_landmarks=int(len(np.unique(pids))),
        ambiguous_query_count=int(np.count_nonzero(query_counts > 1)),
        unique_inlier_queries=int(len(inliers)),
        unique_inlier_landmarks=int(len(np.unique(pids[inliers]))),
        inlier_fraction=float(len(inliers) / unique_queries) if unique_queries else 0.,
        positive_depth_pair_fraction=float(positive.sum() / n) if n else 0.,
        median_inlier_reprojection_px=float(np.median(residuals[inliers])) if len(inliers) else None,
        p90_inlier_reprojection_px=float(np.percentile(residuals[inliers], 90)) if len(inliers) else None,
        spatial_coverage=_hull_area(covered_xy) / (width * height),
        inlier_occupied_4x4_grid_fraction=len(set(map(tuple, bins.tolist()))) / 16.,
        supporting_reference_count=sum(int(c > 0) for c in counts),
        max_reference_unique_inlier_queries=max(counts, default=0),
        second_reference_unique_inlier_queries=sorted(counts, reverse=True)[1] if len(counts) > 1 else 0,
        multi_reference_inlier_query_fraction=sum(len(set(support[int(i)])) >= 2 for i in inliers) / len(inliers) if len(inliers) else 0.,
    )
    for value in features.values():
        if value is not None and isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError("Nonfinite confidence feature")
    result = dict(
        schema_version=1, method_id="logs_no_photo_v1", feature_schema="logs_confidence_v1",
        query_id=prediction.get("query_id", query.get("id")),
        scene=query.get("scene"), split=query.get("split"),
        raw_c2w=raw_pose, initializer_status=status,
        pose_modified=False, pnp_calls=0, photometric_refinement=False,
        reprojection_threshold_px=float(reprojection_threshold_px), features=features,
        reference_support=[dict(reference_name=name, unique_inlier_queries=count)
                           for name, count in zip(refs, counts)],
        definitions=dict(
            inlier_fraction="unique query keypoints with selected positive-depth landmark residual <= threshold / all unique matched query keypoints",
            selected_landmark="lowest positive-depth reprojection residual per query keypoint; landmark ID breaks ties",
            spatial_coverage="convex hull area of in-image unique inlier query keypoints / image area",
            positive_depth_pair_fraction="positive-depth deduplicated query/landmark pairs / all deduplicated pairs",
            keypoint_coordinates="released hloc COLMAP convention: saved SuperPoint keypoints + 0.5",
        ),
    )
    return result
