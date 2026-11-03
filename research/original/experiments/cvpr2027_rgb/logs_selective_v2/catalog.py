"""Public-only input binding for the versioned LoGS geometric RGB route.

No private query labels are read here. Legacy experiment assets are read-only;
the feature index order is the one used to triangulate the training-only SfM.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RGB = ROOT / "experiments/cvpr2027_rgb"
UPSTREAM = RGB / "third_party/LoGS"
SCENES = ("block_small", "block_a", "nyc", "sf", "citysample", "colosseum",
          "elvenruin", "park", "road")
VALIDATION_SCENES = ("block_a", "sf")
EXPECTED_VALIDATION_COUNTS = {"block_a": 366, "sf": 124}
EXPECTED_VALIDATION_COUNT = sum(EXPECTED_VALIDATION_COUNTS.values())
PUBLIC_FIELDS = {"id", "image_path", "image_sha256", "K", "width", "height", "split"}
FEATURE_CONFIG = {
    "output": "feats-superpoint-n4096-r1024",
    "model": {"name": "superpoint", "nms_radius": 3, "max_keypoints": 4096},
    "preprocessing": {"globs": ["*.png"], "grayscale": True, "resize_max": 1024},
}
MATCHER_CONFIG = {"output": "matches-superglue",
                  "model": {"name": "superglue", "weights": "outdoor", "sinkhorn_iterations": 5}}
RECIPE = {
    "method": "logs_no_photo_v1", "retrieval": "netvlad", "top_k": 10,
    "feature_export_float16": True, "pnp_ransac_error_px": 12.0,
    "covisibility_clustering": False, "photometric_refinement": False,
    "gaussian_map_required": False, "acceptance_status": "uncalibrated",
    "pose_solver": "pycolmap.estimate_and_refine_absolute_pose",
    "fallback": "retain_raw_retrieved_camera_without_acceptance_claim",
}
SMOKE_TOLERANCES = {"descriptor_atol": 0.0001, "descriptor_rtol": 0.001,
                    "descriptor_min_cosine": 0.999999, "pose_max_abs_native": 1e-6}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    """Exclusive create, including for plans/receipts: never overwrite evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def canonical_protocol(scene):
    if scene not in SCENES:
        raise ValueError("Unknown frozen nine-scene scene")
    if scene == "block_small":
        return RGB / "logs/full_native_v1/protocol.json"
    if scene == "block_a":
        return RGB / "logs/block_a_full_native_v1/protocol.json"
    leaf = "logs_v2" if scene in ("park", "road") else "logs"
    return RGB / "other_baselines_nine_completion_20261004_v1" / scene / leaf / "protocol.json"


def public_queries(path, split, *, verify_pixels=True):
    if split not in ("validation", "test"):
        raise ValueError("Only frozen validation/test splits are supported")
    value = read(path)
    frames = value["frames"] if isinstance(value, dict) else value
    if not isinstance(frames, list):
        raise ValueError("Public manifest must contain a list of queries")
    result, seen = [], set()
    for q in frames:
        if not isinstance(q, dict) or set(q) != PUBLIC_FIELDS:
            raise ValueError("Query contains forbidden or missing public fields")
        if not isinstance(q["id"], str) or not q["id"] or "/" in q["id"] or "\\" in q["id"]:
            raise ValueError("Unsafe query id")
        if q["id"] in seen:
            raise ValueError("Duplicate query id")
        seen.add(q["id"])
        if q["split"] not in ("validation", "test"):
            raise ValueError("Unrecognized public query split")
        if any(isinstance(q[k], bool) or not isinstance(q[k], int) or q[k] <= 0 for k in ("width", "height")):
            raise ValueError("Invalid public dimensions")
        k = q["K"]
        if (not isinstance(k, list) or len(k) != 3 or any(not isinstance(r, list) or len(r) != 3 for r in k)
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for r in k for x in r)
                or k[2] != [0, 0, 1] or k[0][0] <= 0 or k[1][1] <= 0 or k[0][1] != 0 or k[1][0] != 0):
            raise ValueError("Invalid zero-skew pinhole intrinsics")
        if q["split"] != split:
            continue
        image = Path(q["image_path"])
        if image.name != q["id"] + ".png":
            raise ValueError("Released image naming contract requires <id>.png")
        if verify_pixels:
            if sha(image) != q["image_sha256"]:
                raise ValueError("Public query RGB checksum mismatch")
            from PIL import Image
            with Image.open(image) as source:
                if source.size != (q["width"], q["height"]):
                    raise ValueError("Public RGB dimensions differ from calibration")
        result.append(q)
    if not result:
        raise ValueError("No queries selected")
    if len({str(Path(q["image_path"]).parent) for q in result}) != 1:
        raise ValueError("This released batch adapter requires a common public image root")
    return sorted(result, key=lambda q: q["id"])


def _h5_names(handle):
    import h5py
    names = set()
    handle.visititems(lambda _, obj: names.add(obj.parent.name.strip("/"))
                      if isinstance(obj, h5py.Dataset) else None)
    return names


def qualify_h5_membership(rec, features, refglobal, names):
    import h5py
    import numpy as np
    images = list(rec.images.values())
    landmark_ids = set(rec.points3D)
    if {im.name for im in images} != set(names):
        raise ValueError("SfM images differ from training membership")
    with h5py.File(features, "r") as local, h5py.File(refglobal, "r") as glob:
        if _h5_names(local) != set(names) or _h5_names(glob) != set(names):
            raise ValueError("Feature cache training membership mismatch")
        for im in images:
            row = local[im.name]
            n = len(row["keypoints"])
            # COLMAP may omit keypoints for registered cameras with no 3-D
            # observations. Released hloc skips these cameras before indexing.
            omitted_empty = not len(im.points2D) and n > 0
            if n != len(im.points2D) and not omitted_empty:
                raise ValueError("Reference keypoint indices differ from triangulation")
            if row["descriptors"].dtype != np.float16 or row["keypoints"].dtype != np.float16:
                raise ValueError("Reference cache is not released float16 export")
            if row["descriptors"].shape != (256, n):
                raise ValueError("Reference descriptor shape mismatch")
            desc = glob[im.name]["global_descriptor"]
            if desc.dtype != np.float16 or desc.shape != (4096,):
                raise ValueError("Reference NetVLAD shape/precision mismatch")
            # points2D order, not coordinate quantization, defines feature IDs.
            for pt in im.points2D:
                if pt.has_point3D() and int(pt.point3D_id) not in landmark_ids:
                    raise ValueError("SfM observation references absent landmark")


def reference_assets(scene, *, verify_content=True):
    """Check full cache hashes, training membership and HLOC/SfM index identity.

    No GPU libraries or query labels are loaded. Hashing is intentionally done
    before GPU admission; source training RGB need not be reread to reuse the
    byte-identical frozen descriptors and triangulated landmark model.
    """
    legacy = canonical_protocol(scene)
    p = read(legacy)
    sfm = Path(p["sfm_source"])
    train_path, receipt_path = sfm / "protocol.json", sfm / "summary.json"
    train, receipt = read(train_path), read(receipt_path)
    if train["feature_config"] != FEATURE_CONFIG or train["matcher"] != MATCHER_CONFIG["model"]:
        raise ValueError("Training features/matcher differ from exact LoGS recipe")
    names = [r["name"] for r in train["records"]]
    if len(set(names)) != len(names) or len(names) != p["training_count"]:
        raise ValueError("Training membership is not unique/complete")
    if receipt["status"] != "complete" or receipt["image_count"] != len(names):
        raise ValueError("Training-only SfM is incomplete")
    if receipt["protocol_sha256"] != sha(train_path):
        raise ValueError("Training protocol checksum mismatch")
    features = sfm / (FEATURE_CONFIG["output"] + ".h5")
    refglobal = legacy.parent / "initializer/reference_netvlad.h5"
    initial_receipt = legacy.parent / "initializer/summary.json"
    if read(initial_receipt)["status"] != "complete":
        raise ValueError("Historical reference descriptor producer incomplete")
    assets = {
        "scene": scene, "legacy_protocol": str(legacy), "training_protocol": str(train_path),
        "sfm_receipt": str(receipt_path), "sfm_dir": str(sfm / "sfm"),
        "reference_features": str(features), "reference_netvlad": str(refglobal),
        "reference_names": names, "training_count": len(names),
        "upstream": str(UPSTREAM), "torch_cache": p["torch_cache"],
        "seed": p["seed"], "netvlad_weights": p["netvlad_weights"],
        "netvlad_weights_sha256": p["netvlad_weights_sha256"],
        "meters_per_model_unit": p["meters_per_model_unit"],
        "scale_status": p.get("scale_status", "source_verified"),
        "bindings_sha256": {}, "qualified": False,
        "reference_image_root": str(sfm / "images"),
        "smoke_reference_name": names[0],
        "smoke_reference_source": train["records"][0]["source"],
        "smoke_reference_sha256": train["records"][0]["sha256"],
    }
    bindings = {str(x): sha(x) for x in (legacy, train_path, receipt_path, initial_receipt)}
    expected = dict(receipt["artifacts_sha256"])
    expected[str(features)] = receipt["features_sha256"]
    expected[str(Path(p["netvlad_weights"]))] = p["netvlad_weights_sha256"]
    if not verify_content:
        assets["expected_existing_sha256"] = expected
        return assets
    import pycolmap
    for path, wanted in expected.items():
        actual = sha(path)
        if actual != wanted:
            raise ValueError("Frozen reference asset changed: " + path)
        bindings[path] = actual
    bindings[str(refglobal)] = sha(refglobal)
    first_image = sfm / "images" / names[0]
    if sha(first_image) != train["records"][0]["sha256"]:
        raise ValueError("Pinned smoke reference image differs from training input")
    bindings[str(first_image)] = train["records"][0]["sha256"]
    rec = pycolmap.Reconstruction(sfm / "sfm")
    qualify_h5_membership(rec, features, refglobal, names)
    assets.update(bindings_sha256=bindings, qualified=True,
                  qualification={"training_images": len(names), "sfm_points3D": len(rec.points3D),
                                 "reference_keypoint_index_order": "exact_triangulation_feature_file",
                                 "feature_precision": "float16", "pycolmap": pycolmap.__version__,
                                 "reference_netvlad_historical_digest_available": False,
                                 "reference_netvlad_binding": "New full-file digest; requires two-scene GPU descriptor parity before validation",
                                 "empty_landmark_images_skipped_by_hloc": sum(not im.num_points3D for im in rec.images.values())})
    return assets


def source_bindings():
    sources = list((UPSTREAM / "hloc").rglob("*.py"))
    sources += [HERE / n for n in ("catalog.py", "prepare.py", "run_initializer.py")]
    weights = UPSTREAM / "hloc/third_party/SuperGluePretrainedNetwork/models/weights"
    sources += [weights / "superpoint_v1.pth", weights / "superglue_outdoor.pth"]
    return {str(p): sha(p) for p in sorted(set(sources))}


def shared_query_rows(source_queries, target_queries):
    """Image-only feature reuse; intrinsics and scene-local matches stay separate."""
    by_pixels = {}
    for query in source_queries:
        key = (query["image_sha256"], query["width"], query["height"])
        by_pixels.setdefault(key, query)
    rows = []
    for query in target_queries:
        key = (query["image_sha256"], query["width"], query["height"])
        source = by_pixels.get(key)
        if source is not None:
            rows.append(dict(source_query_id=source["id"], target_query_id=query["id"],
                             image_sha256=key[0], width=key[1], height=key[2]))
    return rows


def verify_plan(plan):
    if plan.get("scene") not in VALIDATION_SCENES:
        raise ValueError("This qualification campaign validates Block A and SF only")
    if plan.get("purpose") not in ("validation", "smoke"):
        raise ValueError("Unsupported initializer plan purpose")
    expected_split = "validation" if plan["purpose"] == "validation" else "test"
    if plan.get("split") != expected_split:
        raise ValueError("Initializer purpose does not match its frozen split")
    if (plan.get("validation_scenes") != list(VALIDATION_SCENES)
            or plan.get("test_scenes") != list(SCENES)):
        raise ValueError("Plan changes the two-scene calibration or nine-scene test scope")
    if plan["recipe"] != RECIPE or plan["feature_config"] != FEATURE_CONFIG or plan["matcher_config"] != MATCHER_CONFIG:
        raise ValueError("Plan changes the released geometric recipe")
    if plan["reference_assets"].get("qualified") is not True:
        raise ValueError("Reference assets have not passed CPU qualification")
    for path, expected in {**plan["source_sha256"], **plan["reference_assets"]["bindings_sha256"]}.items():
        if sha(path) != expected:
            raise ValueError("Frozen source or reference changed: " + path)
    source = plan["public_queries"]
    if sha(source["path"]) != source["sha256"]:
        raise ValueError("Public query manifest changed")
    all_rows = public_queries(source["path"], plan["split"])
    by_id = {q["id"]: q for q in all_rows}
    if len(plan["queries"]) != len({q["id"] for q in plan["queries"]}):
        raise ValueError("Duplicate planned query")
    for q in plan["queries"]:
        if q != by_id.get(q.get("id")):
            raise ValueError("Planned query differs from public source")
    if not plan["queries"]:
        raise ValueError("Empty planned batch")
    if plan["purpose"] == "validation":
        if (plan["queries"] != all_rows
                or len(plan["queries"]) != EXPECTED_VALIDATION_COUNTS[plan["scene"]]):
            raise ValueError("Validation plan must include the complete frozen scene split")
    reuse = plan.get("query_feature_reuse")
    if reuse is not None:
        if plan["purpose"] != "validation" or plan["scene"] != "sf":
            raise ValueError("Only SF validation can reuse completed Block A query features")
        if sha(reuse["source_plan"]) != reuse["source_plan_sha256"]:
            raise ValueError("Query feature producer plan changed")
        producer = read(reuse["source_plan"])
        if (producer.get("scene") != "block_a" or producer.get("purpose") != "validation"
                or producer.get("split") != "validation"
                or producer["source_sha256"] != plan["source_sha256"]
                or any(producer[key] != plan[key] for key in ("recipe", "feature_config", "matcher_config"))
                or reuse["rows"] != shared_query_rows(producer["queries"], plan["queries"])):
            raise ValueError("Query feature reuse changes producer recipe or public image identity")
    if plan.get("purpose") == "smoke":
        if plan["queries"] != all_rows[:2]:
            raise ValueError("Implementation smoke must contain the first two sorted public test IDs")
        if plan["smoke_tolerances"] != SMOKE_TOLERANCES:
            raise ValueError("Smoke tolerances changed")
        binding = plan["smoke_original_predictions"]
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("Original raw initializer predictions changed")
