"""CPU checks for public isolation, exact cache identity and raw-pose semantics."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from PIL import Image
import pytest

import catalog
from run_initializer import (copy_reused_query_features, prediction_record, qualify_descriptor, qualify_pose_parity,
                             require_runtime_version, require_smoke_qualification, run)


def public_fixture(tmp_path):
    image = tmp_path / "q.png"
    Image.new("RGB", (32, 24)).save(image)
    query = dict(id="q", image_path=str(image), image_sha256=catalog.sha(image),
                 K=[[20., 0., 16.], [0., 20., 12.], [0., 0., 1.]],
                 width=32, height=24, split="validation")
    manifest = tmp_path / "public.json"
    manifest.write_text(json.dumps(dict(frames=[query])))
    return manifest, query


@pytest.mark.parametrize("field", ["c2w", "query_pose", "in_map", "label", "depth_path"])
def test_public_boundary_rejects_private_fields(tmp_path, field):
    manifest, query = public_fixture(tmp_path)
    query[field] = "forbidden"
    manifest.write_text(json.dumps(dict(frames=[query])))
    with pytest.raises(ValueError, match="forbidden"):
        catalog.public_queries(manifest, "validation")


def test_public_pixels_and_intrinsics_are_bound(tmp_path):
    manifest, query = public_fixture(tmp_path)
    assert catalog.public_queries(manifest, "validation") == [query]
    Image.new("RGB", (32, 24), color="red").save(query["image_path"])
    with pytest.raises(ValueError, match="checksum"):
        catalog.public_queries(manifest, "validation")


def test_outputs_never_overwrite_completed_or_partial_runs(tmp_path):
    result = tmp_path / "result"
    result.mkdir()
    old = result / "partial.txt"
    old.write_text("preserve")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(dict(output_dir=str(result))))
    with pytest.raises(FileExistsError):
        run(plan)
    assert old.read_text() == "preserve"
    with pytest.raises(FileExistsError):
        catalog.write_new(old, {})


def test_raw_pnp_and_fallback_poses_are_exact_and_uncalibrated():
    w2c = np.eye(4)
    w2c[:3, 3] = [1.2, -3.4, 5.6]
    pose = SimpleNamespace(matrix=lambda: w2c[:3].copy())
    expected = np.linalg.inv(w2c).tolist()
    for pnp in (True, False):
        pred = prediction_record(dict(id="q"), pose, pnp=pnp, names=["train.png"],
                                 num_inliers=30 if pnp else 0, num_matches=40, seconds=.1)
        assert pred["raw_c2w"] == pred["c2w"] == expected
        assert pred["accepted"] is None
        assert pred["decision_status"] == "uncalibrated"
        assert pred["status"] == ("pnp" if pnp else "retrieval_top1_fallback")


def test_no_pose_is_complete_failure_not_infrastructure_failure():
    pred = prediction_record(dict(id="q"), None, pnp=False, names=[], num_inliers=0, num_matches=0, seconds=.1)
    assert pred["completed"] and not pred["raw_pose_available"]
    assert pred["raw_c2w"] is None and pred["accepted"] is None


def cache_fixture(tmp_path):
    local, glob = tmp_path / "local.h5", tmp_path / "global.h5"
    with h5py.File(local, "w") as f:
        f["train.png/keypoints"] = np.zeros((2, 2), np.float16)
        f["train.png/descriptors"] = np.zeros((256, 2), np.float16)
    with h5py.File(glob, "w") as f:
        f["train.png/global_descriptor"] = np.zeros(4096, np.float16)
    point = SimpleNamespace(has_point3D=lambda: True, point3D_id=7)
    rec = SimpleNamespace(images={1: SimpleNamespace(name="train.png", points2D=[point, point])}, points3D={7: object()})
    return rec, local, glob


def test_cache_indices_and_training_membership_are_checked(tmp_path):
    rec, local, glob = cache_fixture(tmp_path)
    catalog.qualify_h5_membership(rec, local, glob, ["train.png"])
    rec.images[1].points2D.pop()
    with pytest.raises(ValueError, match="indices"):
        catalog.qualify_h5_membership(rec, local, glob, ["train.png"])


def test_empty_landmark_cameras_follow_released_hloc_skip_rule(tmp_path):
    rec, local, glob = cache_fixture(tmp_path)
    rec.images[1].points2D = []
    catalog.qualify_h5_membership(rec, local, glob, ["train.png"])


def test_cache_rejects_extra_reference_or_wrong_feature_precision(tmp_path):
    rec, local, glob = cache_fixture(tmp_path)
    with h5py.File(glob, "a") as f:
        f["query.png/global_descriptor"] = np.zeros(4096, np.float16)
    with pytest.raises(ValueError, match="membership"):
        catalog.qualify_h5_membership(rec, local, glob, ["train.png"])
    with h5py.File(glob, "a") as f:
        del f["query.png"]
    with h5py.File(local, "a") as f:
        del f["train.png/descriptors"]
        f["train.png/descriptors"] = np.zeros((256, 2), np.float32)
    with pytest.raises(ValueError, match="float16"):
        catalog.qualify_h5_membership(rec, local, glob, ["train.png"])


def test_exact_recipe_has_no_photo_or_gaussian_dependency():
    assert catalog.RECIPE["method"] == "logs_no_photo_v1"
    assert catalog.RECIPE["top_k"] == 10
    assert catalog.RECIPE["pnp_ransac_error_px"] == 12.0
    assert catalog.FEATURE_CONFIG["preprocessing"]["resize_max"] == 1024
    assert catalog.MATCHER_CONFIG["model"]["sinkhorn_iterations"] == 5
    assert catalog.RECIPE["feature_export_float16"] is True
    assert catalog.RECIPE["photometric_refinement"] is False
    assert catalog.RECIPE["gaussian_map_required"] is False


def test_runtime_solver_must_match_original_and_qualification():
    require_runtime_version("3.13.0", "3.13.0")
    with pytest.raises(ValueError, match="solver"):
        require_runtime_version("3.14.0", "3.14.0")
    with pytest.raises(ValueError, match="solver"):
        require_runtime_version("3.13.0", "3.12.0")


def test_smoke_descriptor_gate_detects_changed_cache():
    vec = np.arange(4096, dtype=np.float32) / 4096
    assert qualify_descriptor(vec, vec.copy(), catalog.SMOKE_TOLERANCES)["passed"]
    changed = vec.copy()
    changed[1] = .5
    assert not qualify_descriptor(vec, changed, catalog.SMOKE_TOLERANCES)["passed"]


def test_smoke_pose_parity_requires_retrieval_and_inliers_too():
    pose = SimpleNamespace(matrix=lambda: np.eye(4)[:3])
    new = prediction_record(dict(id="q"), pose, pnp=True, names=["train.png"],
                            num_inliers=30, num_matches=40, seconds=.1)
    old = copy.deepcopy(new)
    assert qualify_pose_parity([new], [old], catalog.SMOKE_TOLERANCES)["passed"]
    old["num_inliers"] = 29
    assert not qualify_pose_parity([new], [old], catalog.SMOKE_TOLERANCES)["passed"]
    old = copy.deepcopy(new)
    old["c2w"][0][3] = 1e-4
    assert not qualify_pose_parity([new], [old], catalog.SMOKE_TOLERANCES)["passed"]


def test_full_validation_cannot_run_without_both_smokes():
    with pytest.raises(ValueError, match="both"):
        require_smoke_qualification(dict(purpose="validation"))
    with pytest.raises(ValueError, match="both Block A and SF"):
        require_smoke_qualification(dict(purpose="validation", required_smoke_summaries=[
            dict(scene="block_a"), dict(scene="road")]))
    require_smoke_qualification(dict(purpose="smoke"))


def test_full_validation_accepts_only_completed_block_a_and_sf_smokes(tmp_path):
    gates = []
    sources = {"frozen.py": "sha"}
    for scene in catalog.VALIDATION_SCENES:
        plan = tmp_path / (scene + "_plan.json")
        plan.write_text(json.dumps(dict(purpose="smoke", scene=scene, queries=[{}, {}],
                                        source_sha256=sources)))
        summary = tmp_path / (scene + "_summary.json")
        summary.write_text(json.dumps(dict(status="complete", plan_sha256=catalog.sha(plan),
                                           smoke_qualification=dict(passed=True))))
        gates.append(dict(scene=scene, plan=str(plan), summary=str(summary)))
    validation = dict(purpose="validation", source_sha256=sources, required_smoke_summaries=gates)
    require_smoke_qualification(validation)
    summary.write_text(json.dumps(dict(status="complete", plan_sha256=catalog.sha(plan),
                                       smoke_qualification=dict(passed=False))))
    with pytest.raises(ValueError, match="not passed"):
        require_smoke_qualification(validation)


def test_prepare_limits_calibration_to_complete_block_a_and_sf_splits(tmp_path, monkeypatch):
    import prepare as preparation
    called = []
    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    monkeypatch.setattr(preparation, "source_bindings", lambda: {"source.py": "frozen"})

    def references(scene):
        called.append(scene)
        return dict(qualified=True)

    monkeypatch.setattr(preparation, "reference_assets", references)
    for scene in catalog.VALIDATION_SCENES:
        root = tmp_path / "qgvi/.staging/horizongs_eval_v1" / scene
        root.mkdir(parents=True)
        (root / "queries.json").write_text("{}")
        (root / "initializer").mkdir()
        (root / "initializer/predictions.json").write_text("[]")
    monkeypatch.setattr(preparation, "canonical_protocol", lambda scene:
                        tmp_path / "qgvi/.staging/horizongs_eval_v1" / scene / "protocol.json")

    def public(path, split):
        scene = path.parent.name
        count = catalog.EXPECTED_VALIDATION_COUNTS[scene] if split == "validation" else 4
        return [dict(id=f"q{i:03d}", split=split, width=32, height=24,
                     image_sha256=f"shared{i}" if i < 3 else f"{scene}-{i}") for i in range(count)]

    monkeypatch.setattr(preparation, "public_queries", public)
    result = preparation.prepare(tmp_path / "prepared")
    assert called == ["block_a", "sf"]
    assert result["validation_queries"] == 490
    assert result["per_scene_queries"] == {"block_a": 366, "sf": 124}
    assert result["validation_scenes"] == ["block_a", "sf"]
    assert result["test_scenes"] == list(catalog.SCENES) and len(result["test_scenes"]) == 9
    assert sum(j["query_count"] for j in result["smoke_jobs"]) == 4
    assert result["private_labels_read"] is False and result["gpu_jobs_launched"] == 0
    assert result["cache_reuse"]["cross_scene_query_feature_rows_reused"] == 3
    assert result["cache_reuse"]["fresh_validation_feature_queries"] == 487
    for job in result["validation_jobs"]:
        plan = catalog.read(job["plan"])
        assert {g["scene"] for g in plan["required_smoke_summaries"]} == {"block_a", "sf"}
        assert len(plan["queries"]) == catalog.EXPECTED_VALIDATION_COUNTS[job["scene"]]
    for job in result["smoke_jobs"]:
        plan = catalog.read(job["plan"])
        assert plan["split"] == "test" and plan["purpose"] == "smoke"
        assert plan["query_feature_reuse"] is None
        assert [q["id"] for q in plan["queries"]] == ["q000", "q001"]


def test_plan_rejects_changed_scope_and_incomplete_validation(tmp_path, monkeypatch):
    rows = [dict(id=f"q{i:03d}") for i in range(366)]
    monkeypatch.setattr(catalog, "sha", lambda _: "digest")
    monkeypatch.setattr(catalog, "public_queries", lambda *a, **k: rows)
    plan = dict(scene="block_a", split="validation", purpose="validation",
                validation_scenes=list(catalog.VALIDATION_SCENES), test_scenes=list(catalog.SCENES),
                recipe=catalog.RECIPE, feature_config=catalog.FEATURE_CONFIG,
                matcher_config=catalog.MATCHER_CONFIG,
                reference_assets=dict(qualified=True, bindings_sha256={}), source_sha256={},
                public_queries=dict(path="unused", sha256="digest"), queries=rows)
    catalog.verify_plan(plan)
    bad = copy.deepcopy(plan)
    bad["queries"] = rows[:-1]
    with pytest.raises(ValueError, match="complete frozen"):
        catalog.verify_plan(bad)
    bad = copy.deepcopy(plan)
    bad["test_scenes"] = ["block_a", "sf"]
    with pytest.raises(ValueError, match="nine-scene test scope"):
        catalog.verify_plan(bad)
    bad = copy.deepcopy(plan)
    bad["scene"] = "road"
    with pytest.raises(ValueError, match="Block A and SF only"):
        catalog.verify_plan(bad)


def reusable_feature_fixture(tmp_path):
    source_dir, output = tmp_path / "source", tmp_path / "output"
    source_dir.mkdir()
    output.mkdir()
    image = tmp_path / "source.png"
    Image.new("RGB", (32, 24)).save(image)
    target_image = tmp_path / "target.png"
    target_image.write_bytes(image.read_bytes())
    source_query = dict(id="source", width=32, height=24, image_sha256=catalog.sha(image), image_path=str(image))
    target_query = dict(source_query, id="target", image_path=str(target_image))
    common = dict(purpose="validation", split="validation", recipe=catalog.RECIPE,
                  feature_config=catalog.FEATURE_CONFIG, matcher_config=catalog.MATCHER_CONFIG,
                  source_sha256={"shared.py": "digest"}, reference_assets=dict(
                      qualified=True, bindings_sha256={"reference": "digest"}, seed=123,
                      netvlad_weights_sha256="weights"))
    producer = dict(common, scene="block_a", queries=[source_query], output_dir=str(source_dir))
    source_plan = tmp_path / "source_plan.json"
    source_plan.write_text(json.dumps(producer))
    with h5py.File(source_dir / "query_netvlad.h5", "w") as f:
        f["source.png/global_descriptor"] = np.arange(4096, dtype=np.float16)
        f["source.png/image_size"] = np.array([32, 24])
    with h5py.File(source_dir / "query_superpoint.h5", "w") as f:
        f["source.png/keypoints"] = np.array([[1, 2], [3, 4]], np.float16)
        f["source.png/keypoints"].attrs["uncertainty"] = 1.0
        f["source.png/descriptors"] = np.arange(512, dtype=np.float16).reshape(256, 2)
        f["source.png/scores"] = np.array([.5, .7], np.float16)
        f["source.png/image_size"] = np.array([32, 24])
    summary = dict(status="complete", plan_sha256=catalog.sha(source_plan), query_count=1,
                   method=catalog.RECIPE["method"], source_bindings_sha256=producer["source_sha256"],
                   no_query_ground_truth_access=True, photometric_refinement=False,
                   reference_assets=producer["reference_assets"]["bindings_sha256"],
                   artifacts_sha256={str(p): catalog.sha(p) for p in source_dir.glob("*.h5")})
    (source_dir / "summary.json").write_text(json.dumps(summary))
    target = dict(common, scene="sf", queries=[target_query], query_feature_reuse=dict(
        source_plan=str(source_plan), source_plan_sha256=catalog.sha(source_plan),
        rows=catalog.shared_query_rows([source_query], [target_query])))
    return target, output, source_plan, source_dir


def test_cross_scene_feature_reuse_copies_exact_values_and_preserves_source(tmp_path):
    target, output, source_plan, source_dir = reusable_feature_fixture(tmp_path)
    original = {p: (catalog.sha(p), p.stat().st_mtime_ns) for p in source_dir.iterdir()}
    result = copy_reused_query_features(target, output)
    assert result["reused_query_count"] == 1 and result["source_open_mode"] == "read_only"
    assert result["fresh_stages"] == ["retrieval", "query_superglue", "pnp"]
    for filename in ("query_netvlad.h5", "query_superpoint.h5"):
        with h5py.File(source_dir / filename, "r") as src, h5py.File(output / filename, "r") as dst:
            assert set(dst) == {"target.png"}
            for key in src["source.png"]:
                np.testing.assert_array_equal(src["source.png"][key][()], dst["target.png"][key][()])
                assert dict(src["source.png"][key].attrs) == dict(dst["target.png"][key].attrs)
    assert {p: (catalog.sha(p), p.stat().st_mtime_ns) for p in source_dir.iterdir()} == original
    with pytest.raises(FileExistsError):
        copy_reused_query_features(target, output)


@pytest.mark.parametrize("mismatch", ["recipe", "dimensions", "pixels", "stale_plan", "stale_artifact", "partial", "smoke"])
def test_query_feature_reuse_rejects_incompatible_or_incomplete_sources(tmp_path, mismatch):
    target, output, source_plan, source_dir = reusable_feature_fixture(tmp_path)
    if mismatch == "recipe":
        target["feature_config"] = copy.deepcopy(target["feature_config"])
        target["feature_config"]["preprocessing"]["resize_max"] = 960
    elif mismatch == "dimensions":
        target["queries"][0]["width"] = 31
    elif mismatch == "pixels":
        Image.new("RGB", (32, 24), color="red").save(target["queries"][0]["image_path"])
    elif mismatch == "stale_plan":
        source_plan.write_text(source_plan.read_text() + "\n")
    elif mismatch == "stale_artifact":
        with h5py.File(source_dir / "query_netvlad.h5", "a") as f:
            f["source.png/global_descriptor"][0] = 99
    elif mismatch == "partial":
        path = source_dir / "summary.json"
        value = catalog.read(path)
        value["status"] = "running"
        path.write_text(json.dumps(value))
    elif mismatch == "smoke":
        target["purpose"] = "smoke"
    with pytest.raises(ValueError):
        copy_reused_query_features(target, output)
    assert list(output.iterdir()) == []
