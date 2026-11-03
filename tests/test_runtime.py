"""Portable adapter checks using original BM25 and real CPU pycolmap PnP."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pycolmap
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import core


def view(tmp_path, ident, caption, *, color, x, scene="sample"):
    path = tmp_path / (ident + ".png")
    Image.new("RGB", (32, 24), color).save(path)
    pose = np.eye(4)
    pose[0, 3] = x
    return {"view_id": ident, "scene": scene, "caption": caption,
            "image_path": path.name, "image_sha256": core.sha(path),
            "frame": {"c2w": pose.tolist(), "K": [[20, 0, 16], [0, 20, 12], [0, 0, 1]],
                      "width": 32, "height": 24, "domain": "street"}}


def index_file(tmp_path):
    rows = [view(tmp_path, "a", "red car", color="red", x=0),
            view(tmp_path, "b", "red car", color="red", x=1),
            view(tmp_path, "c", "red car", color="pink", x=0),
            view(tmp_path, "d", "blue bus", color="blue", x=2),
            view(tmp_path, "e", "red car", color="white", x=3)]
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"views": rows}))
    return path, rows


def test_original_bm25_and_candidate_deduplication_are_used(tmp_path):
    path, rows = index_file(tmp_path)
    index = core.TextIndex(path)
    actual = index.search_text("red car", limit=5)
    assert [row["view_id"] for row in actual["candidates"]] == ["a", "e"]
    original_scores = index.module.BM25({r["view_id"]: r["caption"] for r in rows}).scores("red car")
    assert [row["score"] for row in actual["candidates"]] == [original_scores["a"], original_scores["e"]]
    assert index.search_text("submarine", limit=5)["candidates"] == []


def test_selected_pixels_and_camera_dimensions_must_match(tmp_path):
    path, rows = index_file(tmp_path)
    index = core.TextIndex(path)
    Image.new("RGB", (32, 24), "green").save(tmp_path / "a.png")
    with pytest.raises(ValueError, match="pixels changed"):
        index.search_text("red car")
    rows[0]["image_sha256"] = core.sha(tmp_path / "a.png")
    rows[0]["frame"]["width"] = 64
    path.write_text(json.dumps({"views": rows}))
    with pytest.raises(ValueError, match="dimensions"):
        core.TextIndex(path).search_text("red car")


def test_invalid_intrinsics_are_rejected_before_retrieval(tmp_path):
    path, rows = index_file(tmp_path)
    rows[0]["frame"]["K"][0][0] = -20
    path.write_text(json.dumps({"views": rows}))
    with pytest.raises(ValueError, match="camera"):
        core.TextIndex(path)


def test_multi_scene_coordinate_systems_are_not_silently_deduplicated(tmp_path):
    path, rows = index_file(tmp_path)
    rows[1]["scene"] = "other"
    path.write_text(json.dumps({"views": rows}))
    with pytest.raises(ValueError):
        core.TextIndex(path)
    assert len(core.TextIndex(path, scene="other").search_text("red car")["candidates"]) == 1


def test_source_binding_excludes_identical_pixels_and_same_camera(tmp_path):
    path, rows = index_file(tmp_path)
    index = core.TextIndex(path)
    index.configure_source("source", rows[0]["image_sha256"], [])
    index.bind_source("source")
    result = index.search_text("red car", bundle_id="source")
    assert [r["view_id"] for r in result["candidates"]] == ["e"]
    assert result["self_exclusion_applied"] is True


def test_bound_source_cannot_switch_to_another_query_silently(tmp_path):
    path, rows = index_file(tmp_path)
    index = core.TextIndex(path)
    index.configure_source("one", rows[0]["image_sha256"])
    index.configure_source("two", rows[3]["image_sha256"])
    index.bind_source("one")
    with pytest.raises(ValueError, match="binding differs"):
        index.search_text("red car", bundle_id="two")


def test_nested_demo_manifest_honors_explicit_reference_exclusions(tmp_path):
    path, rows = index_file(tmp_path)
    rows[0]["reference_name"] = "known_source.png"
    path.write_text(json.dumps({"text": {"views": rows}}))
    index = core.TextIndex(path)
    index.configure_source("one", "0" * 64, ["known_source.png"])
    index.bind_source("one")
    results = index.search_text("red car", bundle_id="one")["candidates"]
    assert "a" not in {row["view_id"] for row in results}
    assert "c" not in {row["view_id"] for row in results}


@pytest.mark.parametrize("field,value", [("id", "../escape"), ("id", "/tmp/escape"),
                                        ("name", "../escape.png"), ("name", "/tmp/escape.png")])
def test_source_names_cannot_escape_fresh_preparation_directory(tmp_path, field, value):
    image = tmp_path / "source.png"
    Image.new("RGB", (32, 24), "gray").save(image)
    config = tmp_path / "scene.json"
    config.write_text('{"scene":"sample"}')
    localizer = core.LiveLocalizer.__new__(core.LiveLocalizer)
    localizer.config_path, localizer.config = config, {"scene": "sample"}
    localizer.workdir = tmp_path / "run"
    localizer.geometry = core.geometry_module()
    localizer.asset, localizer.reference, localizer.runtime = {}, {}, object()
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid path reached feature preparation")
    localizer.preparation = SimpleNamespace(prepare_one_source=forbidden)
    query = {"id": "source", "name": "source.png", "image_path": str(image),
             "width": 32, "height": 24, "K": [[20, 0, 16], [0, 20, 12], [0, 0, 1]]}
    query[field] = value
    with pytest.raises(ValueError):
        localizer.localize(query)


def correspondences(tmp_path, count, *, duplicate_query_ids=False):
    rng = np.random.default_rng(13)
    points = rng.uniform([-2, -1, 4], [2, 1, 8], size=(count, 3))
    k = np.array([[300., 0., 320.], [0., 300., 240.], [0., 0., 1.]])
    xy = (points @ k.T)[:, :2] / points[:, 2:3]
    indices = np.arange(count, dtype=np.int64)
    if duplicate_query_ids:
        points = np.concatenate([points, points])
        xy = np.concatenate([xy, xy])
        indices = np.concatenate([indices, indices])
    n = len(points)
    corr = {"keypoints_colmap": xy, "query_indices": indices,
            "point3d_ids": np.arange(n, dtype=np.int64), "points3d": points,
            "reference_support": [[0, 1] for _ in range(n)],
            "reference_names": ["reference_1", "reference_2"],
            "raw_match_observation_count": n * 2}
    path = tmp_path / "correspondences.npz"
    core.geometry_module().save_corr(path, corr)
    query = {"id": "synthetic", "K": k.tolist(), "width": 640, "height": 480}
    return query, path


@pytest.mark.parametrize("count,accepted", [(40, True), (39, False), (3, False)])
def test_original_pnp_and_distinct_gate_are_executed(tmp_path, count, accepted):
    assert pycolmap.__version__ == "3.13.0", "Run in the qualified release environment"
    query, path = correspondences(tmp_path, count)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = core.solve_correspondences(query, path)
    assert result["accepted"] is accepted
    assert result["fallback_used"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    if count >= 4:
        assert result["status"] == "estimated"
        np.testing.assert_allclose(result["pose_c2w"], np.eye(4), atol=1e-6)
        assert result["evidence"]["unique_inlier_queries"] == count
    else:
        assert result["pose_c2w"] is None
    if accepted:
        assert result["accepted_pose_c2w"] == result["pose_c2w"]
    else:
        assert result["accepted_pose_c2w"] is None


def test_duplicate_landmark_hypotheses_do_not_inflate_distinct_gate(tmp_path):
    query, path = correspondences(tmp_path, 20, duplicate_query_ids=True)
    result = core.solve_correspondences(query, path)
    assert result["status"] == "estimated"
    assert result["evidence"]["unique_inlier_queries"] == 20
    assert result["accepted"] is False


def test_roi_preserves_intrinsics_and_reduces_available_evidence(tmp_path):
    query, path = correspondences(tmp_path, 50)
    original_query = copy.deepcopy(query)
    result = core.solve_correspondences(query, path, region=[0., 0., .1, .1])
    assert result["accepted"] is False
    assert query == original_query
    assert result["roi_semantics"] == "filter full-image correspondences; K unchanged"


def test_original_algorithm_files_still_match_export_manifest():
    manifest = core.read(core.ROOT / "research/source_manifest.json")
    for row in manifest["files"]:
        path = core.ROOT / row["release_path"]
        assert core.sha(path) == row["sha256"], row["source_path"]


def navigation_fixture(tmp_path, monkeypatch, depth=1.):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.yaml").write_text("fixture")
    (checkpoint / "point_cloud.ply").write_text("fixture")
    frame = {"c2w": np.eye(4).tolist(), "K": [[5, 0, 2], [0, 5, 2], [0, 0, 1]],
             "width": 4, "height": 4, "domain": "street"}
    (tmp_path / "cameras.json").write_text(json.dumps({"records": [frame]}))
    config = tmp_path / "scene.json"
    config.write_text(json.dumps({"scene": "fixture", "step_native": .1,
                                 "reference_cameras": "cameras.json",
                                 "gaussian": {"model_path": "checkpoint", "explicit": True}}))
    calls = []
    def render(model, camera):
        calls.append(copy.deepcopy(camera))
        return {"rgb": np.zeros((4, 4, 3), np.uint8),
                "depth": np.full((4, 4), depth, float), "alpha": np.ones((4, 4)),
                "elapsed_s": .001}
    renderer = SimpleNamespace(
        _checkpoint_spec=lambda spec: SimpleNamespace(root=checkpoint, ply=checkpoint / "point_cloud.ply", explicit=True),
        load_scene_model=lambda spec: object(), render_scene_frame=render)
    loader = core.load_original
    monkeypatch.setattr(core, "load_original", lambda relative, name: renderer if name == "qgvi_original_renderer" else loader(relative, name))
    return core.LiveNavigation(config, tmp_path / "renders", tmp_path), frame, calls


def test_render_adapter_rejects_alpha_without_supported_depth(tmp_path, monkeypatch):
    navigation, frame, calls = navigation_fixture(tmp_path, monkeypatch, depth=0.)
    with pytest.raises(ValueError, match="supported pixels"):
        navigation.propose_view(frame)
    assert len(calls) == 1


def test_render_adapter_uses_original_cache_and_camera_bounds(tmp_path, monkeypatch):
    navigation, frame, calls = navigation_fixture(tmp_path, monkeypatch)
    before = copy.deepcopy(frame)
    initial = navigation.propose_view(frame)
    replay = navigation.propose_view(frame)
    assert initial["coverage"] == 1. and initial["valid"] is True
    assert initial["cache_hit"] is False and replay["cache_hit"] is True
    assert initial["image_sha256"] == replay["image_sha256"]
    assert frame == before and len(calls) == 1
    distant = copy.deepcopy(frame)
    distant["c2w"][0][3] = 10.
    with pytest.raises(ValueError, match="camera_out_of_bounds"):
        navigation.propose_view(distant)
    assert len(calls) == 1


def test_changed_checkpoint_invalidates_render_cache(tmp_path, monkeypatch):
    navigation, frame, calls = navigation_fixture(tmp_path, monkeypatch)
    navigation.propose_view(frame)
    (tmp_path / "checkpoint/config.yaml").write_text("changed fixture")
    with pytest.raises(ValueError, match="input changed"):
        navigation.propose_view(frame)
    assert len(calls) == 1
