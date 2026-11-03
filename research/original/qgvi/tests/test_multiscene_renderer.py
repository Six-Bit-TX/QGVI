"""CPU-only checks for the isolated renderer's native-coordinate contract."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from qgvi import multiscene_renderer as renderer


def frame():
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = [100, -200, 50]
    return {"id": "novel:17", "width": 4, "height": 3, "domain": "aerial",
            "K": [[7, 0, 1.5], [0, 8, 1], [0, 0, 1]], "c2w": pose.tolist()}


def appearance_model(policy="fixed", active=True, uid=2, count=4):
    return SimpleNamespace(appearance_policy=policy, appearance_active=active,
                           appearance_uid=uid, appearance_count=count)


def checkpoint(tmp_path, explicit=False, appearance_dim=0):
    repo, root = tmp_path / "horizon", tmp_path / "model"
    (repo / "scene").mkdir(parents=True)
    (repo / "scene/lod_model.py").touch()
    directory = root / "point_cloud/iteration_40000"
    directory.mkdir(parents=True)
    names = ["point_cloud_explicit.ply"] if explicit else ["point_cloud.ply", "opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt"]
    if not explicit and appearance_dim:
        names.append("embedding_appearance.pt")
    for name in names:
        (directory / name).touch()
    config = {"model_params": {"model_config": {"name": "GaussianLoDModel", "kwargs": {
        "gs_attr": "3D", "color_attr": "SH2" if explicit else "RGB",
        "view_dim": 0 if explicit else 3, "appearance_dim": appearance_dim,
        "render_mode": "RGB"}}, "center": [123, -56, 8], "scale": 0.25},
        "pipeline_params": {"add_prefilter": True}}
    (root / "config.yaml").write_text(json.dumps(config))
    return {"id": "example", "model_path": str(root), "horizon_repo": str(repo), "explicit": explicit}


def test_camera_native_units_and_opencv_axes_are_preserved():
    source = frame()
    unchanged = copy.deepcopy(source)
    camera = renderer._camera_spec(source)
    np.testing.assert_array_equal(camera.camera_center, [100, -200, 50])
    w2c = camera.world_view_transform.T
    np.testing.assert_allclose(w2c @ np.asarray(source["c2w"]), np.eye(4), atol=1e-12)
    np.testing.assert_allclose(w2c @ [100, -200, 55, 1], [0, 0, 5, 1])
    assert (camera.fx, camera.fy, camera.cx, camera.cy) == (7, 8, 1.5, 1)
    assert camera.resolution_scale == 1
    assert source == unchanged


def test_lod_scale_is_explicit_not_inferred_from_dimensions():
    source = frame()
    assert renderer._camera_spec(source, 2).resolution_scale == 2
    source["lod_resolution_scale"] = 3
    assert renderer._camera_spec(source, 2).resolution_scale == 3
    source["lod_resolution_scale"] = 0
    with pytest.raises(ValueError, match="lod_resolution_scale"):
        renderer._camera_spec(source)


@pytest.mark.parametrize("key,value", [("width", 0), ("width", 4.5), ("height", True),
                                      ("domain", "unknown")])
def test_bad_camera_metadata_rejected(key, value):
    source = frame()
    source[key] = value
    with pytest.raises(ValueError):
        renderer._camera_spec(source)


@pytest.mark.parametrize("key,row,col,value", [("K", 0, 1, 1), ("K", 0, 0, -1),
                                              ("c2w", 3, 0, 1), ("c2w", 2, 2, -1),
                                              ("c2w", 0, 3, float("nan"))])
def test_bad_camera_matrices_rejected(key, row, col, value):
    source = frame()
    source[key][row][col] = value
    with pytest.raises(ValueError):
        renderer._camera_spec(source)


def test_fixed_appearance_ignores_novel_uid_and_checks_bounds():
    assert renderer._appearance_uid(appearance_model(), {"uid": 999}) == 2
    with pytest.raises(ValueError, match="outside"):
        renderer._appearance_uid(appearance_model(uid=4), {})


def test_frame_appearance_needs_real_training_uid_not_frame_id():
    model = appearance_model("frame_uid")
    with pytest.raises(ValueError, match="requires"):
        renderer._appearance_uid(model, {"id": "train/2.png"})
    assert renderer._appearance_uid(model, {"uid": 3}) == 3
    for uid in (-1, 4, 2.5, True):
        with pytest.raises(ValueError):
            renderer._appearance_uid(model, {"uid": uid})


def test_no_embedding_and_mean_do_not_index_the_training_table():
    assert renderer._appearance_uid(appearance_model("frame_uid", active=False), {}) == 0
    assert renderer._appearance_uid(appearance_model("mean"), {"uid": 999}) == 999


@pytest.mark.parametrize("explicit", [False, True])
def test_checkpoint_config_does_not_apply_scale_or_center(tmp_path, explicit):
    scene = checkpoint(tmp_path, explicit)
    spec = renderer._checkpoint_spec(scene)
    assert spec.kwargs["render_mode"] == "RGB+ED"
    assert spec.params["center"] == [123, -56, 8]
    assert spec.kwargs["view_dim"] == (0 if explicit else 3)
    assert spec.ply.name == ("point_cloud_explicit.ply" if explicit else "point_cloud.ply")
    assert spec.iteration == 40000


def test_implicit_loader_requires_saved_mlps(tmp_path):
    scene = checkpoint(tmp_path, appearance_dim=8)
    spec = renderer._checkpoint_spec(scene)
    (spec.checkpoint / "embedding_appearance.pt").unlink()
    with pytest.raises(FileNotFoundError, match="embedding_appearance"):
        renderer._checkpoint_spec(scene)


@pytest.mark.parametrize("key,value", [("explicit", "False"), ("iteration", -1),
                                      ("appearance_policy", "fit_query"), ("appearance_uid", -1)])
def test_invalid_checkpoint_options_rejected(tmp_path, key, value):
    scene = checkpoint(tmp_path)
    scene[key] = value
    with pytest.raises(ValueError):
        renderer._checkpoint_spec(scene)


def cpu_torch(monkeypatch):
    """Leave real tensor arithmetic intact but route this module off CUDA."""
    torch = pytest.importorskip("torch")
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    for name in ("as_tensor", "ones", "zeros"):
        original = getattr(torch, name)
        def on_cpu(*args, _original=original, **kwargs):
            kwargs.pop("device", None)
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, on_cpu)
    return torch


@pytest.mark.parametrize("explicit", [False, True])
def test_loading_calls_only_frozen_checkpoint_methods(tmp_path, monkeypatch, explicit):
    torch = cpu_torch(monkeypatch)
    scene = checkpoint(tmp_path, explicit)
    calls = []

    class FakeGaussians:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self._anchor = torch.ones(2, 3, requires_grad=True)
            self.mlp_color = torch.nn.Linear(3, 3)

        def load_ply(self, path):
            calls.append(("ply", path))

        def load_explicit(self, path):
            calls.append(("explicit", path))

        def load_mlp_checkpoints(self, path):
            calls.append(("mlp", path))

        def eval(self):
            pass

        def freeze(self):
            pass

    monkeypatch.setattr(renderer, "_import_horizon", lambda repo: (FakeGaussians, object()))
    model = renderer.load_scene_model(scene)
    assert [call[0] for call in calls] == (["explicit"] if explicit else ["ply", "mlp"])
    assert model.gaussians.weed_ratio == 0
    assert model.gaussians.view_dim == (0 if explicit else 3)
    assert not model.gaussians._anchor.requires_grad
    assert all(not value.requires_grad for value in model.gaussians.mlp_color.parameters())
    assert model.pipeline.explicit_prefilter is explicit


def test_render_returns_model_depth_and_correct_dtypes_without_scene(monkeypatch):
    torch = cpu_torch(monkeypatch)
    seen = []

    def fake_render(camera, gaussians, pipeline, background):
        seen.append(camera)
        return {"render": torch.full((3, 3, 4), 0.5),
                "render_depth": torch.full((1, 3, 4), 5.0),
                "render_alphas": torch.full((1, 3, 4), 0.75)}

    model = appearance_model(active=False)
    model.__dict__.update(renderer=fake_render, lod_resolution_scale=1,
                          gaussians=None, pipeline=None, background=None)
    result = renderer.render_scene_frame(model, frame())
    np.testing.assert_array_equal(seen[0].camera_center.numpy(), [100, -200, 50])
    np.testing.assert_array_equal(result["depth"], np.full((3, 4), 5.0, dtype=np.float32))
    assert result["rgb"].dtype == np.uint8 and result["rgb"].shape == (3, 4, 3)
    assert result["depth"].dtype == result["alpha"].dtype == np.float32
    assert result["depth_units"] == "model"
    assert result["rgb"][0, 0, 0] == 128


@pytest.mark.parametrize("policy", ["fixed", "frame_uid", "mean"])
def test_saved_appearance_is_frozen_without_query_fitting(tmp_path, monkeypatch, policy):
    torch = cpu_torch(monkeypatch)
    scene = checkpoint(tmp_path, appearance_dim=2)
    scene.update(appearance_policy=policy, appearance_uid=1)

    class SavedAppearance(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding.from_pretrained(
                torch.tensor([[1., 2.], [3., 4.], [5., 6.]]), freeze=False)

        def forward(self, indices):
            return self.embedding(indices)

    class FakeGaussians:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def load_ply(self, path):
            pass

        def load_mlp_checkpoints(self, path):
            self.embedding_appearance = SavedAppearance()

        def eval(self):
            pass

        def freeze(self):
            pass

    monkeypatch.setattr(renderer, "_import_horizon", lambda repo: (FakeGaussians, object()))
    model = renderer.load_scene_model(scene)
    assert model.appearance_active and model.appearance_count == 3
    assert model.gaussians.ape_code == (1 if policy == "fixed" else -1)
    embedding = model.gaussians.embedding_appearance
    assert not embedding.training
    assert all(not parameter.requires_grad for parameter in embedding.parameters())
    if policy == "mean":
        actual = embedding(torch.tensor([0, 999])).numpy()
        np.testing.assert_array_equal(actual, [[3, 4], [3, 4]])
    else:
        np.testing.assert_array_equal(embedding(torch.tensor([0])).numpy(), [[1, 2]])


@pytest.mark.parametrize("bad", ["depth_nan", "depth_negative", "alpha_range", "shape"])
def test_invalid_render_arrays_are_not_silently_repaired(bad):
    rgb = np.zeros((3, 4, 3), dtype=np.float32)
    depth = np.ones((3, 4), dtype=np.float32)
    alpha = np.ones_like(depth)
    if bad == "depth_nan":
        depth[0, 0] = np.nan
    elif bad == "depth_negative":
        depth[0, 0] = -1
    elif bad == "alpha_range":
        alpha[0, 0] = 1.1
    else:
        rgb = rgb[:, :-1]
    with pytest.raises(ValueError):
        renderer._validate_arrays(rgb, depth, alpha, 3, 4)
