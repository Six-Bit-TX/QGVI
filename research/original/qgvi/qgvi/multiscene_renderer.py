"""Frozen HorizonGS rendering in native checkpoint coordinates, without Scene.

``load_scene_model(scene)`` reads only config, PLY and saved TorchScript MLPs.
It never opens dataset images, creates a Scene, fits appearance, or transforms
checkpoint coordinates. ``render_scene_frame(model, frame)`` takes a zero-skew
pinhole K and an OpenCV c2w in MODEL units and returns RGB, expected camera-Z
depth in the SAME units, and accumulated alpha. In particular, this module
must not inherit the Block A exporter's hard-coded metres conversion.

Scene keys: model_path, horizon_repo, iteration (40000), explicit (False),
lod_resolution_scale (1), appearance_policy ("fixed"), appearance_uid (0).
Appearance choices are frozen: "fixed" uses a saved training embedding row;
"frame_uid" requires frame["uid"]; "mean" uses the mean saved embedding.
A UID is the camera enumeration index used when training, NOT a COLMAP ID,
filename number, sorted camera index or novel-view index. This mapping can be
shuffled in HorizonGS training; callers must supply an established mapping.
The policy has no color effect for appearance_dim=0 or baked explicit PLYs.
No query image or query-fitted embedding is accepted by this API.

Changing raster dimensions does not change LoD: scale defaults to 1, with an
explicit frame["lod_resolution_scale"] override. Native camera centres are
also passed unchanged to the view-dependent MLPs and distance-based LoD.
Torch/HorizonGS are imported lazily; validation helpers are CPU-testable.
"""

from __future__ import annotations

import importlib
import math
import operator
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np


DEFAULT_HORIZON_REPO = Path("/mnt/t5evo/dataset/3drecon/largescale/HorizonGS")


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(result)


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _checkpoint_spec(scene: dict) -> SimpleNamespace:
    """Validate file/config choices without importing Torch or reading images."""
    import yaml

    repo = Path(scene.get("horizon_repo", DEFAULT_HORIZON_REPO)).expanduser().resolve()
    root = Path(scene["model_path"]).expanduser().resolve()
    iteration = _integer(scene.get("iteration", 40000), "iteration", 1)
    explicit = scene.get("explicit", False)
    if not isinstance(explicit, bool):
        raise ValueError("explicit must be a boolean")
    if not (repo / "scene/lod_model.py").is_file():
        raise FileNotFoundError(f"Not a HorizonGS checkout: {repo}")
    config = yaml.safe_load((root / "config.yaml").read_text())
    params = config["model_params"]
    if params["model_config"]["name"] != "GaussianLoDModel":
        raise ValueError("Only HorizonGS GaussianLoDModel checkpoints are supported")
    kwargs = dict(params["model_config"]["kwargs"])
    if kwargs["gs_attr"] != "3D":
        raise ValueError("Expected 3D Gaussian checkpoint for expected camera-Z depth")
    if kwargs.get("view_dim", 0) not in (0, 3):
        raise ValueError("HorizonGS neural renderer supports view_dim 0 or 3")
    if explicit and not kwargs["color_attr"].startswith("SH"):
        raise ValueError("Explicit PLY requires baked spherical-harmonic colors")
    if params.get("random_background", False):
        raise ValueError("Frozen rendering cannot use random_background")
    # Neither scale nor center is applied here: input cameras are already in
    # checkpoint coordinates, including any training-time recentering.
    kwargs["render_mode"] = "RGB+ED"
    policy = scene.get("appearance_policy", "fixed")
    if policy not in {"fixed", "frame_uid", "mean"}:
        raise ValueError("appearance_policy must be fixed, frame_uid, or mean")
    appearance_uid = _integer(scene.get("appearance_uid", 0), "appearance_uid")
    lod_scale = _positive(scene.get("lod_resolution_scale", 1.0), "lod_resolution_scale")
    checkpoint = root / "point_cloud" / f"iteration_{iteration}"
    ply = checkpoint / ("point_cloud_explicit.ply" if explicit else "point_cloud.ply")
    required = [ply]
    if not explicit:
        required.extend(checkpoint / name for name in ("opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt"))
        if kwargs.get("appearance_dim", 0) > 0:
            required.append(checkpoint / "embedding_appearance.pt")
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen checkpoint file is missing: {path}")
    return SimpleNamespace(repo=repo, root=root, checkpoint=checkpoint, ply=ply,
                           iteration=iteration, explicit=explicit, config=config,
                           params=params, kwargs=kwargs, appearance_policy=policy,
                           appearance_uid=appearance_uid, lod_resolution_scale=lod_scale)


def _import_horizon(repo: Path):
    """Fail clearly on generic module-name collisions between checkouts."""
    for name, relative in (("scene", "scene"), ("utils", "utils"),
                           ("gaussian_renderer", "gaussian_renderer")):
        module = sys.modules.get(name)
        if module is not None:
            filename = getattr(module, "__file__", None)
            locations = ([Path(filename).resolve().parent] if filename is not None
                         else [Path(path).resolve() for path in getattr(module, "__path__", [])])
            if locations != [repo / relative]:
                raise RuntimeError(f"Another {name} module is imported; use a fresh scene subprocess")
    if str(repo) in sys.path:
        sys.path.remove(str(repo))
    sys.path.insert(0, str(repo))
    model_module = importlib.import_module("scene.lod_model")
    renderer_module = importlib.import_module("gaussian_renderer")
    if (Path(model_module.__file__).resolve() != repo / "scene/lod_model.py"
            or Path(renderer_module.__file__).resolve().parent != repo / "gaussian_renderer"):
        raise RuntimeError("Loaded HorizonGS modules came from another checkout; use a fresh subprocess")
    return model_module.GaussianLoDModel, renderer_module.render


def _appearance_uid(model: SimpleNamespace, frame: dict) -> int:
    """Resolve a saved embedding index, never infer one from frame IDs."""
    if model.appearance_policy == "frame_uid" and model.appearance_active:
        if "uid" not in frame:
            raise ValueError("frame_uid appearance policy requires the saved training frame uid")
        uid = _integer(frame["uid"], "frame uid")
    elif model.appearance_active and model.appearance_policy == "fixed":
        uid = model.appearance_uid
    else:
        uid = _integer(frame.get("uid", 0), "frame uid")
    if model.appearance_active and model.appearance_policy != "mean":
        if uid >= model.appearance_count:
            raise ValueError(f"Appearance uid {uid} is outside the saved {model.appearance_count}-row table")
    return uid


def _camera_spec(frame: dict, lod_resolution_scale: float = 1.0) -> SimpleNamespace:
    """Construct CPU camera data; translation is deliberately not rescaled."""
    width, height = (_integer(frame[key], key, 1) for key in ("width", "height"))
    k = np.asarray(frame["K"], dtype=np.float64)
    pose = np.asarray(frame["c2w"], dtype=np.float64)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or not np.allclose([k[0, 1], k[1, 0]], 0)):
        raise ValueError("K must be a finite, positive-focal, zero-skew pinhole matrix")
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])):
        raise ValueError("c2w must be a finite homogeneous OpenCV camera pose in MODEL units")
    rotation = pose[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
        raise ValueError("c2w must contain a proper orthonormal rotation")
    if frame["domain"] not in {"street", "aerial"}:
        raise ValueError("domain must be street or aerial")
    return SimpleNamespace(image_name=str(frame["id"]), image_type=frame["domain"],
                           image_width=width, image_height=height,
                           fx=float(k[0, 0]), fy=float(k[1, 1]),
                           cx=float(k[0, 2]), cy=float(k[1, 2]),
                           world_view_transform=np.linalg.inv(pose).T.copy(),
                           camera_center=pose[:3, 3].copy(),
                           resolution_scale=_positive(frame.get("lod_resolution_scale", lod_resolution_scale),
                                                      "lod_resolution_scale"))


def load_scene_model(scene: dict) -> SimpleNamespace:
    """Load an immutable final implicit or explicit checkpoint on current CUDA.

    Use one subprocess per scene to isolate HorizonGS's generic imports and
    release all CUDA allocations on exit. PLY LoD metadata and saved MLPs are
    kept intact; no camera-dependent weeding or training setup is performed.
    """
    spec = _checkpoint_spec(scene)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("HorizonGS rendering requires CUDA in the existing 3dcs environment")
    gaussian_class, renderer = _import_horizon(spec.repo)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    gaussians = gaussian_class(**spec.kwargs)
    gaussians.explicit_gs = spec.explicit
    gaussians.weed_ratio = 0.0
    appearance_active = not spec.explicit and spec.kwargs.get("appearance_dim", 0) > 0
    if spec.explicit:
        gaussians.load_explicit(str(spec.ply))
        # Explicit RGB/SH is already baked and never uses the embedding MLP.
        # eval() otherwise tries to evaluate a nonexistent embedding table.
        gaussians.appearance_dim = 0
    else:
        gaussians.load_ply(str(spec.ply))
        gaussians.load_mlp_checkpoints(str(spec.checkpoint))
    appearance_count = 0
    if appearance_active:
        table = gaussians.embedding_appearance.embedding.weight
        if table.ndim != 2 or table.shape[1] != spec.kwargs["appearance_dim"] or table.shape[0] == 0:
            raise ValueError("Saved appearance embedding table has an incompatible shape")
        appearance_count = int(table.shape[0])
        if spec.appearance_policy == "mean":
            class FrozenMeanAppearance(torch.nn.Module):
                def __init__(self, value):
                    super().__init__()
                    self.register_buffer("value", value.detach().mean(dim=0, keepdim=True))

                def forward(self, indices):
                    return self.value.expand(*indices.shape, self.value.shape[-1])

            gaussians.embedding_appearance = FrozenMeanAppearance(table)
        gaussians.ape_code = spec.appearance_uid if spec.appearance_policy == "fixed" else -1
    gaussians.eval()
    gaussians.freeze()
    for value in vars(gaussians).values():
        if torch.is_tensor(value) and value.is_floating_point():
            value.requires_grad_(False)
        elif isinstance(value, torch.nn.Module):
            value.eval()
            for parameter in value.parameters():
                parameter.requires_grad_(False)
    pipeline = SimpleNamespace(**spec.config.get("pipeline_params", {}))
    pipeline.add_prefilter = bool(getattr(pipeline, "add_prefilter", True))
    # Exact radius-zero culling is needed for the large merged explicit model.
    pipeline.explicit_prefilter = bool(spec.explicit)
    background = (torch.ones if spec.params.get("white_background", False) else torch.zeros)(3, device="cuda")
    torch.cuda.synchronize()
    model = SimpleNamespace(gaussians=gaussians, pipeline=pipeline, background=background,
                            renderer=renderer, scene_id=str(scene.get("id", spec.root.name)),
                            model_path=str(spec.root), checkpoint_path=str(spec.ply),
                            iteration=spec.iteration, explicit=spec.explicit,
                            appearance_policy=spec.appearance_policy, appearance_uid=spec.appearance_uid,
                            appearance_active=appearance_active, appearance_count=appearance_count,
                            lod_resolution_scale=spec.lod_resolution_scale,
                            load_elapsed_s=time.perf_counter() - started,
                            load_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            load_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if model.appearance_active and model.appearance_policy == "fixed":
        _appearance_uid(model, {})
    return model


def _validate_arrays(rgb: np.ndarray, depth: np.ndarray, alpha: np.ndarray,
                     height: int, width: int) -> None:
    if (rgb.shape != (height, width, 3) or depth.shape != (height, width)
            or alpha.shape != (height, width)):
        raise ValueError("Renderer returned RGB/depth/alpha shapes inconsistent with the camera")
    if not all(np.isfinite(array).all() for array in (rgb, depth, alpha)):
        raise ValueError("Renderer returned non-finite RGB/depth/alpha")
    if np.any(depth < 0) or np.any(alpha < -1e-5) or np.any(alpha > 1 + 1e-5):
        raise ValueError("Renderer returned negative camera-Z depth or invalid alpha")


class _EmptyGaussianSet(Exception):
    """Internal signal: visibility/opacity selection produced no primitives."""


class _EmptyGaussianGuard:
    """Delegate model state unchanged; guard generated arrays before gsplat."""

    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        value = getattr(self._model, name)
        if name not in ('generate_neural_gaussians', 'generate_explicit_gaussians'):
            return value

        def generate(*args, **kwargs):
            result = value(*args, **kwargs)
            if result[0].shape[0] == 0:
                raise _EmptyGaussianSet()
            return result

        return generate


def _render_allow_empty(renderer, camera, gaussians, pipeline, background):
    """Zero primitives have exact background RGB and zero depth/opacity.

    The installed gsplat projection divides by N before its empty-input check,
    causing SIGFPE for valid views whose neural opacity selection yields N=0.
    A proxy intercepts that case without changing model methods or nonempty
    rendering, and without modifying the shared external HorizonGS checkout.
    """
    try:
        return renderer(camera, _EmptyGaussianGuard(gaussians), pipeline, background)
    except _EmptyGaussianSet:
        height, width = int(camera.image_height), int(camera.image_width)
        return dict(render=background[:, None, None].expand(3, height, width),
                    render_depth=background.new_zeros((1, height, width)),
                    render_alphas=background.new_zeros((1, height, width)),
                    empty_gaussian_set=True)


def render_scene_frame(model: SimpleNamespace, frame: dict) -> dict:
    """Render RGB uint8, expected camera-Z depth float32 MODEL units, alpha.

    No resizing, metric conversion, scene transform or appearance fitting is
    done. Output planes are finite HxW; empty pixels may have zero depth/alpha.
    Invalid rasterizer output raises instead of silently repairing geometry.
    """
    camera = _camera_spec(frame, model.lod_resolution_scale)
    camera.uid = _appearance_uid(model, frame)
    import torch

    camera.world_view_transform = torch.as_tensor(camera.world_view_transform, dtype=torch.float32, device="cuda")
    camera.camera_center = torch.as_tensor(camera.camera_center, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        result = _render_allow_empty(model.renderer, camera, model.gaussians,
                                     model.pipeline, model.background)
        if result.get("render_depth") is None:
            raise RuntimeError("HorizonGS did not return RGB+ED expected camera-Z depth")
        rgb_float = result["render"].permute(1, 2, 0).to(torch.float32).cpu().numpy()
        depth = result["render_depth"].squeeze(0).to(torch.float32).cpu().numpy()
        alpha = result["render_alphas"].squeeze(0).to(torch.float32).cpu().numpy()
        empty_gaussian_set = bool(result.get('empty_gaussian_set', False))
        del result
    _validate_arrays(rgb_float, depth, alpha, camera.image_height, camera.image_width)
    rgb = np.rint(np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
    alpha = np.clip(alpha, 0, 1)
    torch.cuda.synchronize()
    diagnostics = {"elapsed_s": time.perf_counter() - started,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "depth_type": "expected_camera_z", "depth_units": "model",
                   "coordinate_system": "native_model_opencv", "uid": camera.uid,
                   "appearance_policy": model.appearance_policy,
                   "appearance_active": model.appearance_active,
                   "lod_resolution_scale": camera.resolution_scale,
                   "alpha_fraction_over_half": float(np.mean(alpha > 0.5)),
                   "empty_gaussian_set": empty_gaussian_set}
    return {"rgb": rgb, "depth": depth, "alpha": alpha, **diagnostics, "diagnostics": diagnostics}
