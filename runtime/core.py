"""Path/configuration adapters; algorithm source lives unchanged in research/original."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "research/original"
FULL = ORIGINAL / "experiments/qgvi_a_full_eval_20261029_v1"
sys.path.insert(0, str(ORIGINAL / "qgvi-rag"))


def load_original(relative, name):
    if name in sys.modules:
        return sys.modules[name]
    path = ORIGINAL / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def geometry_module():
    return load_original("experiments/agentic_locator_comparison_20261025_v1/geometry_adapter.py", "qgvi_original_geometry")


class TextIndex:
    """Run the original BM25/scoring/selection methods on a portable caption manifest.

    No model, caption generation, semantic verification or device policy is added.
    Index frames must describe the actual file pixels (including its intrinsics).
    """
    def __init__(self, path, scene=None):
        self.path = Path(path).resolve()
        data = read(self.path)
        source = data.get("views", data.get("text", {}).get("views", []))
        if scene is None and len({r.get("scene") for r in source}) > 1:
            raise ValueError("Choose one scene: camera coordinates and retrieval scores are scene-specific")
        module = load_original("experiments/qgvi_a_full_eval_20261029_v1/retrieval_backend.py", "qgvi_original_retrieval")
        self.module = module
        self.backend = module.RetrievalBackend.__new__(module.RetrievalBackend)
        views = {}
        for row in source:
            if scene and row.get("scene") != scene:
                continue
            ident = row.get("view_id", row.get("id"))
            image = Path(row.get("image_path", row.get("image", "")))
            if not image.is_absolute():
                image = self.path.parent / image
            frame = copy.deepcopy(row["frame"])
            record = {"view_id": ident, "caption": row["caption"], "frame": frame,
                      "image_path": str(image.resolve()),
                      "image_sha256": row.get("image_sha256") or sha(image)}
            if not ident or ident in views:
                raise ValueError("Empty or duplicate view ID")
            module.native_frame(frame)
            views[ident] = record
        if not views:
            raise ValueError("No indexed views match the selected scene")
        self.backend.views = views
        self.backend.bm25 = module.BM25({v: r["caption"] for v, r in views.items()})
        self.backend.excluded = {}
        self.backend.bound_source = None
        self.backend.queries = {}
        self.backend.binding_sha = sha(path)
        # Original resolver validates pixels and frame dimensions on every return.
        self.backend.resolve = self.resolve
        self.backend._query = self._query

    def _query(self, bundle):
        if bundle is not None and self.backend.bound_source not in (None, bundle):
            raise ValueError("Source binding differs")
        ident = bundle or self.backend.bound_source
        if ident is not None and ident not in self.backend.excluded:
            raise ValueError("Unconfigured source binding")
        return ident

    def configure_source(self, bundle_id, image_sha256, reference_names=()):
        data = read(self.path)
        rows = data.get("views", data.get("text", {}).get("views", []))
        named = {r.get("view_id", r.get("id")) for r in rows
                 if r.get("reference_name") in set(reference_names)}
        direct = named | {
            ident for ident, row in self.backend.views.items() if row["image_sha256"] == image_sha256}
        poses = [self.backend.views[v]["frame"]["c2w"] for v in direct if v in self.backend.views]
        self.backend.excluded[bundle_id] = direct | {
            ident for ident, row in self.backend.views.items()
            if any(np.allclose(row["frame"]["c2w"], p, rtol=0, atol=1e-6) for p in poses)}

    def resolve(self, view_id, bundle=None):
        from PIL import Image
        row = self.backend.views[view_id]
        if sha(row["image_path"]) != row["image_sha256"]:
            raise ValueError("Indexed pixels changed")
        with Image.open(row["image_path"]) as im:
            if im.size != (row["frame"]["width"], row["frame"]["height"]):
                raise ValueError("Indexed image dimensions do not match intrinsics")
        return copy.deepcopy(row)

    def bind_source(self, bundle_id):
        self._query(bundle_id)
        self.backend.bound_source = bundle_id

    def search_text(self, query, limit=6, bundle_id=None):
        result = self.backend.search_text(query, limit=limit, bundle_id=bundle_id)
        result["exclusion_policy"] = "Exact supplied pixels and explicit reference names; no reconstruction-held-out claim"
        return result


def solve_correspondences(query, path, region=None):
    geometry = geometry_module()
    result = geometry.solve(query, geometry.load_corr(path), region)
    accepted = (result["status"] == "estimated" and not result["fallback_used"]
                and result["evidence"]["unique_inlier_queries"] >= 40)
    return {**result, "accepted": accepted,
            "accepted_pose_c2w": result["pose_c2w"] if accepted else None,
            "input_scope": "Fresh PnP and gate; supplied correspondences contain prior feature extraction/matching",
            "minimum_unique_inliers": 40}


class LiveLocalizer:
    """Fresh input pixels through the unchanged source-preparation runtime.

    The original runtime's CUDA requirement is preserved. No CPU fallback or
    device override is inserted. The caller controls visibility/environment.
    """
    def __init__(self, scene_config, workdir, upstream=None, torch_cache=None):
        self.config_path = Path(scene_config).resolve()
        self.config = read(self.config_path)
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.geometry = geometry_module()
        if upstream:
            self.geometry.UPSTREAM = Path(upstream).resolve()
        if not (self.geometry.UPSTREAM / "hloc").is_dir():
            raise ValueError("Install the pinned LoGS dependency with scripts/install_upstream.py")
        self.preparation = load_original("experiments/qgvi_a_full_eval_20261029_v1/source_geometry.py", "qgvi_original_source_geometry")
        self.asset = copy.deepcopy(self.config.get("rgb", self.config))
        for key in ("reference_features", "reference_global", "sfm_dir"):
            path = Path(self.asset[key])
            self.asset[key] = str(path if path.is_absolute() else self.config_path.parent / path)
        self.asset["torch_cache"] = str(Path(torch_cache or ROOT / ".cache/torch").resolve())
        self.runtime = self.preparation.Runtime(self.geometry)
        self.reference = self.runtime.load_reference(self.asset)
        self.last = None

    def localize(self, query):
        query = copy.deepcopy(query)
        query["image_path"] = str(Path(query["image_path"]).resolve())
        query["image_sha256"] = sha(query["image_path"])
        query.setdefault("id", query["image_sha256"][:24])
        query.setdefault("name", query["id"] + ".png")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", query["id"]):
            raise ValueError("Invalid query id")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.png", query["name"]):
            raise ValueError("Invalid query filename")
        query.setdefault("scene", self.config.get("scene", "custom"))
        query.setdefault("excluded_reference_names", [])
        from PIL import Image
        with Image.open(query["image_path"]) as im:
            if im.size != (query["width"], query["height"]):
                raise ValueError("Query dimensions do not match K/image metadata")
        # Exclusion is explicit for custom inputs. Do not inherit the research
        # receipt's claim that every possible new input was self-excluded.
        binding = self.geometry.digest({"scene": sha(self.config_path), "query": query})
        receipt = self.preparation.prepare_one_source(query, self.asset, self.workdir,
            self.reference, self.runtime, plan_sha256=binding, top_k=10)
        result = read(receipt["paths"]["result"])
        result["self_reference_excluded"] = bool(query["excluded_reference_names"])
        result["exclusion_policy"] = "Caller-provided reference exclusions; no held-out-data claim"
        result["accepted"] = bool(result.get("pose_c2w") is not None and not result.get("fallback_used")
                                  and result["evidence"]["unique_inlier_queries"] >= 40)
        result["accepted_pose_c2w"] = result["pose_c2w"] if result["accepted"] else None
        self.last = {"query": query, "receipt": receipt, "result": result}
        return result


class LiveNavigation:
    """Call the original frozen HorizonGS renderer and camera motion functions."""
    def __init__(self, scene_config, output, horizon_repo):
        self.path = Path(scene_config).resolve()
        self.config = read(self.path)
        spec = copy.deepcopy(self.config.get("gaussian", self.config))
        path = Path(spec["model_path"])
        spec["model_path"] = str(path if path.is_absolute() else self.path.parent / path)
        spec["horizon_repo"] = str(Path(horizon_repo).resolve())
        self.renderer = load_original("qgvi/qgvi/multiscene_renderer.py", "qgvi_original_renderer")
        self.navigation = load_original("experiments/qgvi_a_full_eval_20261029_v1/navigation_backend.py", "navigation_backend")
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.step = self.config.get("step_native", self.navigation.STEPS.get(self.config.get("scene")))
        if not self.step:
            raise ValueError("scene.json must specify positive native camera step")
        cameras_path = self.path.parent / self.config["reference_cameras"]
        records = read(cameras_path)["records"]
        bounds = self.navigation.camera_bounds(records, self.step)
        checkpoint = self.renderer._checkpoint_spec(spec)
        files = [self.path, cameras_path, checkpoint.root / "config.yaml", checkpoint.ply]
        if not checkpoint.explicit:
            files += sorted(checkpoint.checkpoint.glob("*.pt"))
        binding_files = {str(p): {"sha256": sha(p), "size_bytes": p.stat().st_size,
                                  "mtime_ns": p.stat().st_mtime_ns} for p in files}
        self.model = self.renderer.load_scene_model(spec)
        binding = {"files": binding_files, "step_native": self.step,
                   "camera_center_bounds": bounds, "minimum_alpha_coverage": .05,
                   "appearance_policy": spec.get("appearance_policy", "fixed")}
        self.cache = self.navigation.RenderCache(self.config.get("scene", "custom"), self.output,
            binding, lambda frame: self.renderer.render_scene_frame(self.model, frame))

    def propose_view(self, frame):
        return self.cache.propose_view(frame)

    def navigate(self, frame, movement):
        return self.cache.navigate(frame, movement)
