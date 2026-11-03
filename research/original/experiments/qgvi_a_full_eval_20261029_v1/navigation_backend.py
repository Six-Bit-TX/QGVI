"""Resident, locked HorizonGS virtual camera; native OpenCV coordinates only.

This is free virtual-camera rendering, not collision-aware robot navigation.
No query ground-truth pose or semantic labels are read by this module.
"""
from __future__ import annotations

import argparse
import atexit
from contextlib import contextmanager, redirect_stdout
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import urllib.parse

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REGISTRY = ROOT / "qgvi/.staging/horizongs_eval_v1/scene_registry.json"
LOCKS = (ROOT / "experiments/cityloc_qgvi_v2/resources/gpu0.lock",
         Path("/home/xiongti1/slam/AlloMap/artifacts/cse_scaling_campaign_gpu.lock"))
# Median nearest-neighbour UNIQUE training camera-center spacing in native
# checkpoint units. Repeated camera centers at different orientations do not
# contribute zero-length steps. No source/test camera is used.
STEPS = {
    "block_small": 1.9290624999999988, "block_a": 0.19932434082031225,
    "nyc": 0.5783081054687502, "sf": 0.5783081054687506,
    "citysample": 0.19710083007812784, "colosseum": 0.0710124774340305,
    "elvenruin": 0.08749600816128868, "park": 0.13752715900907828,
    "road": 0.035347952935418814,
}
ORIGINAL_HORIZON = Path("/mnt/t5evo/dataset/3drecon/largescale/HorizonGS")
RECOVERED_HORIZON = Path("/mnt/hdd20tb/archive/t5evo-recovery-20261024/dataset/3drecon/largescale/HorizonGS")
RECOVERY_RECEIPT = Path("/mnt/hdd20tb/archive/storage-migration-20261023/t5-recovery-status.json")
ACTIONS = ("yaw_left", "yaw_right", "pitch_up", "pitch_down", "move_left", "move_right", "move_forward", "move_backward")
MIN_COVERAGE = 0.05
VERSION = "native_opencv_continuous_navigation_speed10_nine_scene_v1"
REFERENCE_ROOT = ROOT / "qgvi-rag/outputs/benchmark_cityloc_v2/original_rgb_banks_v1"
DEFAULT_URL = "http://127.0.0.1:8782"


def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


class NavigationError(ValueError):
    def __init__(self, kind, message=None, details=None):
        self.kind, self.details = kind, details or {}
        super().__init__(message or kind)


def validate_cache_namespace(value):
    if (not isinstance(value, str) or len(value) > 160
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)
            or value in {".", ".."}):
        raise NavigationError("invalid_cache_namespace")
    return value


def validate_pose(value):
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-8):
        raise ValueError("Expected finite homogeneous native OpenCV c2w")
    r = pose[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1e-5) or not np.isclose(np.linalg.det(r), 1, rtol=0, atol=1e-5):
        raise ValueError("Camera rotation must be proper and orthonormal")
    return pose


def move(c2w, movement, step_native):
    """Increment in the current OpenCV camera axes, never in inferred world axes."""
    pose = validate_pose(c2w).copy()
    if not math.isfinite(step_native) or step_native <= 0:
        raise NavigationError("invalid_step")
    data = {"action": movement} if isinstance(movement, str) else movement
    if not isinstance(data, dict) or set(data) - {"action", "distance", "angle_degrees"}:
        raise NavigationError("invalid_movement")
    action = {"forward": "move_forward", "backward": "move_backward", "left": "move_left", "right": "move_right"}.get(data.get("action"), data.get("action"))
    if action not in ACTIONS:
        raise NavigationError("unknown_movement")
    if action.startswith("move_"):
        if "angle_degrees" in data: raise NavigationError("unexpected_angle")
        distance = data.get("distance", step_native)
        if type(distance) not in (int, float) or not math.isfinite(distance) or not 0 < distance <= 2 * step_native:
            raise NavigationError("invalid_distance")
        direction = {"move_left": [-1, 0, 0], "move_right": [1, 0, 0], "move_forward": [0, 0, 1], "move_backward": [0, 0, -1]}[action]
        pose[:3, 3] += pose[:3, :3] @ (np.asarray(direction) * distance)
    else:
        if "distance" in data: raise NavigationError("unexpected_distance")
        degrees = data.get("angle_degrees", 15)
        if type(degrees) not in (int, float) or not math.isfinite(degrees) or not 0 < degrees <= 30:
            raise NavigationError("invalid_angle")
        angle = math.radians(degrees) * (1 if action in {"yaw_right", "pitch_up"} else -1)
        c, sn = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]]) if action.startswith("yaw") else np.array([[1, 0, 0], [0, c, -sn], [0, sn, c]])
        pose[:3, :3] = pose[:3, :3] @ rotation
    return pose.tolist()


def camera_frame(frame):
    if not isinstance(frame, dict) or not {"c2w", "K", "width", "height", "domain"}.issubset(frame):
        raise NavigationError("incomplete_camera")
    try:
        pose = validate_pose(frame["c2w"])
        k = np.asarray(frame["K"], float)
    except (ValueError, TypeError) as exc:
        raise NavigationError("invalid_camera") from exc
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1], rtol=0, atol=1e-8) or not np.allclose([k[0, 1], k[1, 0]], 0, rtol=0, atol=1e-8):
        raise NavigationError("invalid_intrinsics")
    for key in ("width", "height"):
        if type(frame[key]) is not int or not 1 <= frame[key] <= 2048:
            raise NavigationError("invalid_dimensions")
    if frame["domain"] not in {"street", "aerial"}:
        raise NavigationError("invalid_domain")
    # Appearance is fixed by the checkpoint binding. All source/GT/semantic metadata
    # is discarded; this is a virtual camera even when its anchor was a source estimate.
    result = {"c2w": pose.tolist(), "K": k.tolist(), "width": frame["width"], "height": frame["height"], "domain": frame["domain"]}
    result["id"] = "virtual_" + digest(result)[:24]
    return result


def camera_bounds(frames, step_native):
    result = {}
    for domain in ("street", "aerial"):
        centers = [validate_pose(f["c2w"])[:3, 3] for f in frames if f.get("domain") == domain]
        if not centers: continue
        centers = np.asarray(centers); lower = centers.min(0); upper = centers.max(0)
        padding = np.maximum((upper - lower) * .05, 2 * step_native)
        result[domain] = {"lower": (lower - padding).tolist(), "upper": (upper + padding).tolist(),
                          "training_camera_count": len(centers), "padding": padding.tolist()}
    if not result: raise NavigationError("missing_camera_bounds")
    return result


def validate_bounds(frame, binding):
    bounds = binding["camera_center_bounds"].get(frame["domain"])
    if bounds is None: raise NavigationError("unsupported_camera_domain")
    center = np.asarray(frame["c2w"])[:3, 3]
    if np.any(center < bounds["lower"]) or np.any(center > bounds["upper"]):
        raise NavigationError("camera_out_of_bounds", details={"domain": frame["domain"], "bounds": bounds})


def native_step(frames):
    """Use only released training reference cameras; distances are not metres."""
    from scipy.spatial import cKDTree
    centers = np.asarray([validate_pose(frame["c2w"])[:3, 3] for frame in frames])
    centers = np.unique(centers, axis=0)
    if len(centers) < 2: raise NavigationError("insufficient_distinct_training_cameras")
    spacing = cKDTree(centers).query(centers, k=2)[0][:, 1]
    step = float(np.median(spacing))
    if not math.isfinite(step) or step <= 0: raise NavigationError("invalid_training_camera_spacing")
    return step


def resolve_scene_spec(scene):
    """Read-only recovery-root resolution, preserving the original registry."""
    if scene not in STEPS: raise NavigationError("unsupported_scene")
    original = next((r for r in read(REGISTRY)["scenes"] if r.get("id", r.get("scene_id")) == scene), None)
    if original is None: raise NavigationError("missing_scene_registry_entry")
    spec = {k: original[k] for k in ("id", "model_path", "horizon_repo", "iteration", "explicit", "appearance_policy", "appearance_uid", "lod_resolution_scale") if k in original}
    root = Path(spec["model_path"])
    recovery = None
    if not (root / "config.yaml").is_file():
        try: relative = root.relative_to(ORIGINAL_HORIZON)
        except ValueError as exc: raise NavigationError("missing_checkpoint_root") from exc
        receipt = read(RECOVERY_RECEIPT)
        unit = next((row for row in receipt.get("units", [])
                     if row.get("source") == str(ORIGINAL_HORIZON.parent)
                     and row.get("destination") == str(RECOVERED_HORIZON.parent)), None)
        if not unit or unit.get("state") != "complete" or unit.get("returncode") != 0:
            raise NavigationError("unverified_recovery_copy")
        root = RECOVERED_HORIZON / relative
        spec.update(model_path=str(root), horizon_repo=str(RECOVERED_HORIZON))
        recovery = {"original_model_path": original["model_path"],
                    "resolved_model_path": str(root), "receipt_path": str(RECOVERY_RECEIPT),
                    "receipt_sha256": file_sha(RECOVERY_RECEIPT), "completion": unit["finished"]}
    config = root / "config.yaml"
    if file_sha(config) != original["config_sha256"]:
        raise NavigationError("checkpoint_config_hash_mismatch")
    checkpoint = root / "point_cloud" / ("iteration_" + str(spec.get("iteration", 40000)))
    ply = checkpoint / ("point_cloud_explicit.ply" if spec.get("explicit") else "point_cloud.ply")
    expected = next((row.get("size_bytes") for row in original.get("available_checkpoints", [])
                     if row.get("point_cloud_path") == original.get("point_cloud_path")), None)
    if expected is None or ply.stat().st_size != expected:
        raise NavigationError("checkpoint_size_mismatch")
    required = [config, ply, Path(spec["horizon_repo"]) / "scene/lod_model.py"]
    if not spec.get("explicit"):
        required.extend(checkpoint / name for name in ("opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt"))
        if original.get("settings", {}).get("model_config", {}).get("kwargs", {}).get("appearance_dim", 0) > 0:
            required.append(checkpoint / "embedding_appearance.pt")
    if any(not path.is_file() for path in required): raise NavigationError("missing_checkpoint_component")
    return spec, recovery, required


def scene_preflight(scene):
    """Cheap CPU-only prerequisite audit; full checkpoint hashes bind at startup."""
    spec, recovery, required = resolve_scene_spec(scene)
    manifest = REFERENCE_ROOT / scene / "manifest.json"
    data = read(manifest)
    if data.get("metadata", {}).get("heldout_source_images_used") is not False:
        raise NavigationError("training_camera_membership_unverified")
    step = native_step(data["frames"])
    if not math.isclose(step, STEPS[scene], rel_tol=0, abs_tol=1e-12):
        raise NavigationError("training_camera_step_changed")
    return {"scene": scene, "status": "cpu_preflight_passed", "spec": spec,
            "recovery": recovery, "required_files": [str(path) for path in required],
            "reference_manifest_sha256": file_sha(manifest), "step_native": step,
            "step_units": "native_checkpoint_units_not_metres",
            "step_policy": "median_nearest_neighbour_of_unique_training_camera_centers",
            "training_camera_count": len(data["frames"]),
            "camera_center_bounds": camera_bounds(data["frames"], step),
            "gpu_render_smoke_passed": False}


def scene_binding(scene):
    preflight = scene_preflight(scene)
    spec = preflight["spec"]
    model_root = Path(spec["model_path"]); directory = model_root / "point_cloud" / ("iteration_" + str(spec.get("iteration", 40000)))
    checkpoint = directory / ("point_cloud_explicit.ply" if spec.get("explicit") else "point_cloud.ply")
    manifest = REFERENCE_ROOT / scene / "manifest.json"
    bounds = preflight["camera_center_bounds"]
    files = [REGISTRY, Path(__file__), HERE / "navigation_server.py", manifest, ROOT / "qgvi/qgvi/multiscene_renderer.py", model_root / "config.yaml", checkpoint]
    if not spec.get("explicit"):
        files += sorted(directory.glob("*.pt"))
    if preflight["recovery"]:
        files.append(RECOVERY_RECEIPT)
    upstream = Path(spec["horizon_repo"])
    for folder in ("gaussian_renderer", "scene", "utils"):
        files += sorted((upstream / folder).rglob("*.py"))
    records = {}
    for path in files:
        before = path.stat(); content = file_sha(path); after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Renderer input changed while hashing")
        records[str(path)] = {"sha256": content, "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns}
    return {"version": VERSION, "scene": scene, "spec": spec, "files": records,
            "coordinate_system": "native_model_opencv", "step_native": STEPS[scene],
            "step_units": preflight["step_units"], "step_policy": preflight["step_policy"],
            "recovery": preflight["recovery"],
            "rotation_degrees": 15, "minimum_alpha_coverage": MIN_COVERAGE,
            "camera_center_bounds": bounds, "bounds_policy": "training_camera_domain_AABB_plus_max_5pct_extent_2NN_steps",
            "appearance_policy": spec.get("appearance_policy", "fixed"), "appearance_uid": spec.get("appearance_uid", 0)}


def verify_binding(binding):
    for filename, meta in binding["files"].items():
        stat = Path(filename).stat()
        if (stat.st_size, stat.st_mtime_ns) != (meta["size_bytes"], meta["mtime_ns"]):
            raise ValueError("Bound renderer/checkpoint input changed")


class RenderCache:
    def __init__(self, scene, cache_dir, binding, render):
        self.scene, self.root, self.binding, self.render = scene, Path(cache_dir) / scene, binding, render
        self.root.mkdir(parents=True, exist_ok=True)

    def navigate(self, frame, movement):
        source = camera_frame(frame); validate_bounds(source, self.binding)
        target = {**source, "c2w": move(source["c2w"], movement, self.binding["step_native"])}
        return self._render(target, "incremental_virtual_camera", movement=movement, anchor_frame_sha256=digest(source))

    def propose_view(self, frame):
        return self._render(frame, "direct_virtual_camera")

    def _render(self, frame, origin, **context):
        started = time.perf_counter(); verify_binding(self.binding)
        target = camera_frame(frame); validate_bounds(target, self.binding)
        identity = {"binding_sha256": digest(self.binding), "frame": target}
        key = digest(identity); record_path = self.root / (key + ".json"); image_path = self.root / (key + ".png")
        cache_hit = record_path.exists()
        if cache_hit:
            record = read(record_path)
            if record.get("identity") != identity or record.get("image_sha256") != file_sha(image_path):
                raise NavigationError("cache_binding_or_pixels_changed")
            if record.get("record_sha256") != digest({k: v for k, v in record.items() if k != "record_sha256"}):
                raise NavigationError("cache_metadata_changed")
            if record.get("frame") != target or record.get("pose_c2w") != target["c2w"]:
                raise NavigationError("cache_camera_changed")
        else:
            result = self.render(target)
            rgb, depth, alpha = (np.asarray(result[k]) for k in ("rgb", "depth", "alpha"))
            expected = (target["height"], target["width"])
            if rgb.shape != (*expected, 3) or rgb.dtype != np.uint8 or depth.shape != expected or alpha.shape != expected:
                raise NavigationError("invalid_renderer_dimensions")
            if not np.isfinite(depth).all() or not np.isfinite(alpha).all() or np.any(depth < 0) or np.any(alpha < 0) or np.any(alpha > 1):
                raise NavigationError("invalid_renderer_values")
            elapsed = float(result["elapsed_s"])
            if not math.isfinite(elapsed) or elapsed < 0: raise NavigationError("invalid_renderer_time")
            alpha_coverage = float(np.mean(alpha > .5))
            coverage = float(np.mean((alpha > .5) & (depth > 0)))
            valid = coverage >= self.binding["minimum_alpha_coverage"]
            temporary = image_path.with_suffix(".png.tmp"); Image.fromarray(rgb).save(temporary, format="PNG"); temporary.replace(image_path)
            record = {"schema_version": 2, "identity": identity, "image_path": str(image_path.resolve()),
                      "image_sha256": file_sha(image_path), "status": "rendered" if valid else "unusable_render",
                      "valid": valid, "coverage": coverage, "alpha_coverage": alpha_coverage, "compute_seconds": elapsed,
                      "pose_c2w": target["c2w"], "frame": target, "scene": self.scene,
                      "coordinate_units": "model", "appearance_policy": self.binding["appearance_policy"],
                      "free_camera_no_collision_check": True, "physical_motion_or_traversability_claim": False}
            record["record_sha256"] = digest(record)
            write(record_path, record)
        output = {**{k: v for k, v in record.items() if k not in {"identity", "record_sha256"}}, "cache_hit": cache_hit,
                  "elapsed_seconds": time.perf_counter() - started, "render_seconds_this_request": 0.0 if cache_hit else record["compute_seconds"],
                  "pose_origin": origin, "source_pose_estimate": False, "binding_sha256": digest(self.binding), **context}
        if not record["valid"]:
            raise NavigationError("unusable_render", "Insufficient supported pixels in the proposed camera render", details=output)
        return output


_PROCESS_GPU_HANDLES = []


@contextmanager
def gpu_reservation(*, process_lifetime=False):
    handles = []
    try:
        for path in LOCKS:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+"); handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        if process_lifetime:
            # Keep OS reservations until this resident process actually exits,
            # including exceptions and CUDA teardown. They are never reused.
            _PROCESS_GPU_HANDLES.extend(handles)
        else:
            for handle in reversed(handles):
                fcntl.flock(handle, fcntl.LOCK_UN); handle.close()


def worker(scene, cache_dir):
    sys.dont_write_bytecode = True  # Recovery assets remain read-only.
    # Native render libraries can write directly to fd1; reserve a duplicate
    # pipe exclusively for JSON RPC, and send every other stdout write to log.
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    def emit(value): channel.write(canonical(value) + "\n"); channel.flush()
    with redirect_stdout(sys.stderr), gpu_reservation(process_lifetime=True):
        started = time.perf_counter(); binding = scene_binding(scene)
        sys.path.insert(0, str(ROOT / "qgvi"))
        import torch
        from qgvi.multiscene_renderer import load_scene_model, render_scene_frame
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        torch.set_num_threads(2)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.9, 80 * 1024**3 / total), device=0)
        model = cache = None; caches = {}
        try:
            model = load_scene_model(binding["spec"])
            readiness = {"status": "ready", "scene": scene, "worker_pid": os.getpid(),
                         "binding_sha256": digest(binding), "startup_seconds": time.perf_counter() - started,
                         "model_load_seconds": model.load_elapsed_s,
                         "peak_allocated_bytes": model.load_peak_allocated_bytes,
                         "peak_reserved_bytes": model.load_peak_reserved_bytes,
                         "step_native": STEPS[scene], "camera_center_bounds": binding["camera_center_bounds"], "gpu_locks_held": [str(p) for p in LOCKS]}
            write(Path(cache_dir) / scene / "binding.json", binding)
            write(Path(cache_dir) / scene / "worker_ready.json", readiness); emit(readiness)
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    if request.get("command") == "close": emit({"status": "closed"}); return
                    namespace = validate_cache_namespace(request.get("cache_namespace", "default"))
                    if namespace not in caches:
                        caches[namespace] = RenderCache(scene, Path(cache_dir) / "namespaces" / namespace,
                                                        binding, lambda frame: render_scene_frame(model, frame))
                    cache = caches[namespace]
                    if request.get("command") == "navigate": result = cache.navigate(request["frame"], request["movement"])
                    elif request.get("command") == "render": result = cache.propose_view(request["frame"])
                    else: raise NavigationError("unknown_command")
                    emit({"status": "ok", "result": {**result, "cache_namespace": namespace}})
                except Exception as exc:
                    emit({"status": "error", "error_type": getattr(exc, "kind", type(exc).__name__), "message": str(exc)[:300], "details": getattr(exc, "details", {})})
        finally:
            # Destroy the render closure and model before releasing GPU memory.
            cache = None; caches.clear(); model = None; gc.collect()
            torch.cuda.synchronize(); torch.cuda.empty_cache()



class NavigationSession:
    def __init__(self, scene, cache_dir, *, startup_timeout=600, request_timeout=180, executable=sys.executable, cache_namespace="default"):
        if scene not in STEPS: raise ValueError("Unsupported scene")
        self.cache_namespace = validate_cache_namespace(cache_namespace)
        self.scene, self.cache_dir, self.request_timeout = scene, Path(cache_dir).resolve(), request_timeout
        self.cache_dir.mkdir(parents=True, exist_ok=True); self.mutex = threading.Lock(); self.process = None
        self.log = (self.cache_dir / (scene + "_worker.log")).open("a")
        self.process = subprocess.Popen([executable, "-B", str(Path(__file__).resolve()), "worker", "--scene", scene, "--cache-dir", str(self.cache_dir)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
            cwd=str(HERE), start_new_session=True, bufsize=1)
        try:
            self.readiness = self._receive(startup_timeout)
            if self.readiness.get("status") != "ready": raise RuntimeError("Navigation worker failed to start")
        except BaseException:
            self.close(); raise

    def _receive(self, timeout):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout): raise TimeoutError("Navigation worker timeout")
            line = self.process.stdout.readline(4 * 1024 * 1024)
        if not line: raise RuntimeError("Navigation worker exited; inspect worker log")
        return json.loads(line)

    def navigate(self, frame, movement, *, cache_namespace=None):
        normalized = camera_frame(frame)
        move(normalized["c2w"], movement, STEPS[self.scene])
        return self._request({"command": "navigate", "frame": normalized, "movement": movement,
                              "cache_namespace": validate_cache_namespace(cache_namespace if cache_namespace is not None else self.cache_namespace)})

    def propose_view(self, frame, *, cache_namespace=None):
        return self._request({"command": "render", "frame": camera_frame(frame),
                              "cache_namespace": validate_cache_namespace(cache_namespace if cache_namespace is not None else self.cache_namespace)})

    def for_namespace(self, namespace):
        """Share the resident model, never rendered pixels from another arm."""
        return NavigationNamespace(self, validate_cache_namespace(namespace))

    def _request(self, request):
        with self.mutex:
            try:
                self.process.stdin.write(canonical(request) + "\n"); self.process.stdin.flush()
                response = self._receive(self.request_timeout)
            except BaseException:
                self.close(); raise  # Never confuse a late reply with the next request.
            if response.get("status") != "ok":
                raise NavigationError(response.get("error_type", "worker_error"), response.get("message"), response.get("details"))
            return response["result"]

    def close(self):
        process = self.process
        if process is not None:
            if process.poll() is None:
                try:
                    process.stdin.write('{"command":"close"}\n'); process.stdin.flush(); process.wait(timeout=5)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    try: os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
            process.stdin.close(); process.stdout.close(); self.process = None
        if not self.log.closed: self.log.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


class NavigationNamespace:
    def __init__(self, session, namespace):
        self.session, self.cache_namespace = session, validate_cache_namespace(namespace)

    def navigate(self, frame, movement):
        return self.session.navigate(frame, movement, cache_namespace=self.cache_namespace)

    def propose_view(self, frame):
        return self.session.propose_view(frame, cache_namespace=self.cache_namespace)


class NavigationClient:
    """Thin localhost client; the host supplies the query's fixed K and resolution."""
    def __init__(self, scene, *, base_url=DEFAULT_URL, timeout=180, cache_namespace="default"):
        if scene not in STEPS: raise NavigationError("unsupported_scene")
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.port is None:
            raise NavigationError("server_must_be_local")
        self.scene, self.base_url, self.timeout = scene, base_url.rstrip("/"), timeout
        self.cache_namespace = validate_cache_namespace(cache_namespace)

    def _request(self, endpoint, data):
        request = urllib.request.Request(self.base_url + endpoint, canonical({"scene": self.scene, "cache_namespace": self.cache_namespace, **data}).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response: result = json.load(response)
        except urllib.error.HTTPError as exc:
            try: result = json.load(exc)
            except Exception: raise NavigationError("server_http_error") from exc
        except (OSError, TimeoutError) as exc:
            raise NavigationError("server_unavailable") from exc
        if not result.get("ok"):
            raise NavigationError(result.get("kind", "server_error"), result.get("message"), result.get("details"))
        return {key: value for key, value in result.items() if key != "ok"}

    def navigate(self, frame, movement):
        normalized = camera_frame(frame); move(normalized["c2w"], movement, STEPS[self.scene])
        return self._request("/navigate", {"frame": normalized, "movement": movement})

    def propose_view(self, frame):
        return self._request("/render", {"frame": camera_frame(frame)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=["worker"])
    parser.add_argument("--scene", required=True, choices=tuple(STEPS)); parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args(); worker(args.scene, args.cache_dir)
