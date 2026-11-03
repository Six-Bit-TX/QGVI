"""One entry point for preserved QGVI algorithms and the local demonstration."""
from __future__ import annotations

import argparse
import functools
import http.server
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import sys
import webbrowser

from .core import ROOT, TextIndex, LiveLocalizer, LiveNavigation, read, sha, solve_correspondences, write


def query_file(path):
    path = Path(path).resolve()
    query = read(path)
    for key in ("image_path", "source_image"):
        if query.get(key):
            p = Path(query[key])
            query[key] = str(p if p.is_absolute() else path.parent / p)
    return query


def doctor():
    dependencies = {}
    for name in ("numpy", "pycolmap", "Pillow", "h5py", "torch", "torchvision", "gsplat"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    import os
    return {"python": sys.version.split()[0], "dependencies": dependencies,
            "device_policy": "Unchanged original requirements; no CPU/GPU environment override",
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
            "LoGS_installed": (ROOT / "third_party/LoGS/hloc").is_dir(),
            "HorizonGS_installed": (ROOT / "third_party/HorizonGS/scene/lod_model.py").is_file()}


class LazyNavigation:
    def __init__(self, scene, output, repo):
        self.args = scene, output, repo
        self.backend = None

    def get(self):
        if self.backend is None:
            self.backend = LiveNavigation(*self.args)
        return self.backend

    def propose_view(self, frame):
        return self.get().propose_view(frame)

    def navigate(self, frame, movement):
        return self.get().navigate(frame, movement)


class AgentRetrieval:
    """Combine original text retrieval with fresh image initialization.

    Image candidate pictures must be supplied in the portable index via
    reference_name. We never substitute query GT or a cached estimated pose.
    """
    def __init__(self, index, query, scene, output, upstream, torch_cache):
        self.text = TextIndex(index, query.get("scene"))
        self.index = read(index)
        self.query = query
        self.args = scene, output, upstream, torch_cache
        self.localizer = None
        self.base_result = None
        if query.get("source_image"):
            self.text.configure_source(query["host_bundle_id"], sha(query["source_image"]),
                                       query.get("excluded_reference_names", []))

    def bind_source(self, bundle_id):
        if self.query.get("source_image") and bundle_id != self.query["host_bundle_id"]:
            raise ValueError("Source binding differs")
        self.text.bind_source(bundle_id)

    def search_text(self, query, limit=6, bundle_id=None):
        return self.text.search_text(query, limit, bundle_id)

    def prepare(self):
        if self.base_result is None:
            self.localizer = LiveLocalizer(*self.args)
            q = {k: self.query[k] for k in ("id", "scene", "K", "width", "height")}
            q["image_path"] = self.query["source_image"]
            q["excluded_reference_names"] = self.query.get("excluded_reference_names", [])
            self.base_result = self.localizer.localize(q)
        return self.base_result

    def estimate_pose(self, bundle_id, region=None):
        self.bind_source(bundle_id)
        base = self.prepare()
        if region is None:
            return base
        last = self.localizer.last
        return solve_correspondences(last["query"], last["receipt"]["paths"]["correspondences"], region)

    def search_image(self, bundle_id, limit=6):
        self.bind_source(bundle_id)
        self.prepare()
        rows = self.index.get("views", [])
        by_name = {r["reference_name"]: r.get("view_id", r.get("id")) for r in rows
                   if r.get("reference_name") and r.get("scene") == self.query["scene"]}
        pairs = Path(self.localizer.last["receipt"]["paths"]["pairs"]).read_text().splitlines()
        names = [line.split()[1] for line in pairs]
        missing = [name for name in names[:limit] if name not in by_name]
        if missing:
            raise ValueError("Image-search gallery pixels are not bundled: provide an index with reference_name for the retrieved map images")
        candidates = [self.text.resolve(by_name[name]) for name in names[:limit]]
        return {"candidates": candidates, "backend": "fresh_NetVLAD_top10",
                "score_meaning": "Ordered NetVLAD references; no probability or source pose claim"}


def serve(args):
    indices = {}
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/api/text":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 32768:
                    raise ValueError("Request too large")
                data = json.loads(self.rfile.read(size))
                scene = data.get("scene", "block_a")
                if scene not in indices:
                    indices[scene] = TextIndex(ROOT / "examples/index.json", scene)
                result = indices[scene].search_text(data["query"], limit=6)
                for row in result["candidates"]:
                    row["image_path"] = "/" + str(Path(row["image_path"]).relative_to(ROOT / "demo"))
                encoded = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except (ValueError, KeyError, TypeError):
                self.send_error(400, "Invalid text query")

    host = args.bind
    server = http.server.ThreadingHTTPServer((host, args.port), functools.partial(Handler, directory=str(ROOT / "demo")))
    url = f"http://{host}:{server.server_port}/"
    print(f"QGVI demo: {url}", flush=True)
    print("Text search is live. RGB/agent web panels show recorded examples; CLI commands run the original algorithms.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    text = sub.add_parser("text")
    text.add_argument("query")
    text.add_argument("--index", default=str(ROOT / "examples/index.json"))
    text.add_argument("--scene", default="block_a")
    text.add_argument("--limit", type=int, default=5)
    pnp = sub.add_parser("pnp")
    pnp.add_argument("--query", default=str(ROOT / "examples/sf-query.json"))
    pnp.add_argument("--correspondences", default=str(ROOT / "examples/sf-correspondences.npz"))
    localize = sub.add_parser("localize")
    localize.add_argument("--query", required=True)
    localize.add_argument("--scene", required=True)
    localize.add_argument("--output", default="outputs/localize")
    localize.add_argument("--upstream", default=str(ROOT / "third_party/LoGS"))
    localize.add_argument("--torch-cache", default=str(ROOT / ".cache/torch"))
    render = sub.add_parser("render")
    render.add_argument("--scene", required=True)
    render.add_argument("--frame", default=str(ROOT / "examples/render-frame.json"))
    render.add_argument("--output", default="outputs/render")
    render.add_argument("--horizon-repo", default=str(ROOT / "third_party/HorizonGS"))
    agent = sub.add_parser("agent")
    agent.add_argument("--query", required=True)
    agent.add_argument("--scene", required=True)
    agent.add_argument("--index", default=str(ROOT / "examples/index.json"))
    agent.add_argument("--output", default="outputs/agent")
    agent.add_argument("--provider-command", required=True, help="JSON stdin/stdout program; parsed into argv, never run through a shell")
    agent.add_argument("--upstream", default=str(ROOT / "third_party/LoGS"))
    agent.add_argument("--torch-cache", default=str(ROOT / ".cache/torch"))
    agent.add_argument("--horizon-repo", default=str(ROOT / "third_party/HorizonGS"))
    demo = sub.add_parser("demo")
    demo.add_argument("--port", type=int, default=7860)
    demo.add_argument("--bind", default="127.0.0.1")
    demo.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        result = doctor()
    elif args.command == "text":
        result = TextIndex(args.index, args.scene).search_text(args.query, args.limit)
    elif args.command == "pnp":
        result = solve_correspondences(query_file(args.query), args.correspondences)
    elif args.command == "localize":
        result = LiveLocalizer(args.scene, args.output, args.upstream, args.torch_cache).localize(query_file(args.query))
        write(Path(args.output) / "public_result.json", result)
    elif args.command == "render":
        result = LiveNavigation(args.scene, args.output, args.horizon_repo).propose_view(read(args.frame))
    elif args.command == "agent":
        from .agent import run_agent
        query = query_file(args.query)
        if query.get("source_image"):
            query["source_image_sha256"] = sha(query["source_image"])
            query.setdefault("host_bundle_id", query["id"])
        retrieval = AgentRetrieval(args.index, query, args.scene, Path(args.output) / "rgb", args.upstream, args.torch_cache)
        navigation = LazyNavigation(args.scene, Path(args.output) / "renders", args.horizon_repo)
        result = run_agent(query, args.output, retrieval, navigation, shlex.split(args.provider_command))
    else:
        serve(args)
        return
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
