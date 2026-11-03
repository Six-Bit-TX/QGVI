"""Nine-scene, image-bound BM25/NetVLAD retrieval and genuine cached PnP.

Original and novel-grid caption banks are searched together. MiniLM is not used
by this backend. Native OpenCV c2w and intrinsics belong to the returned pixels.
Scores order candidates; neither retrieval score establishes correctness.

bundle_id is a HOST-ONLY handle, never included in public responses. Pass it to
every query-dependent call, or bind_source() once on a per-query instance.
This is the existing training-source development pilot, not a held-out test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOVEL = ROOT / 'experiments/novel_view_contribution_20261024_v1/runs/full_v1'
OLD = ROOT / 'experiments/agentic_locator_comparison_20261025_v1'
GEOMETRY = HERE / 'geometry' / 'target250'
RGB = ROOT / 'experiments/cvpr2027_rgb'
FASTGS = Path('/home/xiongti1/miniconda3/envs/fastgs/bin/python')
VERSION = 'nine_scene_image_bound_bm25_netvlad_pnp_v1'
SCENES = ('block_a', 'block_small', 'citysample', 'colosseum', 'elvenruin', 'nyc', 'park', 'road', 'sf')

sys.path.insert(0, str(ROOT / 'qgvi-rag'))
from caption_rag.pipeline import collect_chunks, digest as caption_digest
from caption_rag.retrieval import _tokens


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def checked_sha(path, expected):
    if sha(path) != expected:
        raise ValueError('Bound retrieval asset changed')


def native_frame(row):
    frame = {k: copy.deepcopy(row[k]) for k in ('K', 'c2w', 'width', 'height', 'domain')}
    pose = np.asarray(frame['c2w'], dtype=float)
    intrinsics = np.asarray(frame['K'], dtype=float)
    if (pose.shape != (4, 4) or intrinsics.shape != (3, 3)
        or not np.isfinite(pose).all() or not np.isfinite(intrinsics).all()
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-5)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=2e-5)
        or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0
        or not np.allclose(intrinsics[2], [0, 0, 1])):
        raise ValueError('Invalid native OpenCV camera')
    if any(type(frame[k]) is not int or frame[k] <= 0 for k in ('width', 'height')):
        raise ValueError('Invalid camera image dimensions')
    if frame['domain'] not in ('street', 'aerial'):
        raise ValueError('Unknown camera domain')
    return frame


class BM25:
    """Exact existing caption adapter BM25 recipe; no embedding/model load."""
    def __init__(self, captions):
        self.ids = sorted(captions)
        self.counts = {v: Counter(_tokens(captions[v])) for v in self.ids}
        self.lengths = {v: sum(c.values()) for v, c in self.counts.items()}
        self.average = sum(self.lengths.values()) / max(1, len(self.ids)) or 1.0
        self.df = Counter(term for counts in self.counts.values() for term in counts)

    def scores(self, query):
        terms = sorted(set(_tokens(query)))
        total = len(self.ids)
        scores = {}
        for view in self.ids:
            counts = self.counts[view]
            norm = 1.2 * (.25 + .75 * self.lengths[view] / self.average)
            score = 0.0
            for term in terms:
                count = counts.get(term, 0)
                if count:
                    frequency = self.df[term]
                    score += math.log1p((total - frequency + .5) / (frequency + .5)) * count * 2.2 / (count + norm)
            scores[view] = score
        return scores


class RetrievalBackend:
    def __init__(self, scene, workdir, geometry_dir=None):
        started = time.perf_counter()
        if scene not in SCENES:
            raise ValueError('Unknown qualified nine-scene map')
        self.scene = scene
        self.geometry_dir = Path(geometry_dir or GEOMETRY).resolve()
        self.workdir = Path(workdir).resolve()
        self.cache = self.workdir / 'retrieval_cache' / scene
        self.cache.mkdir(parents=True, exist_ok=True)
        self.bound_source = None
        self._load_assets()
        self.bm25 = BM25({v: r['caption'] for v, r in self.views.items()})
        self.init_seconds = time.perf_counter() - started

    def _load_assets(self):
        """Validate existing captions/features; build only a lightweight CPU adapter."""
        plan = read(NOVEL / 'plan.json')
        gp_path = self.geometry_dir / 'plan.json'
        geometry_plan = read(gp_path)
        completed = read(self.geometry_dir / 'COMPLETED.json')
        if completed.get('status') != 'complete' or completed['plan_sha256'] != sha(gp_path):
            raise ValueError('Geometry completion binding invalid')
        checked_sha(OLD / 'geometry_adapter.py', geometry_plan['source_sha256'])
        self.geometry_plan = geometry_plan
        self.geometry_completed = completed
        self.geometry_plan_sha = sha(gp_path)
        self.queries = {q['id']: q for q in geometry_plan['queries'] if q['scene'] == self.scene}
        scene_geometry = geometry_plan['scenes'][self.scene]
        training_path = Path(scene_geometry['training_protocol'])
        checked_sha(training_path, scene_geometry['training_protocol_sha256'])
        training = read(training_path)['records']
        self.training = {r['name']: r for r in training}
        if set(self.training) != set(scene_geometry['reference_names']):
            raise ValueError('Reference inventory differs from geometry')

        selected = [s for s in plan['caption_sources'] if s['scene'] == self.scene
                    and s['kind'] in ('original', 'current_grid')]
        if {s['kind'] for s in selected} != {'original', 'current_grid'} or len(selected) != 2:
            raise ValueError('Expected original plus current-grid caption sources')
        views = {}
        bindings = {}
        training_to_view = {}
        for source in selected:
            path = Path(source['manifest'])
            checked_sha(path, source['manifest_sha256'])
            frames = read(path)['frames']
            directory = Path(source['caption_dir'])
            caption_run = read(directory / 'run.json')
            if caption_run['binding']['manifest_sha256'] != source['manifest_sha256']:
                raise ValueError('Caption manifest binding changed')
            if caption_digest(caption_run['frames']) != caption_run['binding']['frames_sha256']:
                raise ValueError('Caption frame binding changed')
            caption_frames = {f['id']: f for f in caption_run['frames']}
            if set(caption_frames) != {f['id'] for f in frames}:
                raise ValueError('Caption inventory differs from index')
            for frame in frames:
                if frame.get('scene_id') != self.scene or frame.get('coordinate_units') != 'model':
                    raise ValueError('Caption bank coordinate convention mismatch')
                for key in ('image_path', 'image_sha256', 'c2w', 'K', 'width', 'height', 'domain'):
                    if caption_frames[frame['id']].get(key) != frame.get(key):
                        raise ValueError('Caption frame metadata differs from manifest')
            chunks = collect_chunks(directory, caption_run['frames'], caption_digest(caption_run['binding']))
            cache_meta_path = NOVEL / 'embedding_cache' / source['source_id'] / 'cache.json'
            cache_meta = read(cache_meta_path)
            if (cache_meta['binding']['source_manifest_sha256'] != source['manifest_sha256']
                or cache_meta['binding']['chunks_sha256'] != caption_digest(chunks)):
                raise ValueError('Caption content differs from completed source cache')
            chunks_by_id = {c['id']: c for c in chunks}
            bindings[source['source_id']] = {'manifest_sha256': sha(path),
                'caption_run_sha256': sha(directory / 'run.json'), 'chunks_sha256': caption_digest(chunks),
                'source_cache_sha256': sha(cache_meta_path)}
            for row in frames:
                vid = 'v_' + digest({'scene': self.scene, 'source_view': row['id'],
                                      'image_sha256': row['image_sha256']})[:24]
                if vid in views:
                    raise ValueError('Opaque view ID collision')
                views[vid] = {'view_id': vid, 'frame': native_frame(row), 'image_path': row['image_path'],
                    'image_sha256': row['image_sha256'], 'caption': chunks_by_id[row['id']]['text'],
                    'kind': source['kind'], 'training_id': row.get('training_template_id'),
                    'source_image_sha256': row.get('source_image_sha256')}
                if source['kind'] == 'original':
                    tid = row['training_template_id']
                    if tid in training_to_view:
                        raise ValueError('Duplicate original training identity')
                    training_to_view[tid] = vid
        self.views = views
        self.reference_names = list(scene_geometry['reference_names'])
        self.reference_views = []
        for name in self.reference_names:
            record = self.training[name]
            vid = training_to_view.get(record['id'])
            if vid is None:
                raise ValueError('Visual reference has no indexed camera')
            row = self.views[vid]
            if (not np.allclose(row['frame']['c2w'], record['c2w'], atol=1e-6, rtol=0)
                or row['source_image_sha256'] != record['sha256']):
                raise ValueError('Caption camera and native geometric map differ')
            self.reference_views.append(vid)
        ref_path = Path(scene_geometry['reference_global'])
        query_path = self.geometry_dir / self.scene / 'query_netvlad.h5'
        checked_sha(ref_path, scene_geometry['reference_global_sha256'])
        checked_sha(query_path, completed['outputs_sha256'][str(query_path)])
        with h5py.File(ref_path, 'r') as refs:
            self.reference_vectors = np.stack([np.asarray(refs[n]['global_descriptor'], np.float32).reshape(-1)
                                              for n in self.reference_names])
        with h5py.File(query_path, 'r') as queries:
            self.query_vectors = {ident: np.asarray(queries[q['name']]['global_descriptor'], np.float32).reshape(-1)
                                  for ident, q in self.queries.items()}
        dimension = self.reference_vectors.shape[1]
        if (not np.isfinite(self.reference_vectors).all()
            or any(v.shape != (dimension,) or not np.isfinite(v).all() for v in self.query_vectors.values())):
            raise ValueError('Cached NetVLAD vector invalid')
        self.binding = {'version': VERSION, 'scene': self.scene, 'caption_sources': bindings,
            'geometry_plan_sha256': self.geometry_plan_sha,
            'training_protocol_sha256': sha(training_path), 'reference_vectors_sha256': sha(ref_path),
            'query_vectors_sha256': sha(query_path), 'source_binding': 'image_sha256_no_query_pose',
            'tokenizer_sha256': sha(ROOT / 'qgvi-rag/caption_rag/retrieval.py'),
            'source_sha256': sha(__file__)}
        self.binding_sha = digest(self.binding)
        self._prepare_exclusions()
        catalog = {'binding': self.binding, 'view_count': len(self.views),
                   'visual_reference_count': len(self.reference_names), 'views_sha256': digest(self.views)}
        with lock(self.cache / 'prepare.lock'):
            receipt = self.cache / 'catalog.json'
            if receipt.exists():
                if read(receipt) != catalog:
                    raise ValueError('Frozen retrieval adapter cache changed; use a new workdir')
            else:
                write(receipt, catalog)

    def _prepare_exclusions(self):
        """Exclude query-identical indexed pixels and duplicated reference cameras.

        Query poses are never accepted by prepare_geometry. Camera deduplication
        uses only the camera of a reference independently identified by exact
        pixels; it is a private exclusion operation, never a pose estimate.
        """
        self.excluded = {}
        for handle, query in self.queries.items():
            checked_sha(query['image_path'], query['image_sha256'])
            reference_excluded = set(query['excluded_reference_names'])
            if not reference_excluded <= set(self.reference_names):
                raise ValueError('Unknown self-exclusion reference')
            excluded_poses = [np.asarray(self.training[n]['c2w']) for n in reference_excluded]
            blocked = {vid for name, vid in zip(self.reference_names, self.reference_views)
                       if name in reference_excluded}
            for vid, row in self.views.items():
                if (row['image_sha256'] == query['image_sha256']
                    or any(np.allclose(np.asarray(row['frame']['c2w']), pose, atol=1e-6, rtol=0)
                           for pose in excluded_poses)):
                    blocked.add(vid)
            self.excluded[handle] = blocked

    def for_namespace(self, namespace):
        """Reuse immutable banks while isolating every arm's pose-result cache.

        The shallow clone shares captions, descriptors, frame records and frozen
        exclusions. Only source binding, the small cache path, and workdir differ.
        Source features and initial PnP are prepared identically outside timed
        queries; adapter-result and ROI solve caches must not cross speed arms.
        """
        if (not isinstance(namespace, str) or not 1 <= len(namespace) <= 120
            or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in namespace)):
            raise ValueError('Invalid pose cache namespace')
        scoped = copy.copy(self)
        scoped.workdir = self.workdir / 'namespaces' / namespace
        scoped.cache = scoped.workdir / 'retrieval_cache' / self.scene
        scoped.cache.mkdir(parents=True, exist_ok=True)
        scoped.bound_source = None
        return scoped

    def bind_source(self, bundle_id):
        """Optional HOST-only binding for a backend dedicated to one query."""
        if bundle_id is not None and bundle_id not in self.queries:
            raise ValueError('Unknown source handle for this scene')
        self.bound_source = bundle_id
        return self

    def _query(self, bundle_id=None):
        if self.bound_source is not None:
            if bundle_id is not None and bundle_id != self.bound_source:
                raise ValueError('Backend is bound to another source')
            bundle_id = self.bound_source
        if bundle_id is not None:
            if bundle_id not in self.queries:
                raise ValueError('Unknown source handle for this scene')
            q = self.queries[bundle_id]
            checked_sha(q['image_path'], q['image_sha256'])
        return bundle_id

    @staticmethod
    def _limit(limit):
        if type(limit) is not int or not 1 <= limit <= 30:
            raise ValueError('limit must be an integer from 1 to 30')
        return limit

    def resolve(self, view_id, bundle_id=None):
        bundle = self._query(bundle_id)
        if view_id not in self.views or view_id in self.excluded.get(bundle, set()):
            raise ValueError('Unknown or excluded view')
        row = self.views[view_id]
        checked_sha(row['image_path'], row['image_sha256'])
        with Image.open(row['image_path']) as im:
            if im.size != (row['frame']['width'], row['frame']['height']):
                raise ValueError('Indexed pixels do not match camera intrinsics')
        return {k: copy.deepcopy(row[k]) for k in ('view_id', 'frame', 'image_path', 'image_sha256', 'caption')}

    def _select(self, ranked, limit, bundle):
        candidates = []
        pixels, cameras = set(), set()
        for vid, score in ranked:
            if vid in self.excluded.get(bundle, set()):
                continue
            row = self.views[vid]
            camera = digest(np.round(row['frame']['c2w'], 6).tolist())
            if row['image_sha256'] in pixels or camera in cameras:
                continue
            result = self.resolve(vid, bundle)
            result['score'] = float(score)
            candidates.append(result)
            pixels.add(row['image_sha256'])
            cameras.add(camera)
            if len(candidates) == limit:
                break
        return candidates

    def search_text(self, query, limit=6, bundle_id=None):
        started = time.perf_counter()
        self._limit(limit)
        if not isinstance(query, str) or len(query) > 8000:
            raise ValueError('Text query must be a string of at most 8000 characters')
        bundle = self._query(bundle_id)
        scores = self.bm25.scores(query)
        ranked = sorted(((v, s) for v, s in scores.items() if s > 0), key=lambda r: (-r[1], r[0]))
        candidates = self._select(ranked, limit, bundle)
        return {'candidates': candidates, 'timing': {'total_seconds': time.perf_counter() - started},
            'backend': 'BM25_original_plus_Gaussian_grid',
            'score_meaning': 'Higher means stronger caption-word overlap. Not a probability, geometric evidence, or proof of absence; zero-overlap views are omitted.',
            'index_view_count': len(self.views), 'coordinate_convention': 'native_model_OpenCV_c2w',
            'self_exclusion_applied': bundle is not None, 'evidence_id': 'e_' + digest({'binding': self.binding_sha, 'text': query, 'source': bundle})[:24]}

    def search_image(self, bundle_id, limit=6):
        started = time.perf_counter()
        self._limit(limit)
        bundle = self._query(bundle_id)
        if bundle is None:
            raise ValueError('An existing cached source descriptor is required')
        scores = self.reference_vectors @ self.query_vectors[bundle]
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.reference_names[i]))
        candidates = self._select(((self.reference_views[i], scores[i]) for i in order), limit, bundle)
        return {'candidates': candidates, 'timing': {'total_seconds': time.perf_counter() - started},
            'backend': 'cached_NetVLAD_training_reference_bank',
            'score_meaning': 'Higher descriptor inner product indicates greater visual similarity. It is not calibrated confidence and does not estimate the SOURCE pose.',
            'index_view_count': len(self.reference_names), 'coordinate_convention': 'native_model_OpenCV_c2w',
            'self_exclusion_applied': True, 'evidence_id': 'e_' + digest({'binding': self.binding_sha, 'image': self.queries[bundle]['image_sha256']})[:24],
            'timing_scope': 'cached SOURCE descriptor ranking and returned-pixel checks; excludes historical NetVLAD extraction'}

    def estimate_pose(self, bundle_id, region=None):
        started = time.perf_counter()
        bundle = self._query(bundle_id)
        if bundle is None:
            raise ValueError('A host source handle is required')
        checked_sha(self.geometry_dir / 'plan.json', self.geometry_plan_sha)
        checked_sha(OLD / 'geometry_adapter.py', self.geometry_plan['source_sha256'])
        checked_sha(ROOT / 'experiments/cvpr2027_rgb/logs_selective_v2/confidence.py',
                    self.geometry_plan['confidence_sha256'])
        for path in (self.geometry_dir / 'results' / (bundle + '.json'),
                     self.geometry_dir / 'correspondences' / (bundle + '.npz')):
            checked_sha(path, self.geometry_completed['outputs_sha256'][str(path)])
        if region is not None:
            r = np.asarray(region, float)
            if r.shape != (4,) or not np.isfinite(r).all() or not (0 <= r[0] < r[2] <= 1 and 0 <= r[1] < r[3] <= 1):
                raise ValueError('Region must be a normalized nonempty keep-box')
            region = r.tolist()
        key = digest({'binding': self.binding_sha, 'image': self.queries[bundle]['image_sha256'], 'region': region})
        path = self.cache / 'poses' / (key + '.json')
        with lock(self.workdir / 'retrieval_cache' / 'geometry_cpu.lock'):
            if path.exists():
                cached = read(path)
                if cached.get('binding') != key or digest(cached['result']) != cached.get('result_sha256'):
                    raise ValueError('Pose cache binding changed')
                result = cached['result']
                hit = True
            else:
                command = [str(FASTGS), str(Path(__file__).resolve()), '--pose-worker', '--geometry-dir', str(self.geometry_dir), '--bundle-id', bundle]
                if region is not None:
                    command += ['--region', json.dumps(region, allow_nan=False)]
                response = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60, check=False)
                if response.returncode:
                    raise RuntimeError('Cached geometric pose subprocess failed')
                result = json.loads(response.stdout)
                write(path, {'binding': key, 'result': result, 'result_sha256': digest(result)})
                hit = False
        result = copy.deepcopy(result)
        result['hypothesis_id'] = 'pose_' + key[:24]
        recorded_solver_seconds = result.get('compute_seconds', 0)
        solved_now = region is not None and not hit
        result['timing'] = {'total_seconds': time.perf_counter() - started,
                            'solver_seconds': recorded_solver_seconds if solved_now else 0,
                            'recorded_original_solver_seconds': recorded_solver_seconds}
        result['compute_seconds'] = recorded_solver_seconds if solved_now else 0
        result['adapter_cache_hit'] = hit
        result['cache_hit'] = hit or bool(result.get('cache_hit'))
        result['solver_executed_this_call'] = solved_now
        result['backend'] = 'cached_SuperPoint_SuperGlue_SfM_PnP'
        result['score_meaning'] = 'Geometric support is evidence for the SOURCE pose, not semantic target satisfaction; acceptance is not calibrated for adaptive attempts.'
        result['evidence_id'] = 'e_' + key[:24]
        return result


def pose_worker(bundle_id, region, geometry_dir=None):
    """Read qualified per-source evidence, then use the unchanged PnP solver.

    Validate the requested source, not every cohort image on every tool call.
    Full-cohort pixels were already verified during geometry preparation.
    """
    started = time.perf_counter()
    geometry_dir = Path(geometry_dir or GEOMETRY).resolve()
    geometry = geometry_module()
    plan_path = geometry_dir / 'plan.json'; plan = read(plan_path); plan_sha = sha(plan_path)
    checked_sha(OLD / 'geometry_adapter.py', plan['source_sha256'])
    checked_sha(geometry.CONFIDENCE, plan['confidence_sha256'])
    selected = [q for q in plan['queries'] if q['id'] == bundle_id]
    if len(selected) != 1: raise ValueError('Unknown host source handle')
    query = selected[0]; checked_sha(query['image_path'], query['image_sha256'])
    completed = read(geometry_dir / 'COMPLETED.json')
    if completed.get('status') != 'complete' or completed['plan_sha256'] != plan_sha:
        raise ValueError('Geometry completion binding changed')
    base_path = geometry_dir / 'results' / (bundle_id + '.json')
    corr_path = geometry_dir / 'correspondences' / (bundle_id + '.npz')
    for path in (base_path, corr_path):
        checked_sha(path, completed['outputs_sha256'][str(path)])
    base = read(base_path)
    if (base['plan_sha256'] != plan_sha or base['correspondences_sha256'] != sha(corr_path)
            or base['query_image_sha256'] != query['image_sha256']):
        raise ValueError('Source geometry cache binding changed')
    if region is None:
        return geometry.public_result(base, elapsed=time.perf_counter() - started, cache_hit=True)
    corr = geometry.load_corr(corr_path)
    result = geometry.solve(query, corr, region)
    result.update(reference_count=base['reference_count'], self_reference_excluded=True)
    return geometry.public_result(result, elapsed=result['compute_seconds'], cache_hit=False)


def geometry_module():
    spec = importlib.util.spec_from_file_location('_speed10_geometry', OLD / 'geometry_adapter.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene_protocol(scene):
    if scene not in SCENES:
        raise ValueError('Unknown scene')
    if scene in ('block_a', 'block_small'):
        return RGB / 'logs' / ('block_a_full_native_v1' if scene == 'block_a' else 'full_native_v1') / 'protocol.json'
    return (RGB / 'other_baselines_nine_completion_20261004_v1' / scene
            / ('logs_v2' if scene in ('park', 'road') else 'logs') / 'protocol.json')


def source_exclusions(query, frames, training):
    """Identify self reference by exact source pixels, not a query GT pose."""
    matched = [f for f in frames if f['image_sha256'] == query['image_sha256']]
    if not matched:
        # Non-indexed inputs have no exact self reference. Do not guess a match.
        return []
    training_ids = {f['training_template_id'] for f in matched}
    selected = [r for r in training if r['id'] in training_ids]
    if not selected or {r['id'] for r in selected} != training_ids:
        raise ValueError('Pixel-matched original reference absent from map')
    selected_hashes = {r['sha256'] for r in selected}
    selected_poses = [np.asarray(r['c2w']) for r in selected]
    return sorted(r['name'] for r in training
                  if r['id'] in training_ids or r['sha256'] in selected_hashes
                  or any(np.allclose(np.asarray(r['c2w']), c, rtol=0, atol=1e-6)
                         for c in selected_poses))


def normalize_query(task):
    """Only pixels/intrinsics and opaque bookkeeping enter the feature plan."""
    source = task.get('source_image')
    if isinstance(source, dict):
        path, checksum = source['image_path'], source['image_sha256']
        width, height = source['width'], source['height']
        intrinsics = task.get('K', source.get('K'))
    else:
        path, checksum = source, task['source_image_sha256']
        width, height = task['width'], task['height']
        intrinsics = task['K']
    handle = task.get('host_bundle_id') or task.get('pair_id') or task['id']
    if not isinstance(handle, str) or not handle or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in handle):
        raise ValueError('Invalid opaque source handle')
    if task['scene'] not in SCENES:
        raise ValueError('Unknown scene')
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError('Invalid source checksum')
    checked_sha(path, checksum)
    with Image.open(path) as image:
        if image.size != (width, height):
            raise ValueError('Source image dimensions do not match intrinsics')
    k = np.asarray(intrinsics, dtype=float)
    if (k.shape != (3, 3) or not np.isfinite(k).all()
        or min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], [0, 0, 1])):
        raise ValueError('Invalid source intrinsics')
    return {'id': handle, 'scene': task['scene'], 'name': handle + '.png',
            'image_path': str(Path(path).resolve()), 'image_sha256': checksum,
            'K': k.tolist(), 'width': width, 'height': height}


def prepare_geometry(tasks, out=GEOMETRY):
    """CPU preparation only; reuse all map/reference features, freeze new inputs.

    Pass already selected public task dictionaries. This function never opens the
    dataset's annotation bundle, accepted labels, witnesses, or query GT poses.
    Features for actual source pixels are computed by run_geometry, not copied
    from an indexed reference or historical query id.
    """
    geometry = geometry_module()
    queries, by_handle, scenes, scene_rows = [], {}, {}, {}
    for task in tasks:
        query = normalize_query(task)
        handle, scene = query['id'], query['scene']
        if handle in by_handle:
            previous = {k: v for k, v in by_handle[handle].items() if k != 'excluded_reference_names'}
            if previous != query:
                raise ValueError('Source handle reused for different pixels or intrinsics')
            continue
        if scene not in scenes:
            protocol_path = scene_protocol(scene)
            protocol = read(protocol_path)
            sfm = Path(protocol['sfm_source'])
            training_path = sfm / 'protocol.json'
            training = read(training_path)
            summary = read(sfm / 'summary.json')
            if summary.get('status') != 'complete' or training['feature_config'] != geometry.FEATURE_CONFIG:
                raise ValueError('Unqualified reference SuperPoint recipe')
            reference_features = sfm / 'feats-superpoint-n4096-r1024.h5'
            reference_global = protocol_path.parent / 'initializer/reference_netvlad.h5'
            if not reference_features.is_file() or not reference_global.is_file():
                raise FileNotFoundError('Reference feature bank missing')
            original_manifest = ROOT / 'qgvi-rag/outputs/benchmark_cityloc_v2/original_rgb_banks_v1' / scene / 'manifest.json'
            scenes[scene] = {'protocol_path': str(protocol_path), 'protocol_sha256': sha(protocol_path),
                'training_protocol': str(training_path), 'training_protocol_sha256': sha(training_path),
                'reference_features': str(reference_features), 'reference_features_sha256': summary['features_sha256'],
                'reference_global': str(reference_global), 'reference_global_sha256': sha(reference_global),
                'sfm_dir': str(sfm / 'sfm'), 'sfm_files': summary['artifacts_sha256'],
                'reference_names': [r['name'] for r in training['records']],
                'torch_cache': protocol['torch_cache'], 'seed': protocol['seed'],
                'netvlad_weights': protocol['netvlad_weights'], 'netvlad_weights_sha256': protocol['netvlad_weights_sha256'],
                'meters_per_model_unit': protocol['meters_per_model_unit'],
                'original_manifest': str(original_manifest), 'original_manifest_sha256': sha(original_manifest)}
            scene_rows[scene] = (read(original_manifest)['frames'], training['records'])
        asset = scenes[scene]
        frames, records = scene_rows[scene]
        query['excluded_reference_names'] = source_exclusions(query, frames, records)
        queries.append(query)
        by_handle[handle] = query
    if not queries:
        raise ValueError('No selected source queries')
    plan = {'schema_version': 1, 'recipe': 'SP4096-r1024-fp16_SG-outdoor-it5_NetVLAD10_PnP12px_seed20261025_no_fallback',
        'source_sha256': sha(OLD / 'geometry_adapter.py'), 'confidence_sha256': sha(geometry.CONFIDENCE),
        'adapter_source_sha256': sha(__file__), 'timing_source_sha256': sha(HERE / 'source_geometry.py'),
        'cache_import_source_sha256': sha(HERE / 'rgb_cache_import.py'),
        'queries': queries, 'scenes': scenes,
        'feature_config': geometry.FEATURE_CONFIG, 'matcher_config': geometry.MATCHER_CONFIG,
        'top_k': 10, 'roi_convention': 'normalized keep-box; filter full-image correspondences; unchanged K',
        'source_binding': 'actual source pixels and intrinsics; no source GT pose consumed',
        'self_exclusion': 'exact pixel matching to map reference; exclude duplicated indexed cameras privately',
        'disclosure': 'Reconstruction-exposed development inputs; query descriptors newly extracted, reference map/features reused.'}
    target = Path(out) / 'plan.json'
    if target.exists() and read(target) != plan:
        raise ValueError('Frozen geometry preparation changed; use a new output directory')
    if not target.exists():
        write(target, plan)
    return plan


def run_geometry(out=GEOMETRY):
    out = Path(out).resolve()
    plan = read(Path(out) / 'plan.json')
    checked_sha(__file__, plan['adapter_source_sha256'])
    for scene in plan['scenes'].values():
        checked_sha(scene['original_manifest'], scene['original_manifest_sha256'])
    spec = importlib.util.spec_from_file_location('_speed10_timed_geometry', HERE / 'source_geometry.py')
    timed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timed)
    return timed.run(Path(out), geometry_module())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-tasks', type=Path)
    parser.add_argument('--geometry-dir', type=Path, default=GEOMETRY)
    parser.add_argument('--audit-rgb-cache', action='store_true')
    parser.add_argument('--run-geometry', action='store_true')
    parser.add_argument('--pose-worker', action='store_true')
    parser.add_argument('--bundle-id')
    parser.add_argument('--region', type=json.loads)
    parser.add_argument('--scene', choices=SCENES)
    parser.add_argument('--workdir', type=Path, default=HERE)
    parser.add_argument('--query', default='store entrance red sign')
    args = parser.parse_args()
    if args.prepare_tasks:
        tasks = read(args.prepare_tasks)
        if isinstance(tasks, dict):
            tasks = tasks['queries']
        for task in tasks:
            if 'task_path' in task and 'task_sha256' in task:
                checked_sha(task['task_path'], task['task_sha256'])
        tasks = [read(t['task_path']) if 'task_path' in t else t for t in tasks]
        plan = prepare_geometry(tasks, args.geometry_dir)
        print(json.dumps({'status': 'prepared', 'queries': len(plan['queries']), 'scenes': list(plan['scenes']), 'gpu_launched': False}))
    elif args.audit_rgb_cache:
        import rgb_cache_import
        print(json.dumps(rgb_cache_import.audit(args.geometry_dir), allow_nan=False))
    elif args.run_geometry:
        print(json.dumps(run_geometry(args.geometry_dir), allow_nan=False))
    elif args.pose_worker:
        print(json.dumps(pose_worker(args.bundle_id, args.region, args.geometry_dir), allow_nan=False))
    else:
        if not args.scene:
            parser.error('--scene is required')
        backend = RetrievalBackend(args.scene, args.workdir, args.geometry_dir)
        print(json.dumps(backend.search_text(args.query, bundle_id=args.bundle_id), allow_nan=False))


if __name__ == '__main__':
    main()
