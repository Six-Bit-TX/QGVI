"""Timed source preparation; reuse immutable LoGS maps and reference features.

Each source is processed separately with the unchanged feature/matcher/PnP
recipe. Model initialization inside hloc extraction/matching IS timed. Loading
reference descriptors and the sparse map is scene setup outside source timing.
No source GT pose or semantic annotation is consumed.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent


def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())
    tmp.replace(path)
def require(value, message):
    if not value: raise ValueError(message)


class Runtime:
    def __init__(self, geometry):
        import torch
        import pycolmap
        sys.path.insert(0, str(geometry.UPSTREAM))
        from hloc import extract_features, match_features
        require(torch.cuda.is_available(), 'CUDA unavailable; no silent CPU feature inference')
        self.geometry, self.torch, self.pycolmap = geometry, torch, pycolmap
        self.extract_features, self.match_features = extract_features, match_features
        self.confidence = geometry.confidence_module()
        self.reuse_rows = {}; self.batch_globals = {}; self.reused_stages = {}; self.match_query = {}
        torch.set_num_threads(2); torch.manual_seed(geometry.SEED); np.random.seed(geometry.SEED)
        torch.hub.download_url_to_file = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('Unprepared model download prohibited'))

    def synchronize(self): self.torch.cuda.synchronize()

    def load_reference(self, asset):
        self.torch.hub.set_dir(asset['torch_cache'])
        self.synchronize(); started = time.perf_counter()
        names = sorted(asset['reference_names'])
        with h5py.File(asset['reference_global'], 'r') as refs:
            matrix = np.stack([np.asarray(refs[n]['global_descriptor'], np.float32).reshape(-1) for n in names])
        descriptor_seconds = time.perf_counter() - started
        started = time.perf_counter()
        reconstruction = self.pycolmap.Reconstruction(asset['sfm_dir'])
        proxy = SimpleNamespace(images=reconstruction.images, points3D=reconstruction.points3D)
        name_to_id = {im.name: ident for ident, im in reconstruction.images.items()}
        map_seconds = time.perf_counter() - started
        return {'names': names, 'matrix': matrix, 'reconstruction': reconstruction,
                'proxy': proxy, 'name_to_id': name_to_id,
                'setup_seconds': {'load_reference_descriptors': descriptor_seconds, 'load_sparse_map': map_seconds}}

    def configure_reuse(self, audit):
        self.reuse_rows = {row['query_id']: row for row in (audit or {}).get('queries', {}).values() if row.get('eligible')}

    def prepare_scene(self, queries, scene_work, plan_sha256):
        """Batch fresh NetVLAD for reuse-qualified RGB; its old bytes were unbound."""
        import rgb_cache_import as reuse
        selected = [q for q in queries if q['id'] in self.reuse_rows]
        if not selected: return {}
        root = Path(scene_work) / 'fresh_rgb_netvlad'
        receipt = root / 'COMPLETED.json'
        if receipt.exists():
            record = read(receipt)
            require(record['plan_sha256'] == plan_sha256, 'RGB global descriptor plan changed')
            require(sha(record['path']) == record['sha256'], 'RGB fresh global descriptors changed')
            require(record['query_names'] == [q['name'] for q in selected], 'RGB global membership changed')
        else:
            root.mkdir(parents=True, exist_ok=True)
            n = 1
            while (root / f'attempt_{n:03d}').exists(): n += 1
            attempt = root / f'attempt_{n:03d}'; inputs = attempt / 'inputs'; inputs.mkdir(parents=True)
            self.synchronize(); start = time.perf_counter()
            for q in selected:
                require(sha(q['image_path']) == q['image_sha256'], 'RGB batch source pixels changed')
                shutil.copyfile(q['image_path'], inputs / q['name'])
            path = self.extract_features.main(self.extract_features.confs['netvlad'], inputs, attempt,
                as_half=True, image_list=[q['name'] for q in selected], feature_path=attempt / 'query_netvlad.h5')
            self.synchronize(); elapsed = time.perf_counter() - start
            record = {'plan_sha256': plan_sha256, 'path': str(path), 'sha256': sha(path),
                'query_names': [q['name'] for q in selected], 'batch_wall_seconds': elapsed,
                'timing_scope': 'Fresh batched NetVLAD plus input staging and model initialization; throughput cost, not single-query latency'}
            write(receipt, record)
        for q in selected: self.batch_globals[q['name']] = record['path']
        return {'rgb_fresh_netvlad_batch_seconds': record['batch_wall_seconds'], 'rgb_batch_query_count': len(selected)}

    def extract_global(self, inputs, attempt, name):
        if name in self.batch_globals:
            import rgb_cache_import as reuse
            return reuse.copy_feature(self.batch_globals[name], name, attempt / 'query_netvlad.h5', name)
        return self.extract_features.main(self.extract_features.confs['netvlad'], inputs, attempt,
            as_half=True, image_list=[name], feature_path=attempt / 'query_netvlad.h5')

    def extract_local(self, inputs, attempt, name):
        row = self.reuse_rows.get(name.removesuffix('.png'))
        if row:
            import rgb_cache_import as reuse
            self.reused_stages.setdefault(name, []).append('query_superpoint')
            return reuse.copy_feature(row['query_superpoint'], row['source_name'], attempt / 'query_superpoint.h5', name)
        return self.extract_features.main(self.geometry.FEATURE_CONFIG, inputs, attempt,
            as_half=True, image_list=[name], feature_path=attempt / 'query_superpoint.h5')

    def match(self, pairs, features, attempt, asset):
        lines = [line.split() for line in Path(pairs).read_text().splitlines()]
        name = lines[0][0] if lines else ''
        row = self.reuse_rows.get(name.removesuffix('.png'))
        references = [p[1] for p in lines]
        if row and references == row['reference_names']:
            import rgb_cache_import as reuse
            path = reuse.copy_matches(row['matches'], row['source_name'], name, references, attempt / 'matches.h5')
            self.reused_stages.setdefault(name, []).append('query_superglue')
            self.match_query[str(path)] = row
            return path
        return self.match_features.main(self.geometry.MATCHER_CONFIG, pairs, features,
            matches=attempt / 'matches.h5', features_ref=Path(asset['reference_features']))

    def assemble(self, query, retrieved, features, matches, reference):
        corr = self.confidence.assemble_correspondences(reference['proxy'], query['name'], retrieved,
            features, matches, name_to_id=reference['name_to_id'])
        row = self.match_query.get(str(matches))
        if row:
            require(corr['raw_match_observation_count'] == row['raw_match_observation_count'], 'Reused SG correspondence count differs from producer')
        return corr

    def query_reuse(self, query):
        return {'source_preparation_reused': query['id'] in self.reuse_rows,
            'reused_stages': self.reused_stages.get(query['name'], []),
            'netvlad_fresh_batched': query['name'] in self.batch_globals}

    def solve(self, query, corr): return self.geometry.solve(query, corr)
    def save_corr(self, path, corr): return self.geometry.save_corr(path, corr)
    def release_reference(self): self.torch.cuda.empty_cache()


def rank_references(query, global_path, reference, top_k):
    with h5py.File(global_path, 'r') as features:
        descriptor = np.asarray(features[query['name']]['global_descriptor'], np.float32).reshape(-1)
    require(descriptor.shape == (reference['matrix'].shape[1],) and np.isfinite(descriptor).all(),
            'Invalid new source NetVLAD descriptor')
    scores = reference['matrix'] @ descriptor
    names = reference['names']; excluded = set(query['excluded_reference_names'])
    require(excluded <= set(names), 'Unknown excluded reference')
    order = sorted(range(len(names)), key=lambda i: (-float(scores[i]), names[i]))
    retrieved = [names[i] for i in order if names[i] not in excluded][:top_k]
    require(len(retrieved) == min(top_k, len(names) - len(excluded)), 'Retrieval count differs')
    require(not set(retrieved) & excluded, 'Self reference survived')
    return retrieved


def prepare_one_source(query, asset, scene_work, reference, runtime, *, plan_sha256, top_k=10):
    """Time a complete single-source preparation with fresh output files.

    A completed source is replayed by verified receipt. An incomplete attempt is
    retained and a fresh attempt directory prevents hloc silently skipping work.
    This routine can be tested with CPU fixture runtime methods.
    """
    source_root = Path(scene_work) / 'source_preparation' / query['id']
    receipt_path = source_root / 'PREPARED.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt['plan_sha256'] == plan_sha256 and receipt['query_image_sha256'] == query['image_sha256'],
                'Prepared source binding changed')
        for path, checksum in receipt['outputs_sha256'].items():
            require(sha(path) == checksum, 'Prepared source output changed')
        return receipt
    source_root.mkdir(parents=True, exist_ok=True)
    number = 1
    while (source_root / f'attempt_{number:03d}').exists(): number += 1
    attempt = source_root / f'attempt_{number:03d}'; attempt.mkdir()
    inputs = attempt / 'inputs'; inputs.mkdir()
    timings = {}
    def measured(name, operation):
        runtime.synchronize(); begin = time.perf_counter()
        result = operation(); runtime.synchronize()
        timings[name] = time.perf_counter() - begin
        return result
    runtime.synchronize(); begin = time.perf_counter()
    def stage_input():
        require(sha(query['image_path']) == query['image_sha256'], 'Source pixels changed')
        destination = inputs / query['name']
        shutil.copyfile(query['image_path'], destination)
        require(sha(destination) == query['image_sha256'], 'Staged source pixels changed')
    measured('stage_source_image', stage_input)
    global_path = measured('query_netvlad', lambda: runtime.extract_global(inputs, attempt, query['name']))
    features = measured('query_superpoint', lambda: runtime.extract_local(inputs, attempt, query['name']))
    retrieved = measured('reference_ranking', lambda: rank_references(query, global_path, reference, top_k))
    pairs = attempt / 'pairs.txt'
    measured('write_retrieval_pairs', lambda: pairs.write_text(''.join(f"{query['name']} {r}\n" for r in retrieved)))
    matches = measured('query_superglue', lambda: runtime.match(pairs, features, attempt, asset))
    corr = measured('assemble_correspondences', lambda: runtime.assemble(query, retrieved, features, matches, reference))
    result = measured('solve_pnp', lambda: runtime.solve(query, corr))
    corr_path = attempt / 'correspondences.npz'
    measured('save_correspondences', lambda: runtime.save_corr(corr_path, corr))
    result.update(hypothesis_id=hashlib.sha256(json.dumps({'query': query['image_sha256'], 'plan': plan_sha256, 'roi': None}, sort_keys=True).encode()).hexdigest()[:24],
        query_image_sha256=query['image_sha256'], plan_sha256=plan_sha256,
        correspondences_sha256=sha(corr_path), reference_count=len(retrieved), self_reference_excluded=True)
    result_path = attempt / 'result.json'
    measured('save_result', lambda: write(result_path, result))
    runtime.synchronize(); elapsed = time.perf_counter() - begin
    timing = {'source_preparation_wall_seconds': elapsed, 'source_preparation_wall_ms': 1000 * elapsed,
        'stage_seconds': timings, 'cuda_synchronized': True,
        'scope': 'single new source: input staging, NetVLAD, SuperPoint, ranking, SuperGlue, correspondence assembly, PnP and result persistence',
        'feature_and_match_model_initialization_included': True,
        'reference_descriptor_and_sparse_map_setup_excluded': True,
        'cache_export_and_receipt_hashing_excluded': True,
        'attempt_number': number, 'prior_incomplete_attempts_retained': number - 1}
    reuse = runtime.query_reuse(query) if hasattr(runtime, 'query_reuse') else {}
    timing.update(reuse)
    timing['current_preparation_wall_seconds'] = elapsed
    if reuse.get('source_preparation_reused'):
        timing.update(source_preparation_wall_seconds=None, source_preparation_wall_ms=None,
            scope='Partial cached-source preparation; current wall excludes historical reused stages and the separately measured fresh NetVLAD batch. Fresh-input latency is unavailable.',
            feature_and_match_model_initialization_included=False)
    result['source_preparation_timing'] = timing
    write(result_path, result)
    paths = {'query_netvlad': str(global_path), 'query_superpoint': str(features), 'matches': str(matches),
             'pairs': str(pairs), 'correspondences': str(corr_path), 'result': str(result_path)}
    receipt = {'status': 'complete', 'query_id': query['id'], 'scene': query['scene'],
        'plan_sha256': plan_sha256, 'query_image_sha256': query['image_sha256'], 'timing': timing,
        'paths': paths, 'outputs_sha256': {path: sha(path) for path in paths.values()}}
    write(receipt_path, receipt)
    return receipt


def group_equal(left, right):
    if type(left) is not type(right) or set(left.attrs) != set(right.attrs): return False
    if any(not np.array_equal(left.attrs[k], right.attrs[k]) for k in left.attrs): return False
    if isinstance(left, h5py.Dataset):
        return left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left[()], right[()])
    return set(left) == set(right) and all(group_equal(left[k], right[k]) for k in left)


def merge_h5(source, destination):
    """Merge disjoint source groups; never silently overwrite different data."""
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, 'r') as original, h5py.File(destination, 'a') as merged:
        for name in original:
            if name in merged:
                require(group_equal(original[name], merged[name]), 'Conflicting merged source feature group')
            else:
                original.copy(name, merged)


def export_receipt(receipt, out, scene_work):
    out, scene_work = Path(out), Path(scene_work)
    paths = receipt['paths']; outputs = []
    for key, filename in [('query_netvlad', 'query_netvlad.h5'), ('query_superpoint', 'query_superpoint.h5'), ('matches', 'matches.h5')]:
        target = scene_work / filename; merge_h5(paths[key], target); outputs.append(target)
    for key, folder, suffix in [('correspondences', 'correspondences', '.npz'), ('result', 'results', '.json')]:
        target = out / folder / (receipt['query_id'] + suffix); target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists(): require(sha(target) == sha(paths[key]), 'Canonical prepared source output changed')
        else: shutil.copyfile(paths[key], target)
        outputs.append(target)
    outputs += [Path(p) for p in paths.values()]
    return outputs


def run(out, geometry, *, runtime_factory=Runtime):
    out = Path(out).resolve(); plan = geometry.validate_plan(out)
    require(sha(__file__) == plan['timing_source_sha256'], 'Timed preparation source changed')
    if (out / 'COMPLETED.json').exists():
        done = read(out / 'COMPLETED.json')
        require(done['plan_sha256'] == sha(out / 'plan.json'), 'Completion plan changed')
        for path, checksum in done['outputs_sha256'].items(): require(sha(path) == checksum, 'Completed cache changed')
        return done
    checks_started = time.perf_counter()
    for asset in plan['scenes'].values():
        checks = {asset['protocol_path']: asset['protocol_sha256'], asset['training_protocol']: asset['training_protocol_sha256'],
            asset['reference_features']: asset['reference_features_sha256'], asset['reference_global']: asset['reference_global_sha256'],
            asset['original_manifest']: asset['original_manifest_sha256'], **asset['sfm_files']}
        if asset.get('netvlad_weights'): checks[asset['netvlad_weights']] = asset['netvlad_weights_sha256']
        for path, checksum in checks.items(): require(sha(path) == checksum, 'Reference cache changed: ' + path)
    import rgb_cache_import as reuse
    cache_audit = reuse.load_verified(out)
    validation_seconds = time.perf_counter() - checks_started
    with geometry.gpu_lock():
        start = time.perf_counter(); runtime = runtime_factory(geometry)
        if hasattr(runtime, 'configure_reuse'): runtime.configure_reuse(cache_audit)
        runtime_initialization_seconds = time.perf_counter() - start
        outputs, per_query, setups = [], {}, {}; stage_seconds = defaultdict(float); completed = 0
        plan_sha = sha(out / 'plan.json')
        for scene, asset in plan['scenes'].items():
            scene_work = out / scene; scene_work.mkdir(parents=True, exist_ok=True)
            write(out / 'status.json', {'status': 'running', 'stage': 'load_reference_and_sparse_map', 'scene': scene,
                                       'completed': completed, 'total': len(plan['queries']), 'updated_unix': time.time()})
            setup_start = time.perf_counter(); reference = runtime.load_reference(asset)
            setups[scene] = {**reference['setup_seconds'], 'total_scene_setup_seconds': time.perf_counter() - setup_start}
            stage_seconds[scene + '/load_sparse_map'] = reference['setup_seconds']['load_sparse_map']
            if hasattr(runtime, 'prepare_scene'):
                setups[scene].update(runtime.prepare_scene([q for q in plan['queries'] if q['scene'] == scene], scene_work, plan_sha))
            batch_receipt = scene_work / 'fresh_rgb_netvlad/COMPLETED.json'
            if batch_receipt.exists():
                outputs += [batch_receipt, Path(read(batch_receipt)['path'])]
            scene_receipts = []
            for query in [q for q in plan['queries'] if q['scene'] == scene]:
                write(out / 'status.json', {'status': 'running', 'stage': 'single_source_preparation', 'scene': scene,
                    'query_id': query['id'], 'completed': completed, 'total': len(plan['queries']), 'updated_unix': time.time()})
                receipt = prepare_one_source(query, asset, scene_work, reference, runtime, plan_sha256=plan_sha, top_k=plan['top_k'])
                outputs += export_receipt(receipt, out, scene_work); scene_receipts.append(receipt)
                per_query[query['id']] = receipt['timing']
                for name, elapsed in receipt['timing']['stage_seconds'].items(): stage_seconds[scene + '/' + name] += elapsed
                completed += 1
                result = read(receipt['paths']['result'])
                print(json.dumps({'event': 'geometry_pose', 'completed': completed, 'total': len(plan['queries']), 'scene': scene,
                    'pnp_success': result['status'] == 'estimated', 'unique_inliers': result['evidence']['unique_inlier_queries'],
                    'source_preparation_seconds': receipt['timing']['source_preparation_wall_seconds']}), flush=True)
            pairs = scene_work / 'pairs.txt'
            pairs.write_text(''.join(Path(r['paths']['pairs']).read_text() for r in scene_receipts)); outputs.append(pairs)
            reference = None; runtime.release_reference()
        source_seconds = {key: timing['source_preparation_wall_seconds'] for key, timing in per_query.items()}
        done = {'status': 'complete', 'query_count': completed, 'plan_sha256': plan_sha,
            'outputs_sha256': {str(p): sha(p) for p in sorted(set(outputs))},
            'total_seconds': time.perf_counter() - start, 'stage_seconds': dict(stage_seconds),
            'per_query_timings': per_query, 'per_query_preparation_seconds': source_seconds,
            'source_preparation_seconds_by_query': source_seconds,
            'total_recorded_source_preparation_seconds': sum(v for v in source_seconds.values() if v is not None),
            'fresh_source_latency_unavailable_count': sum(v is None for v in source_seconds.values()),
            'source_preparation_reused_by_query': {q: bool(t.get('source_preparation_reused')) for q, t in per_query.items()},
            'cache_import_audit_sha256': sha(out / 'RGB_CACHE_AUDIT.json') if cache_audit else None,
            'setup_seconds': {'reference_cache_validation': validation_seconds,
                              'runtime_initialization': runtime_initialization_seconds, 'by_scene': setups},
            'reference_database_rebuilt': False, 'source_gt_used_for_inference': False,
            'source_gt_used_privately_for_self_exclusion': False,
            'indexed_camera_metadata_used_for_pixel_matched_duplicate_exclusion': True,
            'timing_scope': 'Measured individual new-source preparation; each feature/matcher model initialization included. Scene map/reference setup separate. Add once to each arm only as a reconstructed total.',
            'model_versions': {'torch': runtime.torch.__version__, 'pycolmap': runtime.pycolmap.__version__}}
        write(out / 'COMPLETED.json', done)
        write(out / 'status.json', {'status': 'complete', 'stage': 'complete', 'completed': completed,
                                   'total': len(plan['queries']), 'updated_unix': time.time()})
        return done
