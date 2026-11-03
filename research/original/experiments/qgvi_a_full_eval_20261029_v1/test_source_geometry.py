"""Meaningful preparation/timing/cache tests using CPU-only deterministic fixtures."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from contextlib import nullcontext

import h5py
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('timed_geometry_fixture', HERE / 'source_geometry.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class FixtureRuntime:
    def __init__(self, geometry=None):
        self.calls = []; self.sync_count = 0
        self.torch = SimpleNamespace(__version__='fixture')
        self.pycolmap = SimpleNamespace(__version__='fixture')
    def synchronize(self): self.sync_count += 1
    def load_reference(self, asset):
        self.calls.append('load_reference')
        return {'names': ['ref0.png', 'ref1.png', 'ref2.png'], 'matrix': np.eye(3, dtype=np.float32),
                'setup_seconds': {'load_reference_descriptors': .003, 'load_sparse_map': .004}}
    def release_reference(self): self.calls.append('release_reference')
    def extract_global(self, inputs, attempt, name):
        self.calls.append(('netvlad', name)); path = attempt / 'query_netvlad.h5'
        with h5py.File(path, 'w') as f:
            f.create_group(name).create_dataset('global_descriptor', data=np.array([1., .5, .2], np.float16))
        return path
    def extract_local(self, inputs, attempt, name):
        self.calls.append(('superpoint', name)); path = attempt / 'query_superpoint.h5'
        with h5py.File(path, 'w') as f:
            group = f.create_group(name); ds = group.create_dataset('keypoints', data=np.array([[1, 1], [2, 2]], np.float16)); ds.attrs['uncertainty'] = 1.
            group.create_dataset('image_size', data=np.array([16, 12]))
        return path
    def match(self, pairs, features, attempt, asset):
        self.calls.append(('superglue', pairs.read_text())); path = attempt / 'matches.h5'
        with h5py.File(path, 'w') as f:
            for line in pairs.read_text().splitlines():
                name, ref = line.split(); f.create_group(name + '_' + ref).create_dataset('matches0', data=np.array([0, 1], np.int16))
        return path
    def assemble(self, query, retrieved, features, matches, reference):
        self.calls.append(('assemble', query['id'])); return {'query': query['id'], 'references': retrieved}
    def solve(self, query, corr):
        self.calls.append(('pnp', query['id']))
        return {'status': 'estimated', 'pose_c2w': np.eye(4).tolist(), 'evidence': {'unique_inlier_queries': 4}, 'compute_seconds': .001}
    def save_corr(self, path, corr):
        self.calls.append('save_corr')
        with path.open('wb') as f: np.savez_compressed(f, metadata=np.array(json.dumps(corr)))


class TimedSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.runtime = FixtureRuntime()
        self.reference = self.runtime.load_reference({})
        self.image = self.root / 'source.png'; Image.new('RGB', (16, 12), 'red').save(self.image)
    def query(self, name='first'):
        return {'id': name, 'name': name + '.png', 'scene': 'sf', 'image_path': str(self.image),
            'image_sha256': m.sha(self.image), 'K': [[10,0,8],[0,10,6],[0,0,1]], 'width': 16, 'height': 12,
            'excluded_reference_names': ['ref0.png']}
    def prepare(self, name='first'):
        return m.prepare_one_source(self.query(name), {}, self.root / 'sf', self.reference, self.runtime,
                                    plan_sha256='frozen_plan', top_k=2)
    def test_full_sequence_timing_no_gt_and_self_exclusion(self):
        receipt = self.prepare(); timing = receipt['timing']
        expected = {'stage_source_image', 'query_netvlad', 'query_superpoint', 'reference_ranking',
                    'write_retrieval_pairs', 'query_superglue', 'assemble_correspondences', 'solve_pnp',
                    'save_correspondences', 'save_result'}
        self.assertEqual(set(timing['stage_seconds']), expected)
        self.assertGreater(timing['source_preparation_wall_seconds'], 0)
        self.assertGreaterEqual(timing['source_preparation_wall_seconds'], sum(timing['stage_seconds'].values()))
        self.assertEqual(self.runtime.sync_count, 22)
        self.assertTrue(timing['feature_and_match_model_initialization_included'])
        self.assertTrue(timing['reference_descriptor_and_sparse_map_setup_excluded'])
        pairs = Path(receipt['paths']['pairs']).read_text()
        self.assertNotIn('ref0.png', pairs)
        self.assertIn('ref1.png', pairs); self.assertIn('ref2.png', pairs)
        self.assertNotIn('source_c2w', str(receipt))
    def test_two_query_h5_merge_preserves_both_sources_and_attributes(self):
        first, second = self.prepare(), self.prepare('second')
        for receipt in [first, second]: m.export_receipt(receipt, self.root, self.root / 'sf')
        with h5py.File(self.root / 'sf/query_netvlad.h5') as f:
            self.assertEqual(set(f), {'first.png', 'second.png'})
        with h5py.File(self.root / 'sf/query_superpoint.h5') as f:
            self.assertEqual(f['first.png/keypoints'].attrs['uncertainty'], 1.)
            self.assertEqual(set(f), {'first.png', 'second.png'})
        with h5py.File(self.root / 'sf/matches.h5') as f: self.assertEqual(len(f), 4)
        m.export_receipt(first, self.root, self.root / 'sf')  # identical repeated export allowed
    def test_completed_receipt_replays_without_recomputation(self):
        first = self.prepare(); count = len(self.runtime.calls)
        self.assertEqual(self.prepare(), first); self.assertEqual(len(self.runtime.calls), count)
    def test_replay_rejects_changed_preparation_output(self):
        receipt = self.prepare(); Path(receipt['paths']['result']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'output changed'): self.prepare()
    def test_incomplete_attempt_retained_and_new_directory_used(self):
        interrupted = self.root / 'sf/source_preparation/first/attempt_001'; interrupted.mkdir(parents=True)
        (interrupted / 'sentinel').write_text('old incomplete output')
        receipt = self.prepare()
        self.assertEqual(receipt['timing']['attempt_number'], 2)
        self.assertTrue((interrupted / 'sentinel').exists())
    def test_merge_rejects_conflicting_existing_group(self):
        receipt = self.prepare(); destination = self.root / 'merged.h5'
        with h5py.File(destination, 'w') as f:
            f.create_group('first.png').create_dataset('global_descriptor', data=np.zeros(3, np.float16))
        with self.assertRaisesRegex(ValueError, 'Conflicting'): m.merge_h5(receipt['paths']['query_netvlad'], destination)
    def test_changed_source_rejected_before_inference(self):
        query = self.query(); Image.new('RGB', (16, 12), 'blue').save(self.image)
        with self.assertRaisesRegex(ValueError, 'Source pixels changed'):
            m.prepare_one_source(query, {}, self.root / 'sf', self.reference, self.runtime, plan_sha256='frozen_plan')
        self.assertFalse(any(isinstance(v, tuple) and v[0]=='netvlad' for v in self.runtime.calls))
    def test_reused_source_does_not_claim_fresh_latency(self):
        self.runtime.query_reuse = lambda q: {'source_preparation_reused': True, 'reused_stages': ['query_superpoint'], 'netvlad_fresh_batched': True}
        receipt = self.prepare(); timing = receipt['timing']
        self.assertIsNone(timing['source_preparation_wall_seconds'])
        self.assertIsNone(timing['source_preparation_wall_ms'])
        self.assertGreater(timing['current_preparation_wall_seconds'], 0)
        self.assertFalse(timing['feature_and_match_model_initialization_included'])

    def test_full_run_contract_and_cached_completion(self):
        evidence = self.root / 'evidence.json'; evidence.write_text('{}'); h = m.sha(evidence)
        asset = {'protocol_path': str(evidence), 'protocol_sha256': h,
            'training_protocol': str(evidence), 'training_protocol_sha256': h,
            'reference_features': str(evidence), 'reference_features_sha256': h,
            'reference_global': str(evidence), 'reference_global_sha256': h,
            'original_manifest': str(evidence), 'original_manifest_sha256': h, 'sfm_files': {str(evidence): h}}
        plan = {'timing_source_sha256': m.sha(HERE/'source_geometry.py'), 'queries': [self.query(), self.query('second')],
                'scenes': {'sf': asset}, 'top_k': 2}
        m.write(self.root/'plan.json', plan)
        geometry = SimpleNamespace(validate_plan=lambda out:m.read(out/'plan.json'), gpu_lock=lambda:nullcontext())
        result = m.run(self.root, geometry, runtime_factory=lambda _:self.runtime)
        self.assertEqual(set(result['source_preparation_seconds_by_query']), {'first','second'})
        self.assertEqual(result['query_count'], 2)
        self.assertFalse(result['reference_database_rebuilt']); self.assertFalse(result['source_gt_used_for_inference'])
        self.assertEqual(result['source_preparation_seconds_by_query']['first'], result['per_query_timings']['first']['source_preparation_wall_seconds'])
        count = len(self.runtime.calls)
        self.assertEqual(m.run(self.root, geometry, runtime_factory=lambda _:self.runtime), result)
        self.assertEqual(len(self.runtime.calls), count)
        for query in plan['queries']:
            self.assertTrue((self.root / 'results' / (query['id']+'.json')).is_file())
            self.assertTrue((self.root / 'correspondences' / (query['id']+'.npz')).is_file())


if __name__ == '__main__': unittest.main()
