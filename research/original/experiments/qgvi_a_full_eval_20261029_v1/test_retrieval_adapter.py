"""CPU-only fixture tests: no models, annotation reads, or production writes."""
import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('speed10_retrieval_tested', HERE / 'retrieval_backend.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class SourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'source.png'
        Image.new('RGB', (24, 16), (40, 80, 120)).save(self.path)
        self.task = {'id': 'scene_q1', 'host_bundle_id': 'scene_pair', 'scene': 'sf',
                     'source_image': str(self.path), 'source_image_sha256': b.sha(self.path),
                     'width': 24, 'height': 16, 'K': [[12, 0, 12], [0, 12, 8], [0, 0, 1]]}

    def test_normalization_drops_all_labels_and_gt(self):
        task = self.task | {'source_c2w': 'secret', 'accepted_gallery_ids': ['A'],
                            'instruction': 'not needed by feature extraction', 'witnesses': ['secret']}
        value = b.normalize_query(task)
        self.assertEqual(set(value), {'id', 'scene', 'name', 'image_path', 'image_sha256', 'K', 'width', 'height'})
        self.assertEqual(value['id'], 'scene_pair')
        self.assertNotIn('secret', str(value))

    def test_nested_public_source_supported(self):
        nested = self.task | {'source_image': {'image_path': str(self.path),
            'image_sha256': b.sha(self.path), 'width': 24, 'height': 16}}
        self.assertEqual(b.normalize_query(nested), b.normalize_query(self.task))

    def test_changed_pixels_fail(self):
        Image.new('RGB', (24, 16), (80, 10, 60)).save(self.path)
        with self.assertRaises(ValueError):
            b.normalize_query(self.task)

    def test_dimension_mismatch_fails(self):
        with self.assertRaises(ValueError):
            b.normalize_query(self.task | {'width': 25})

    def test_handle_path_escape_fails(self):
        with self.assertRaises(ValueError):
            b.normalize_query(self.task | {'host_bundle_id': '../escape'})

    def test_bad_intrinsics_fail(self):
        with self.assertRaises(ValueError):
            b.normalize_query(self.task | {'K': [[-12, 0, 12], [0, 12, 8], [0, 0, 1]]})

    def test_all_nine_scene_protocols_resolve(self):
        self.assertEqual(len(b.SCENES), 9)
        self.assertTrue(str(b.scene_protocol('park')).endswith('park/logs_v2/protocol.json'))
        self.assertTrue(str(b.scene_protocol('road')).endswith('road/logs_v2/protocol.json'))
        self.assertTrue(str(b.scene_protocol('block_small')).endswith('logs/full_native_v1/protocol.json'))
        with self.assertRaises(ValueError):
            b.scene_protocol('other')


class ExclusionTests(unittest.TestCase):
    def setUp(self):
        self.pose = np.eye(4).tolist()
        distant = np.eye(4); distant[0, 3] = 4
        self.rows = [
            {'id': 'source', 'name': 's.png', 'sha256': 'raw_source', 'c2w': self.pose},
            {'id': 'same_camera', 'name': 'same.png', 'sha256': 'other', 'c2w': self.pose},
            {'id': 'other', 'name': 'other.png', 'sha256': 'third', 'c2w': distant.tolist()}]
        self.frames = [{'image_sha256': 'png_source', 'training_template_id': 'source'}]

    def test_pixel_binding_excludes_self_and_reference_camera_duplicates(self):
        names = b.source_exclusions({'image_sha256': 'png_source'}, self.frames, self.rows)
        self.assertEqual(names, ['s.png', 'same.png'])

    def test_nonindexed_source_does_not_invent_exclusion(self):
        self.assertEqual(b.source_exclusions({'image_sha256': 'unseen'}, self.frames, self.rows), [])

    def test_missing_pixel_bound_training_reference_fails(self):
        with self.assertRaises(ValueError):
            b.source_exclusions({'image_sha256': 'png_source'}, self.frames, self.rows[1:])

    def test_backend_exclusions_do_not_read_query_pose(self):
        backend = b.RetrievalBackend.__new__(b.RetrievalBackend)
        backend.queries = {'handle': {'image_path': 'unchecked_fixture', 'image_sha256': 'png_source', 'excluded_reference_names': ['s.png']}}
        backend.reference_names = ['s.png', 'other.png']; backend.reference_views = ['v_self', 'v_other']
        backend.training = {r['name']: r for r in self.rows}
        backend.views = {
            'v_self': {'image_sha256': 'png_source', 'frame': {'c2w': self.pose}},
            'v_same_camera_grid': {'image_sha256': 'grid_other', 'frame': {'c2w': self.pose}},
            'v_other': {'image_sha256': 'unrelated', 'frame': {'c2w': self.rows[2]['c2w']}}}
        with mock.patch.object(b, 'checked_sha'):
            backend._prepare_exclusions()
        self.assertEqual(backend.excluded['handle'], {'v_self', 'v_same_camera_grid'})

    def test_source_binding_and_query_mismatch_fail(self):
        backend = b.RetrievalBackend.__new__(b.RetrievalBackend)
        backend.queries = {'one': {}, 'two': {}}; backend.bound_source = None
        backend.bind_source('one')
        with self.assertRaises(ValueError):
            backend._query('two')
        with self.assertRaises(ValueError):
            backend.bind_source('missing')


class NamespaceTests(unittest.TestCase):
    def test_pose_caches_isolated_while_heavy_banks_shared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = b.RetrievalBackend.__new__(b.RetrievalBackend)
            root.workdir = Path(tmp)
            root.scene = 'sf'
            root.cache = root.workdir / 'retrieval_cache' / root.scene
            root.bound_source = 'previous'
            root.views = {'view': object()}
            root.reference_vectors = np.zeros((2, 8), dtype=np.float32)
            root.query_vectors = {'source': np.ones(8, dtype=np.float32)}
            root.queries = {'source': {}}
            root.excluded = {'source': set()}
            root.bm25 = object()
            root.geometry_dir = Path('cohort/rgb')
            normal = root.for_namespace('query_normal')
            self.assertEqual(normal.geometry_dir, Path('cohort/rgb'))
            fast = root.for_namespace('query_fast')
            self.assertNotEqual(normal.cache, fast.cache)
            self.assertEqual(normal.bound_source, None)
            self.assertEqual(fast.bound_source, None)
            for name in ['views', 'reference_vectors', 'query_vectors', 'queries', 'excluded', 'bm25']:
                self.assertIs(getattr(normal, name), getattr(root, name))
                self.assertIs(getattr(fast, name), getattr(root, name))
            normal.bind_source('source')
            self.assertEqual(fast.bound_source, None)
            self.assertEqual(root.bound_source, 'previous')
            key = 'identical_pose_cache_key.json'
            (normal.cache / key).write_text('only normal cached')
            self.assertFalse((fast.cache / key).exists())
            self.assertEqual((root.for_namespace('query_normal').cache / key).read_text(), 'only normal cached')

    def test_namespace_rejects_escape_or_empty(self):
        root = b.RetrievalBackend.__new__(b.RetrievalBackend)
        for namespace in ['', '../escape', 'with/slash', 'with space', 'x' * 121, None]:
            with self.assertRaises(ValueError):
                root.for_namespace(namespace)


class PoseWorkerCohortTests(unittest.TestCase):
    def test_pose_worker_checks_only_bound_source_and_cohort(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source.png'; source.write_bytes(b'pixels')
            confidence = root / 'confidence.py'; confidence.write_text('fixture')
            plan = {'source_sha256': b.sha(b.OLD / 'geometry_adapter.py'), 'confidence_sha256': b.sha(confidence),
                'queries': [{'id':'requested','image_path':str(source),'image_sha256':b.sha(source)},
                            {'id':'other','image_path':str(root/'unread_other_source'),'image_sha256':'not_accessed'}]}
            b.write(root/'plan.json', plan); plan_sha = b.sha(root/'plan.json')
            corr = root/'correspondences/requested.npz'; corr.parent.mkdir(); corr.write_bytes(b'corr')
            base = root/'results/requested.json'
            b.write(base, {'status':'estimated','pose_c2w':np.eye(4).tolist(),'plan_sha256':plan_sha,
                'correspondences_sha256':b.sha(corr),'query_image_sha256':b.sha(source),'reference_count':10})
            b.write(root/'COMPLETED.json', {'status':'complete','plan_sha256':plan_sha,
                'outputs_sha256':{str(base):b.sha(base),str(corr):b.sha(corr)}})
            geometry = SimpleNamespace(CONFIDENCE=confidence, public_result=lambda value,**kw:{'status':value['status'],**kw})
            with mock.patch.object(b,'geometry_module',return_value=geometry):
                result=b.pose_worker('requested',None,root)
                self.assertEqual(result['status'],'estimated');self.assertTrue(result['cache_hit'])
                with self.assertRaisesRegex(ValueError,'Unknown host'):b.pose_worker('absent',None,root)
                corr.write_bytes(b'tampered')
                with self.assertRaises(ValueError):b.pose_worker('requested',None,root)


if __name__ == '__main__':
    unittest.main()
