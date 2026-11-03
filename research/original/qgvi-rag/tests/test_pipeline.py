import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from caption_rag import pipeline as p


def caption():
    return dict(summary='A red brick building beside a tree-lined road.', objects=['building', 'trees'],
                attributes=['red brick'], relations=['trees beside the road'], visible_text=[],
                uncertainty=['the distant structure might be a fountain'])


def verification(verdict='match', status='supported'):
    return dict(verdict=verdict, score=0.9, evidence='A red brick facade is visible.',
                requirements=[dict(requirement='red brick building', status=status,
                                   evidence='Visible facade')])


class FakeVision:
    def __init__(self, fail=None, verdict='match', status='supported'):
        self.captions = 0
        self.checks = 0
        self.fail = fail
        self.verdict, self.status = verdict, status

    def caption(self, image_path):
        self.captions += 1
        if self.fail is not None and self.captions == self.fail:
            raise RuntimeError('simulated service failure')
        return caption()

    def verify(self, image_path, query):
        self.checks += 1
        return verification(self.verdict, self.status)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'manifest.json'
        self.output = self.root / 'output'
        rows = []
        for i in range(6):
            image = self.root / f'image-{i}.png'
            Image.new('RGB', (80, 64), (100 + i, 0, 0)).save(image)
            rows.append(dict(id=f'view-{i}', split='reference', image_path=image.name,
                             place_id=f'place-{i // 2}', c2w=[[1, 0], [0, 1]]))
        rows.append(dict(id='held-out', split='test', image_path='does-not-exist.png'))
        p.atomic_json(self.manifest, dict(frames=rows))
        self.settings = dict(model='fake', base_url='https://example.invalid', max_side=1280)

    def prepare(self):
        frames = p.load_frames(self.manifest)
        binding = p.run_binding(self.manifest, frames, self.settings)
        run_id = p.bind_run(self.output, binding, frames)
        return frames, binding, run_id

    def test_reference_only_and_relative_paths(self):
        frames = p.load_frames(self.manifest, limit=3)
        self.assertEqual([f['id'] for f in frames], ['view-0', 'view-2', 'view-5'])
        self.assertTrue(all(Path(f['image_path']).is_absolute() for f in frames))
        self.assertTrue(all(len(f['image_sha256']) == 64 for f in frames))

    def test_missing_split_or_duplicate_id_rejected(self):
        data = p.read_json(self.manifest)
        del data['frames'][0]['split']
        p.atomic_json(self.manifest, data)
        with self.assertRaisesRegex(ValueError, 'explicit split'):
            p.load_frames(self.manifest)
        data['frames'][0]['split'] = 'reference'
        data['frames'][1]['id'] = data['frames'][0]['id']
        p.atomic_json(self.manifest, data)
        with self.assertRaisesRegex(ValueError, 'unique'):
            p.load_frames(self.manifest)

    def test_caption_resume_and_complete_chunks(self):
        frames, _, run_id = self.prepare()
        client = FakeVision()
        p.caption_all(self.output, frames, run_id, lambda: client, workers=1)
        self.assertEqual(client.captions, 6)
        p.caption_all(self.output, frames, run_id, lambda: self.fail('Must not create API client'), workers=1)
        chunks = p.collect_chunks(self.output, frames, run_id)
        self.assertEqual(len(chunks), 6)
        self.assertNotIn('fountain', chunks[0]['text'])
        self.assertEqual(chunks[0]['view']['c2w'], frames[0]['c2w'])

    def test_failure_preserves_successes_and_resume_only_missing(self):
        frames, _, run_id = self.prepare()
        client = FakeVision(fail=3)
        with self.assertRaisesRegex(RuntimeError, 'resume'):
            p.caption_all(self.output, frames, run_id, lambda: client, workers=1)
        self.assertEqual(len(list((self.output / 'captions').glob('*.json'))), 2)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            p.collect_chunks(self.output, frames, run_id)
        next_client = FakeVision()
        p.caption_all(self.output, frames, run_id, lambda: next_client, workers=1)
        self.assertEqual(next_client.captions, 4)

    def test_changed_image_refuses_cached_run(self):
        frames, _, _ = self.prepare()
        Image.new('RGB', (80, 64), 'blue').save(frames[0]['image_path'])
        changed = p.load_frames(self.manifest)
        with self.assertRaisesRegex(ValueError, 'changed'):
            p.bind_run(self.output, p.run_binding(self.manifest, changed, self.settings), changed)

    def test_changed_caption_settings_refuse_cached_run(self):
        frames, binding, _ = self.prepare()
        binding['caption_settings']['model'] = 'different-model'
        with self.assertRaisesRegex(ValueError, 'changed'):
            p.bind_run(self.output, binding, frames)

    def test_output_lock_blocks_second_writer(self):
        with p.output_lock(self.output):
            with self.assertRaisesRegex(ValueError, 'Another process'):
                with p.output_lock(self.output):
                    pass

    def fake_retriever(self, frames):
        class Retriever:
            def __init__(self, *args, **kwargs):
                self.chunks = {f['id']: f for f in frames}

            def search(self, query, top_k=100):
                return [dict(view_id=f['id'], score=1 / (i + 1), semantic_score=0.5,
                             lexical_score=0.5, matched_sentences=['red brick building'])
                        for i, f in enumerate(frames[:top_k])]
        return Retriever

    def prepared_query(self, client, **kwargs):
        frames, _, run_id = self.prepare()
        p.atomic_json(self.output / 'index/run_binding.json', dict(run_id=run_id))
        with patch('caption_rag.retrieval.CaptionRetriever', self.fake_retriever(frames)):
            summary = p.query_views(self.output, frames, 'red brick building', lambda: client,
                                    candidates=6, verify_k=4, top_k=3, **kwargs)
        return frames, summary, p.read_json(summary['result_json'])

    def test_verified_results_place_diversity_and_cache(self):
        client = FakeVision()
        frames, summary, result = self.prepared_query(client)
        self.assertEqual(client.checks, 4)
        self.assertEqual(summary['matches'], 3)
        self.assertEqual(len({r['view']['place_id'] for r in result['results']}), 3)
        self.assertTrue(Path(summary['result_html']).exists())
        with patch('caption_rag.retrieval.CaptionRetriever', self.fake_retriever(frames)):
            p.query_views(self.output, frames, 'red brick building',
                          lambda: self.fail('Cached verification must not call API'),
                          candidates=6, verify_k=4, top_k=3)

    def test_unclear_requirement_cannot_be_accepted(self):
        _, summary, result = self.prepared_query(FakeVision(status='unclear'))
        self.assertEqual(summary['status'], 'no_verified_match_within_budget')
        self.assertEqual(result['results'], [])

    def test_unverified_mode_is_explicit_and_never_calls_vision(self):
        client = FakeVision()
        _, summary, result = self.prepared_query(client, verify=False)
        self.assertEqual(client.checks, 0)
        self.assertEqual(summary['status'], 'unverified_candidates')
        self.assertEqual(result['inspected'], [])

    def test_tampered_caption_identity_rejected(self):
        frames, _, run_id = self.prepare()
        p.caption_all(self.output, frames, run_id, FakeVision, workers=1)
        path = p.caption_path(self.output, frames[0]['id'])
        data = p.read_json(path)
        data['view_id'] = 'different-view'
        p.atomic_json(path, data)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            p.collect_chunks(self.output, frames, run_id)

    def test_malformed_cached_verification_rejected(self):
        frames, summary, _ = self.prepared_query(FakeVision())
        cache = next((Path(summary['result_json']).parent / 'verification').glob('*.json'))
        record = p.read_json(cache)
        record['verification'] = dict(verdict='match', score=1, requirements=[dict(status='supported')])
        p.atomic_json(cache, record)
        with patch('caption_rag.retrieval.CaptionRetriever', self.fake_retriever(frames)):
            with self.assertRaisesRegex(RuntimeError, 'schema'):
                p.query_views(self.output, frames, 'red brick building', FakeVision,
                              candidates=6, verify_k=4, top_k=3)

    def test_index_rebuild_retains_captions_and_previous_index(self):
        frames, _, run_id = self.prepare()
        p.caption_all(self.output, frames, run_id, FakeVision, workers=1)
        chunks = p.collect_chunks(self.output, frames, run_id)
        upstream = self.root / 'arag'
        (upstream / 'scripts').mkdir(parents=True)
        (upstream / 'scripts/build_index.py').write_text('# fixture')

        def builder(chunks, output_dir, arag_root, model_name, **kwargs):
            (output_dir / 'model.txt').write_text(model_name)
            return {}

        with patch('caption_rag.retrieval.build_index', builder):
            p.ensure_index(self.output, chunks, run_id, upstream, 'first', 'cpu', 32)
            with self.assertRaisesRegex(ValueError, '--rebuild-index'):
                p.ensure_index(self.output, chunks, run_id, upstream, 'second', 'cpu', 32)
            p.ensure_index(self.output, chunks, run_id, upstream, 'second', 'cpu', 32, rebuild=True)
        self.assertEqual((self.output / 'index/model.txt').read_text(), 'second')
        backup = next(self.output.glob('index-backup-*'))
        self.assertEqual((backup / 'model.txt').read_text(), 'first')
        p.caption_all(self.output, frames, run_id,
                      lambda: self.fail('Index rebuild must preserve caption cache'), workers=1)

    def test_invalid_query_rejected_before_reading_inputs(self):
        from caption_rag.cli import main
        with patch('caption_rag.cli.load_frames', side_effect=AssertionError('too late')):
            with self.assertRaises(SystemExit) as exc:
                main(['--query', 'x' * 16001])
        self.assertEqual(exc.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
