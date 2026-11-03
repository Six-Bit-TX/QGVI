"""Deterministic scheduler checks using local files and fake image clients."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from caption_rag import pipeline as p


def caption():
    return dict(summary='A brick building beside a road.', objects=['building'],
                attributes=['brick'], relations=['beside a road'], visible_text=[],
                uncertainty=[])


class CaptionSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'output'
        self.frames = []
        for number in range(12):
            path = self.root / f'view-{number}'
            path.write_bytes(f'image-{number}'.encode())
            self.frames.append(dict(id=path.name, image_path=str(path),
                                    image_sha256=p.file_hash(path)))

    def run_captions(self, client, count=6, workers=2):
        p.caption_all(self.output, self.frames[:count], 'run', lambda: client, workers)

    def test_replaces_fast_request_while_slow_request_is_running(self):
        slow_started = threading.Event()
        release_slow = threading.Event()
        slow_finished = threading.Event()
        replacement_overlapped = []

        class Client:
            def caption(self, image_path):
                name = Path(image_path).name
                if name == 'view-0':
                    slow_started.set()
                    if not release_slow.wait(3):
                        raise RuntimeError('rolling replacement never started')
                    slow_finished.set()
                elif name == 'view-1':
                    if not slow_started.wait(3):
                        raise RuntimeError('slow request never started')
                elif name == 'view-2':
                    replacement_overlapped.append(not slow_finished.is_set())
                    release_slow.set()
                return caption()

        self.run_captions(Client())
        self.assertEqual(replacement_overlapped, [True])
        self.assertEqual(len(list((self.output / 'captions').glob('*.json'))), 6)

    def test_failure_stops_refills_and_drains_successful_active_request(self):
        active_started = threading.Event()
        release_active = threading.Event()
        calls = []
        real_wait = p.wait

        class Client:
            def caption(self, image_path):
                name = Path(image_path).name
                calls.append(name)
                if name == 'view-0':
                    if not active_started.wait(3):
                        raise RuntimeError('second request never started')
                    raise RuntimeError('simulated schema failure')
                active_started.set()
                if not release_active.wait(3):
                    raise RuntimeError('active request was not drained')
                return caption()

        def observe_failure(*args, **kwargs):
            result = real_wait(*args, **kwargs)
            if any(future.exception() is not None for future in result[0]):
                release_active.set()
            return result

        with patch.object(p, 'wait', side_effect=observe_failure):
            with self.assertRaisesRegex(RuntimeError, 'view-0.*successful captions are saved'):
                self.run_captions(Client())
        self.assertCountEqual(calls, ['view-0', 'view-1'])
        self.assertIsNone(p.checked_caption(self.output, self.frames[0], 'run'))
        self.assertIsNotNone(p.checked_caption(self.output, self.frames[1], 'run'))

    def test_inflight_window_and_progress_are_bounded(self):
        workers = 3
        barrier = threading.Barrier(workers, timeout=3)
        lock = threading.Lock()
        active = peak_active = peak_live = 0
        submitted = []

        class RecordingPool(ThreadPoolExecutor):
            def submit(self, *args, **kwargs):
                nonlocal peak_live
                future = super().submit(*args, **kwargs)
                submitted.append(future)
                peak_live = max(peak_live, sum(not item.done() for item in submitted))
                return future

        class Client:
            def caption(self, image_path):
                nonlocal active, peak_active
                with lock:
                    active += 1
                    peak_active = max(peak_active, active)
                try:
                    if int(Path(image_path).name.split('-')[1]) < workers:
                        barrier.wait()
                    return caption()
                finally:
                    with lock:
                        active -= 1

        progress = io.StringIO()
        with patch.object(p, 'ThreadPoolExecutor', RecordingPool), redirect_stderr(progress):
            self.run_captions(Client(), count=12, workers=workers)
        self.assertEqual(peak_active, workers)
        self.assertLessEqual(peak_live, workers)
        self.assertEqual(len(submitted), 12)
        self.assertLessEqual(len(progress.getvalue().splitlines()), 6)
        self.assertIn('Captions: 12/12 saved.', progress.getvalue())

    def test_interrupt_preserves_active_saves_without_scheduling_more(self):
        both_started = threading.Event()
        release_active = threading.Event()
        lock = threading.Lock()
        calls = []

        class Client:
            def caption(self, image_path):
                with lock:
                    calls.append(Path(image_path).name)
                    if len(calls) == 2:
                        both_started.set()
                if not release_active.wait(3):
                    raise RuntimeError('interrupted active request was not drained')
                return caption()

        def interrupt(*args, **kwargs):
            self.assertTrue(both_started.wait(3))
            release_active.set()
            raise KeyboardInterrupt()

        with patch.object(p, 'wait', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_captions(Client())
        self.assertCountEqual(calls, ['view-0', 'view-1'])
        for frame in self.frames[:2]:
            self.assertIsNotNone(p.checked_caption(self.output, frame, 'run'))


if __name__ == '__main__':
    unittest.main()
