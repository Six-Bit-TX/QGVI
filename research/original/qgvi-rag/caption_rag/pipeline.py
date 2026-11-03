"""Resumable reference captioning, A-RAG indexing, and visual verification."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from .vision import (CAPTION_PROMPT, VERIFY_PROMPT, DeepSeekVision,
                     validate_caption, validate_verification)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def output_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'Another process is using {output}; wait for it to finish.') from None
        yield


def load_frames(manifest, limit=None, *, hash_images=True):
    """Only explicit reference splits are eligible; paths are manifest-relative."""
    manifest = Path(manifest).expanduser().resolve(strict=True)
    data = read_json(manifest)
    if not isinstance(data, dict) or not isinstance(data.get('frames'), list):
        raise ValueError('Expected a QGVI manifest containing a frames list.')
    frames, seen = [], set()
    for source in data['frames']:
        if not isinstance(source, dict) or 'split' not in source:
            raise ValueError('Every manifest frame must have an explicit split.')
        if source['split'] != 'reference':
            continue
        view_id = source.get('id')
        if not isinstance(view_id, str) or not view_id or view_id in seen:
            raise ValueError('Reference IDs must be unique nonempty strings.')
        seen.add(view_id)
        if not isinstance(source.get('image_path'), str) or not source['image_path']:
            raise ValueError(f'Reference {view_id} has no image_path.')
        frame = {key: source[key] for key in ('id', 'c2w', 'K', 'width', 'height',
                                             'domain', 'place_id') if key in source}
        image_path = Path(source['image_path']).expanduser()
        if not image_path.is_absolute():
            image_path = manifest.parent / image_path
        frame['image_path'] = str(image_path.resolve(strict=True))
        if not image_path.is_file():
            raise ValueError(f'Reference image is not a file: {image_path}')
        frame['place_id'] = frame.get('place_id') or view_id
        frames.append(frame)
    if not frames:
        raise ValueError('Manifest contains no reference views.')
    if limit is not None:
        if limit < 1:
            raise ValueError('--limit must be positive.')
        if limit < len(frames):
            # Spread pilots over manifest order instead of taking one spatial block.
            indices = [0] if limit == 1 else [i * (len(frames) - 1) // (limit - 1)
                                               for i in range(limit)]
            frames = [frames[i] for i in indices]
    if hash_images:
        print(f'Checking image content for {len(frames)} reference views...', file=sys.stderr)
        with ThreadPoolExecutor(max_workers=4) as pool:
            hashes = pool.map(file_hash, (frame['image_path'] for frame in frames))
            for frame, sha in zip(frames, hashes):
                frame['image_sha256'] = sha
    return frames


def run_binding(manifest, frames, settings):
    return dict(schema_version=1, manifest=str(Path(manifest).resolve()),
                manifest_sha256=file_hash(manifest), frames_sha256=digest(frames),
                count=len(frames), caption_settings=settings,
                caption_prompt_sha256=digest(CAPTION_PROMPT))


def bind_run(output, binding, frames):
    target = Path(output) / 'run.json'
    if target.exists():
        existing = read_json(target)
        if existing.get('binding') != binding or existing.get('frames') != frames:
            raise ValueError('Reference images, manifest, or caption settings changed. '
                             'Choose a new --output directory; cached captions were preserved.')
    else:
        if (Path(output) / 'captions').exists() or (Path(output) / 'index').exists():
            raise ValueError('Output has artifacts without run.json; choose a new --output.')
        atomic_json(target, dict(binding=binding, frames=frames))
    return digest(binding)


def caption_path(output, view_id):
    return Path(output) / 'captions' / (digest(view_id) + '.json')


def checked_caption(output, frame, run_id):
    path = caption_path(output, frame['id'])
    if not path.exists():
        return None
    record = read_json(path)
    if (record.get('view_id') != frame['id'] or record.get('run_id') != run_id
            or record.get('image_sha256') != frame['image_sha256']):
        raise ValueError(f'Caption cache identity mismatch for {frame["id"]}.')
    record['caption'] = validate_caption(record['caption'])
    return record


def caption_text(caption):
    summary = caption.get('summary')
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError('Caption must contain a nonempty summary.')
    parts = [summary.strip()]
    for name in ('objects', 'attributes', 'relations', 'visible_text', 'uncertainty'):
        items = caption.get(name)
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            raise ValueError(f'Caption {name} must be a list of strings.')
        # Speculative objects in uncertainty must not become positive search terms.
        if items and name != 'uncertainty':
            parts.append(name.replace('_', ' ').capitalize() + ': ' + '; '.join(items))
    return '\n'.join(parts)


def caption_all(output, frames, run_id, client_factory, workers=4):
    pending = [frame for frame in frames if checked_caption(output, frame, run_id) is None]
    completed = len(frames) - len(pending)
    print(f'Captions: {completed}/{len(frames)} cached, {len(pending)} pending.', file=sys.stderr)
    if not pending:
        return
    client = client_factory()

    def work(frame):
        # Catch edits since preflight before sending pixels to the API.
        if file_hash(frame['image_path']) != frame['image_sha256']:
            raise ValueError(f'Image changed during captioning: {frame["id"]}')
        caption = validate_caption(client.caption(frame['image_path']))
        caption_text(caption)
        if file_hash(frame['image_path']) != frame['image_sha256']:
            raise ValueError(f'Image changed during captioning: {frame["id"]}')
        atomic_json(caption_path(output, frame['id']), dict(view_id=frame['id'],
                    image_sha256=frame['image_sha256'], run_id=run_id, caption=caption))

    # Keep a rolling window, never a full-bank queue. Stop refilling as soon as
    # any completed request reports a failure; already running work still saves.
    pool = ThreadPoolExecutor(max_workers=workers)
    futures, failures = {}, []
    remaining = iter(pending)
    last_reported, last_report_at = completed, time.monotonic()
    reported = False

    def report(final=False):
        nonlocal last_reported, last_report_at, reported
        now = time.monotonic()
        if ((final and (not reported or completed != last_reported))
                or completed - last_reported >= workers or now - last_report_at >= 5):
            print(f'Captions: {completed}/{len(frames)} saved.', file=sys.stderr)
            last_reported, last_report_at, reported = completed, now, True

    def refill():
        while len(futures) < workers:
            frame = next(remaining, None)
            if frame is None:
                break
            futures[pool.submit(work, frame)] = frame

    try:
        refill()
        while futures:
            done, _ = wait(futures, timeout=max(0, 5 - (time.monotonic() - last_report_at)),
                           return_when=FIRST_COMPLETED)
            # Inspect the whole completed set before refilling: a success must
            # not schedule more paid work while another result reports failure.
            for future in done:
                frame = futures.pop(future)
                if future.cancelled():
                    continue
                try:
                    future.result()
                    completed += 1
                except Exception as exc:
                    failures.append((frame['id'], exc))
            if failures:
                for future in futures:
                    future.cancel()
            else:
                refill()
            report()
    finally:
        # On KeyboardInterrupt, discard queued work and let the bounded set of
        # running requests finish its atomic saves before releasing the run lock.
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        for future in futures:
            if not future.cancelled() and future.done() and future.exception() is None:
                completed += 1
        report(final=True)
    if failures:
        view_id, error = failures[0]
        raise RuntimeError(f'Captioning stopped at view {view_id}; successful captions are saved. '
                           f'Rerun the same command to resume. {error}') from error


def collect_chunks(output, frames, run_id):
    chunks = []
    for frame in frames:
        record = checked_caption(output, frame, run_id)
        if record is None:
            raise ValueError('Caption bank is incomplete. Run --stage caption or --stage all first.')
        chunks.append(dict(id=frame['id'], text=caption_text(record['caption']),
                           view={k: v for k, v in frame.items() if k != 'image_sha256'},
                           image_sha256=frame['image_sha256']))
    return chunks


def ensure_index(output, chunks, run_id, arag_root, model_name, device, batch_size,
                 rebuild=False):
    from .retrieval import build_index
    output = Path(output)
    target = output / 'index'
    binding = dict(run_id=run_id, chunks_sha256=digest(chunks), model_name=model_name,
                   arag_builder_sha256=file_hash(Path(arag_root) / 'scripts' / 'build_index.py'))
    if target.exists() and not rebuild:
        if not (target / 'run_binding.json').is_file() or read_json(target / 'run_binding.json') != binding:
            raise ValueError('Index settings or caption content changed. Use --rebuild-index '
                             'to rebuild embeddings while retaining all saved captions.')
        return target
    staging = Path(tempfile.mkdtemp(prefix='.index-', dir=output))
    try:
        build_index(chunks, staging, Path(arag_root), model_name, device=device, batch_size=batch_size)
        atomic_json(staging / 'run_binding.json', binding)
        backup = None
        if target.exists():
            backup = Path(tempfile.mkdtemp(prefix='index-backup-', dir=output))
            backup.rmdir()
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except BaseException:
            if backup is not None and not target.exists():
                os.replace(backup, target)
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def diverse_order(candidates, frames):
    first, rest, seen = [], [], set()
    for candidate in candidates:
        place = frames[candidate['view_id']]['place_id']
        (rest if place in seen else first).append(candidate)
        seen.add(place)
    return first + rest


def verified_match(verification):
    requirements = verification.get('requirements')
    return (verification.get('verdict') == 'match' and isinstance(requirements, list)
            and bool(requirements) and all(r.get('status') == 'supported' for r in requirements))


def query_views(output, frames, query, client_factory, *, candidates=100, verify_k=20,
                top_k=5, device='cpu', verify=True, vision_settings=None):
    from .retrieval import CaptionRetriever
    index_dir = Path(output) / 'index'
    binding = read_json(index_dir / 'run_binding.json')
    if binding['run_id'] != digest(read_json(Path(output) / 'run.json')['binding']):
        raise ValueError('Index does not belong to this caption run.')
    retriever = CaptionRetriever(index_dir, device=device)
    by_id = {frame['id']: frame for frame in frames}
    if set(retriever.chunks) != set(by_id):
        raise ValueError('A-RAG index membership differs from the reference manifest.')
    settings = dict(query=query, candidates=candidates, verify_k=verify_k, top_k=top_k,
                    verify=verify, vision_settings=vision_settings,
                    verifier_prompt_sha256=digest(VERIFY_PROMPT), index_binding=binding)
    query_dir = Path(output) / 'queries' / digest(settings)[:24]
    query_dir.mkdir(parents=True, exist_ok=True)
    ranking = retriever.search(query, top_k=candidates)
    ordered = diverse_order(ranking, by_id)
    inspected, results = [], []
    if verify:
        client = None
        for number, candidate in enumerate(ordered[:verify_k], 1):
            frame = by_id[candidate['view_id']]
            if file_hash(frame['image_path']) != frame['image_sha256']:
                raise ValueError(f'Image changed since indexing: {frame["id"]}')
            cache = query_dir / 'verification' / (digest(frame['id']) + '.json')
            if cache.exists():
                record = read_json(cache)
                if (record.get('settings_sha256') != digest(settings)
                        or record.get('image_sha256') != frame['image_sha256']
                        or record.get('view_id') != frame['id']):
                    raise ValueError('Visual verification cache identity mismatch.')
                verification = validate_verification(record['verification'])
            else:
                if client is None:
                    client = client_factory()
                verification = validate_verification(client.verify(frame['image_path'], query))
                if file_hash(frame['image_path']) != frame['image_sha256']:
                    raise ValueError(f'Image changed during verification: {frame["id"]}')
                atomic_json(cache, dict(settings_sha256=digest(settings), view_id=frame['id'],
                            image_sha256=frame['image_sha256'], verification=verification))
            row = dict(candidate, view=frame, verification=verification)
            inspected.append(row)
            if verified_match(verification):
                results.append(row)
            print(f'Visual check {number}/{min(verify_k, len(ordered))}: '
                  f'{verification["verdict"]}', file=sys.stderr)
        results.sort(key=lambda row: (-row['verification']['score'], -row['score'], row['view_id']))
    else:
        results = [dict(candidate, view=by_id[candidate['view_id']]) for candidate in ordered]
    chosen, places = [], set()
    for row in results:
        place = row['view']['place_id']
        if place not in places:
            places.add(place)
            chosen.append(row)
        if len(chosen) >= top_k:
            break
    status = ('unverified_candidates' if not verify else
              'verified_matches' if chosen else 'no_verified_match_within_budget')
    result = dict(schema_version=1, query=query, status=status, settings=settings,
                  results=chosen, retrieved=ranking, inspected=inspected,
                  note='Ranks and model scores are not calibrated probabilities. '
                       'Results identify reference cameras; multiple views may satisfy a query.')
    atomic_json(query_dir / 'result.json', result)
    write_html(query_dir / 'result.html', result)
    return dict(status=status, matches=len(chosen), inspected=len(inspected),
                result_json=str(query_dir / 'result.json'), result_html=str(query_dir / 'result.html'))


def write_html(path, result):
    cards = []
    for row in result['results']:
        view = row['view']
        evidence = row.get('verification', {}).get('evidence', 'Visual verification disabled.')
        cards.append('<article><h2>' + html.escape(row['view_id']) + '</h2><img src="'
                     + html.escape(Path(view['image_path']).as_uri(), quote=True) + '"><p>'
                     + html.escape(evidence) + '</p></article>')
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Reference views</title>'
                '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px}'
                'article{display:inline-block;vertical-align:top;width:46%;margin:1%;overflow-wrap:anywhere}'
                'img{max-width:100%}h2{font-size:14px}</style><h1>' + html.escape(result['query'])
                + '</h1><p>' + html.escape(result['status']) + '</p>' + ''.join(cards) + '</html>')
    Path(path).write_text(document, encoding='utf-8')
