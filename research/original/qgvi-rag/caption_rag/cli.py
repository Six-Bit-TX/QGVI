"""One-command caption, index, retrieve, and verify workflow."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .pipeline import (DeepSeekVision, bind_run, caption_all, collect_chunks, ensure_index,
                       load_frames, output_lock, query_views, run_binding)

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT.parent / 'qgvi/.staging/streaming_reference_v1/full_grid_v1/manifest.json'


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--query', help='Text description to retrieve; omit to build the caption index only.')
    cli.add_argument('--env-file', type=Path, help='Read only the DeepSeek API key from a local dotenv, JSON, or labelled key file; never execute its contents.')
    cli.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    cli.add_argument('--output', type=Path, default=PROJECT / 'outputs/block_a_deepseek')
    cli.add_argument('--arag-root', type=Path, default=Path.home() / 'others/arag-upstream')
    cli.add_argument('--model', default=os.environ.get('DEEPSEEK_MODEL', 'deepseek-flash'))
    cli.add_argument('--base-url', default=os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'))
    cli.add_argument('--embedding-model', default='sentence-transformers/all-MiniLM-L6-v2')
    cli.add_argument('--device', default='cpu', help='Embedding device; captions/verification use the API.')
    cli.add_argument('--stage', choices=('all', 'caption', 'index', 'query'), default='all')
    cli.add_argument('--rebuild-index', action='store_true', help='Rebuild embeddings using saved captions; preserve the previous index as a backup.')
    cli.add_argument('--limit', type=positive, help='Caption a deterministic spread of N references; use a separate --output.')
    cli.add_argument('--workers', type=positive, default=4)
    cli.add_argument('--batch-size', type=positive, default=32)
    cli.add_argument('--max-side', type=positive, default=1280)
    cli.add_argument('--timeout', type=positive, default=90)
    cli.add_argument('--retries', type=int, default=3)
    cli.add_argument('--candidates', type=positive, default=100)
    cli.add_argument('--verify-k', type=positive, default=20, help='Maximum images visually checked per query.')
    cli.add_argument('--top-k', type=positive, default=5)
    cli.add_argument('--no-verify', action='store_true', help='Return explicitly unverified retrieval candidates.')
    cli.add_argument('--check', action='store_true', help='Check inputs and report readiness without API calls or output writes.')
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if not 0 <= args.retries <= 8:
        cli.error('--retries must be between 0 and 8')
    if not 64 <= args.max_side <= 8192:
        cli.error('--max-side must be between 64 and 8192')
    if args.query is not None and not args.query.strip():
        cli.error('--query must not be blank')
    if args.query is not None and len(args.query) > 16000:
        cli.error('--query must be at most 16000 characters')
    if args.rebuild_index and args.stage not in ('all', 'index'):
        cli.error('--rebuild-index requires --stage all or --stage index')
    if not args.model.strip() or not args.embedding_model.strip():
        cli.error('Model names must not be blank')
    if args.stage == 'query' and not args.query:
        cli.error('--stage query requires --query')
    if args.verify_k > args.candidates or args.top_k > args.candidates:
        cli.error('--verify-k and --top-k must not exceed --candidates')
    if not args.no_verify and args.query and args.top_k > args.verify_k:
        cli.error('--top-k must not exceed --verify-k when visual verification is enabled')
    try:
        args.base_url = DeepSeekVision._validate_url(args.base_url)
        if args.env_file is not None:
            from .credentials import load_deepseek_key
            os.environ['DEEPSEEK_API_KEY'] = load_deepseek_key(args.env_file)
        args.manifest = args.manifest.expanduser().resolve(strict=True)
        args.output = args.output.expanduser().resolve()
        args.arag_root = args.arag_root.expanduser().resolve(strict=True)
        builder = args.arag_root / 'scripts/build_index.py'
        if not builder.is_file():
            raise ValueError(f'A-RAG builder missing: {builder}')
        frames = load_frames(args.manifest, args.limit, hash_images=not args.check)
        if args.check:
            print(json.dumps(dict(reference_views=len(frames), manifest=str(args.manifest),
                                  output=str(args.output), arag_builder=str(builder),
                                  model=args.model, embedding_model=args.embedding_model,
                                  api_key_configured=bool(os.environ.get('DEEPSEEK_API_KEY')),
                                  api_calls=0), indent=2))
            return 0
        settings = dict(model=args.model, base_url=args.base_url.rstrip('/'), max_side=args.max_side)
        binding = run_binding(args.manifest, frames, settings)

        def client_factory():
            return DeepSeekVision(**settings, timeout=args.timeout, retries=args.retries)

        with output_lock(args.output):
            run_id = bind_run(args.output, binding, frames)
            if args.stage in ('all', 'caption'):
                caption_all(args.output, frames, run_id, client_factory, args.workers)
            if args.stage in ('all', 'index'):
                chunks = collect_chunks(args.output, frames, run_id)
                ensure_index(args.output, chunks, run_id, args.arag_root, args.embedding_model,
                             args.device, args.batch_size, rebuild=args.rebuild_index)
            if args.query and args.stage in ('all', 'query'):
                summary = query_views(args.output, frames, args.query.strip(), client_factory,
                                      candidates=args.candidates, verify_k=args.verify_k,
                                      top_k=args.top_k, device=args.device, verify=not args.no_verify,
                                      vision_settings=settings)
            else:
                summary = dict(status='captions_ready' if args.stage == 'caption' else 'index_ready',
                               reference_views=len(frames), output=str(args.output))
        print(json.dumps(summary, indent=2))
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Saved captions and visual checks will be reused on the next run.', file=sys.stderr)
        return 130
    except Exception as exc:
        # Do not leak a credential if a third-party exception happens to contain it.
        message = str(exc)
        key = os.environ.get('DEEPSEEK_API_KEY')
        if key:
            message = message.replace(key, '[redacted]')
        print(f'Error: {message}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
