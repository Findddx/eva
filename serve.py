"""Start eva's persistent model worker and streaming HTTP API."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'pack', 'dense-cache', 'engram-manifest'):
        parser.add_argument('--'+name, type=Path,
                            default=os.environ.get('EVA_'+name.upper().replace('-', '_')))
    parser.add_argument('--counts', type=Path, default=os.environ.get('EVA_COUNTS'))
    parser.add_argument('--jit-dir', type=Path, default=ROOT/'.cache/torch_extensions')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--context', type=int, default=131072)
    parser.add_argument('--max-output-tokens', type=int, default=4096)
    parser.add_argument('--mode', choices=('greedy', 'lookup'), default='greedy')
    parser.add_argument('--ram-gib', type=float, default=64)
    parser.add_argument('--host-headroom-gib', type=float, default=8)
    parser.add_argument('--host-allocator-reserve-gib', type=float, default=1)
    parser.add_argument('--max-queue', type=int, default=2)
    args = parser.parse_args(argv)
    for name in ('model', 'pack', 'dense_cache', 'engram_manifest'):
        if getattr(args, name) is None:
            parser.error('Required: --'+name.replace('_', '-')+' or its EVA_ environment variable')
    worker = [sys.executable, '-u', '-B', str(ROOT/'engine_worker.py'),
              '--source', str(ROOT/'runtime')]
    for name in ('model', 'pack', 'dense_cache', 'engram_manifest', 'jit_dir',
                 'context', 'max_output_tokens', 'mode', 'ram_gib',
                 'host_headroom_gib', 'host_allocator_reserve_gib'):
        worker += ['--'+name.replace('_', '-'), str(getattr(args, name))]
    if args.counts is not None:
        worker += ['--counts', str(args.counts)]
    from server import main as serve
    return serve(['--host', args.host, '--port', str(args.port), '--max-queue', str(args.max_queue),
                  '--worker', *worker])


if __name__ == '__main__':
    raise SystemExit(main())
