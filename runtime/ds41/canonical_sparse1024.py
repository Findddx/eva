"""Explicit1024 target adapter over the immutable isolated primitive sources.

No production caller selects this module except the exact1024 owner. Import
uses stdlib only. Source preservation is not a full-model GPU certificate.
"""
import hashlib
from pathlib import Path
from engine.experiments.canonical1024 import sparse1024 as _core

SOURCE_ROOT = Path(__file__).resolve().parents[1]
MAX_QUERIES, MAX_KEYS, MAX_KV_ROWS = _core.MAX_QUERIES, _core.MAX_KEYS, _core.MAX_KV_ROWS
query_spans, workspace_metadata = _core.query_spans, _core.workspace_metadata


def source_metadata():
    return dict(_core.source_metadata(SOURCE_ROOT),
        production_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        primitive_build_identity_unchanged=True, production_GPU_certificate=False)


def build_backend(*, verbose=False):
    return _core.build_backend(SOURCE_ROOT, verbose=verbose)


class CanonicalSparse1024Runner(_core.CanonicalSparse1024Runner):
    def __init__(self, device='cuda', query_tile=32, max_queries=1024, *, verbose=False):
        super().__init__(device, query_tile, max_queries, source=SOURCE_ROOT, verbose=verbose)
        self.metadata = source_metadata()
