"""Exact1024 target adapter; isolated device arithmetic remains unchanged."""
import hashlib
from pathlib import Path
from engine.experiments.canonical1024 import rope1024 as _core

SOURCE_ROOT = Path(__file__).resolve().parents[1]
MAX_QUERIES, workspace_metadata = _core.MAX_QUERIES, _core.workspace_metadata


def source_metadata():
    return dict(_core.source_metadata(SOURCE_ROOT),
        production_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        primitive_build_identity_unchanged=True, production_GPU_certificate=False)


def build_backend(*, verbose=False):
    return _core.build_backend(SOURCE_ROOT, verbose=verbose)


class CanonicalRope1024Runner(_core.CanonicalRope1024Runner):
    def __init__(self, device='cuda', max_queries=1024, *, verbose=False):
        super().__init__(device, max_queries, source=SOURCE_ROOT, verbose=verbose)
        self.metadata = source_metadata()
