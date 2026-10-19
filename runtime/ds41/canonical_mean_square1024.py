"""Exact1024 target adapter; original isolated M1 reduction tree is immutable."""
import hashlib
from pathlib import Path
from engine.experiments.canonical1024 import mean1024 as _core

SOURCE_ROOT = Path(__file__).resolve().parents[1]
WIDTHS, MAX_ROWS, m1_geometry = _core.WIDTHS, _core.MAX_ROWS, _core.m1_geometry
validate_installed_torch = _core.validate_installed_torch


def source_metadata():
    return dict(_core.source_metadata(SOURCE_ROOT),
        production_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        primitive_build_identity_unchanged=True, production_GPU_certificate=False)


def build_backend(*, verbose=False):
    return _core.build_backend(SOURCE_ROOT, verbose=verbose)


class CanonicalMeanSquare1024Runner(_core.CanonicalMeanSquare1024Runner):
    def __init__(self, device='cuda', *, verbose=False):
        super().__init__(device, source=SOURCE_ROOT, verbose=verbose)
        self.metadata = source_metadata()
