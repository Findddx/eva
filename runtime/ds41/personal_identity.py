"""Opt-in metadata fingerprints for personal expert statistics, no payload scan.

The pack identity hashes only its validated v1 header/table, plus file size and
mtime_ns. Config/index/quantization metadata and tokenizer are bounded complete
file hashes. Engram receipts are hashed as claims, not reverified against GGUF
payloads. This detects many accidental runtime changes; it is NOT a whole-weight
content attestation, and restored mtimes can hide same-size payload changes.

No environment access, network, model loading, file writes or path disclosure.
The caller supplies only public execution settings, never request data/secrets.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct

from ds41.personal_profile import ProfileIdentity


FINGERPRINT_SCOPE = 'ds41_runtime_metadata_v1_not_whole_weight_payload_attestation'
MAX_METADATA_BYTES = 128 * 1024**2
PACK_HEADER = struct.Struct('<8sIIII')
PACK_TABLE_BYTES = 40 * 384 * 128
MAX_PACK_METADATA_BYTES = 2 * 1024**2 + PACK_HEADER.size
_PRIVATE_FIELDS = frozenset({
    'prompt', 'prompt_text', 'text', 'messages', 'input_ids', 'output_ids',
    'token_ids', 'request_id', 'user_id', 'credentials', 'password', 'secret',
    'api_key', 'apikey', 'access_token', 'hf_token', 'gpu_key', 'authorization',
    'cookie', 'environ', 'environment',
})


def _public_json(value, *, reject_private=False):
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('JSON settings must contain only finite numbers')
        return
    if type(value) is list:
        for item in value:
            _public_json(item, reject_private=reject_private)
        return
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError('JSON setting keys must be strings')
            if reject_private and key.lower().replace('-', '_') in _PRIVATE_FIELDS:
                raise ValueError('execution settings contain a private field')
            _public_json(item, reject_private=reject_private)
        return
    raise ValueError('settings must contain JSON values only')


def _canonical_hash(value):
    _public_json(value)
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                     allow_nan=False).encode('utf-8')
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError('JSON settings exceed metadata limit')
    return hashlib.sha256(raw).hexdigest()


def _stat_key(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _assert_stable(stream, path, before):
    if (_stat_key(os.fstat(stream.fileno())) != _stat_key(before)
            or _stat_key(path.stat()) != _stat_key(before)):
        raise ValueError('metadata source changed while fingerprinting')


def _read_metadata(path, *, retain=False):
    """Hash a regular file with a strict stream cap; optional bytes for JSON."""
    path = Path(path)
    digest = hashlib.sha256()
    kept = bytearray() if retain else None
    with path.open('rb', buffering=0) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_METADATA_BYTES:
            raise ValueError('metadata must be a regular file at most 128 MiB')
        read = 0
        while True:
            chunk = stream.read(min(1024**2, MAX_METADATA_BYTES + 1 - read))
            if not chunk:
                break
            read += len(chunk)
            if read > MAX_METADATA_BYTES:
                raise ValueError('metadata exceeds stream limit')
            digest.update(chunk)
            if retain:
                kept.extend(chunk)
        if read != before.st_size:
            raise ValueError('metadata length changed while fingerprinting')
        _assert_stable(stream, path, before)
    return digest.hexdigest(), bytes(kept) if retain else None


def _reject_constant(_value):
    raise ValueError('metadata JSON contains a non-finite constant')


def _unique_fields(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('metadata JSON contains duplicate fields')
        value[key] = item
    return value


def _pack_metadata(path):
    path = Path(path)
    with path.open('rb', buffering=0) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('expert pack must be a regular file')
        header = stream.read(PACK_HEADER.size)
        if len(header) != PACK_HEADER.size:
            raise ValueError('truncated expert pack header')
        if PACK_HEADER.unpack(header) != (b'DS41PK01', 1, 40, 384, 128):
            raise ValueError('expert pack must be DS41PK01 v1 with 40/384/128 geometry')
        prefix_bytes = PACK_HEADER.size + PACK_TABLE_BYTES
        if prefix_bytes > MAX_PACK_METADATA_BYTES or before.st_size < prefix_bytes:
            raise ValueError('expert pack metadata is truncated or exceeds limit')
        digest = hashlib.sha256(header)
        remaining = PACK_TABLE_BYTES
        while remaining:
            chunk = stream.read(min(1024**2, remaining))
            if not chunk:
                raise ValueError('truncated expert pack table')
            digest.update(chunk)
            remaining -= len(chunk)
        _assert_stable(stream, path, before)
    return {'header_table_sha256': digest.hexdigest(), 'header_table_bytes': prefix_bytes,
            'file_size': before.st_size, 'file_mtime_ns': before.st_mtime_ns}


def _engram_identity(manifest_path, source_index_sha256):
    if manifest_path is None:
        return _canonical_hash({'scope': FINGERPRINT_SCOPE, 'kind': 'raw_checkpoint_source_index',
                                'source_index_sha256': source_index_sha256})
    manifest_path = Path(manifest_path)
    manifest_sha, raw = _read_metadata(manifest_path, retain=True)
    manifest = json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_unique_fields)
    if not isinstance(manifest, dict):
        raise ValueError('Engram manifest must be a JSON object')
    _public_json(manifest)
    receipts = {'manifest_directory': manifest_path.parent / 'verified-copy.json'}
    if manifest.get('format') == 'ranma_mxfp8_gguf':
        index = manifest.get('index_path')
        if not isinstance(index, str) or not index:
            raise ValueError('GGUF Engram manifest requires a string index_path')
        index = Path(index)
        if not index.is_absolute():
            index = manifest_path.parent / index
        receipts['index_directory'] = index.parent / 'verified-copy.json'
    receipt_hashes = {role: _read_metadata(path)[0] if path.exists() else None
                      for role, path in receipts.items()}
    return _canonical_hash({'scope': FINGERPRINT_SCOPE, 'kind': 'external_manifest_metadata',
                            'manifest_sha256': manifest_sha, 'receipt_sha256': receipt_hashes})


def identity_from_runtime(model_dir, pack_path, dense_dir=None, engram_manifest=None,
                          engram_present=False, *, execution, max_routed_slots=256):
    """Return five digests/ablation plus call geometry, never source paths.

    All four model metadata files are required. Missing/oversized/unstable input
    fails closed. Metadata hashes do not prove that referenced payloads match.
    Engram without an external manifest is identified by the source index;
    an external manifest without receipts has metadata-only scope as well.
    """
    if type(execution) is not dict:
        raise ValueError('execution must be a public JSON object')
    _public_json(execution, reject_private=True)
    if max_routed_slots == 1536:
        # This tier changes actual-call statistics, never physical residency.
        # Reject a detached capacity declaration before reading any metadata.
        from ds41.native2048_runtime import validate_configuration
        kernels, experts = execution.get('kernel_settings'), execution.get('expert_coop512')
        if type(kernels) is not dict or type(experts) is not dict:
            raise ValueError('1536-slot profile identity requires explicit native2048 execution')
        native = experts.get('native2048')
        configured = validate_configuration(kernels,
            cap=execution.get('verification_max_rows'), speculation=execution.get('speculation'),
            prefill_chunk_size=execution.get('prefill_chunk_size'),
            volatile_experts=execution.get('volatile_experts'))
        if (execution.get('personal_max_routed_slots') != 1536 or configured.get('requested') is not True
                or type(native) is not dict or not all(k in native for k in configured)
                or _canonical_hash({k:native[k] for k in configured}) != _canonical_hash(configured)):
            raise ValueError('1536-slot profile identity requires explicit matching native2048 execution')
    if type(engram_present) is not bool:
        raise ValueError('engram_present must be a boolean')
    model_dir = Path(model_dir)
    config_sha = _read_metadata(model_dir / 'config.json')[0]
    source_sha = _read_metadata(model_dir / 'model.safetensors.index.json')[0]
    quant_sha = _read_metadata(model_dir / 'exl3_k_map.json')[0]
    tokenizer_sha = _read_metadata(model_dir / 'tokenizer.json')[0]
    if 'edit_candidate' in execution and (type(execution['edit_candidate']) is not dict
            or execution['edit_candidate'].get('tokenizer_sha256') != tokenizer_sha):
        raise ValueError('Stable edit execution identity differs from the actual tokenizer metadata')
    pack = _pack_metadata(pack_path)
    dense_sha = None if dense_dir is None else _read_metadata(
        Path(dense_dir) / 'model.safetensors.index.json')[0]
    return ProfileIdentity(
        model_sha256=_canonical_hash({'scope': FINGERPRINT_SCOPE, 'config_sha256': config_sha,
                                      'source_index_sha256': source_sha}),
        quantization_sha256=_canonical_hash({'scope': FINGERPRINT_SCOPE, 'kmap_sha256': quant_sha,
                                             'pack': pack, 'dense_index_sha256': dense_sha}),
        tokenizer_sha256=tokenizer_sha,
        engram_sha256_or_ablation=(_engram_identity(engram_manifest, source_sha)
                                  if engram_present else 'ablation'),
        execution_and_call_geometry_sha256=_canonical_hash(execution),
        max_routed_slots=max_routed_slots)
