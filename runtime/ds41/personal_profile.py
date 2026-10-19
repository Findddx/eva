"""Unwired CPU-only request-level expert statistics prototype.

No model hooks, GPU operations, weights, prompt text, embeddings or token IDs.
The caller supplies actual target MoE call unions, excluding diagnostic probes.
Candidate snapshots appear only at completed request intervals (default 50).
Working state can be saved after every completion, including partial intervals.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from threading import RLock

import numpy as np


PHASES = ('prefill', 'decode')
SCHEMA = 1
SEMANTICS = 'actual_target_moe_call_union_executed_rows_v1'
MAX_METADATA_BYTES = 16 * 1024**2


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('fingerprints must be lowercase SHA256 digests')


def _atomic_save(payload, path):
    if len(payload) > MAX_METADATA_BYTES:
        raise ValueError('profile exceeds bounded metadata size')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _int64_array(value, shape):
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in 'iu'
            or np.any(array < 0) or np.any(array > np.iinfo(np.int64).max)):
        raise ValueError('invalid aggregate integer accounting')
    return array.astype(np.int64)


@dataclass(frozen=True)
class ProfileIdentity:
    model_sha256: str
    quantization_sha256: str
    tokenizer_sha256: str
    engram_sha256_or_ablation: str
    execution_and_call_geometry_sha256: str
    layers: int = 40
    experts: int = 384
    max_routed_slots: int = 256

    def __post_init__(self):
        for name in ('model_sha256', 'quantization_sha256', 'tokenizer_sha256',
                     'execution_and_call_geometry_sha256'):
            _digest(getattr(self, name))
        if self.engram_sha256_or_ablation != 'ablation':
            _digest(self.engram_sha256_or_ablation)
        _integer(self.layers, 'layers', 1)
        _integer(self.experts, 'experts', 1)
        if type(self.max_routed_slots) is not int or self.max_routed_slots not in (256, 384, 1536):
            raise ValueError('max_routed_slots must be explicitly 256, 384 or 1536')

    def as_metadata(self):
        """Keep the legacy 256-slot identity payload/digest unchanged.

        Capacities 384/1536 are explicit call-geometry variants, never inferred from
        the installed kernel. Omitting the legacy default also lets old working
        states resume without rewriting their statistical identity.
        """
        value = asdict(self)
        if self.max_routed_slots == 256:
            value.pop('max_routed_slots')
        return value

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.as_metadata(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class RequestRoutes:
    """One request, one successful completion; integer IDs stay in RAM only."""
    def __init__(self, identity):
        self.identity = identity
        self.presence = np.zeros((2, identity.layers, identity.experts), dtype=np.int64)
        self.calls = np.zeros(2, dtype=np.int64)
        self.route_uses = np.zeros(2, dtype=np.int64)
        self._consumed = False
        self._lock = RLock()

    @property
    def max_routed_slots(self):
        return self.identity.max_routed_slots

    def record_target_call(self, phase, layer, routed_ids):
        with self._lock:
            if self._consumed:
                raise ValueError('request statistics already consumed')
            if phase not in PHASES:
                raise ValueError('only actual prefill/decode target calls are eligible')
            layer = _integer(layer, 'layer')
            if layer >= self.identity.layers:
                raise ValueError('layer outside profile geometry')
            ids = [_integer(expert, 'expert') for expert in routed_ids]
            if not ids or len(ids) > self.max_routed_slots or max(ids) >= self.identity.experts:
                raise ValueError('invalid actual routed-slot IDs')
            index = PHASES.index(phase)
            unique = list(dict.fromkeys(ids))
            self.presence[index, layer, unique] += 1
            self.calls[index] += 1
            self.route_uses[index] += len(ids)


@dataclass(frozen=True)
class ProfileSnapshot:
    """Serialized immutable training cut: later requests cannot change it."""
    payload: bytes
    trained_through_request: int
    identity_sha256: str

    def save(self, path):
        _atomic_save(self.payload, path)


class PersonalExpertProfile:
    """Two timescales, separate phases, snapshots every N completed requests.

    Rates are call-union presence per processed prefill / committed decode token.
    Decode calls include rejected verifier rows and replay work if actually run.
    This is a statistical producer, never an automatic GPU placement executor.
    """
    def __init__(self, identity, *, update_every=50, fast_half_life=25., slow_half_life=200.):
        self.identity = identity
        self.update_every = _integer(update_every, 'update_every', 1)
        if not (math.isfinite(fast_half_life) and math.isfinite(slow_half_life)
                and 0 < fast_half_life < slow_half_life):
            raise ValueError('half lives must satisfy 0 < fast < slow')
        self.fast_half_life, self.slow_half_life = float(fast_half_life), float(slow_half_life)
        shape = (2, identity.layers, identity.experts)
        self.fast = np.zeros(shape, dtype=np.float64)
        self.slow = np.zeros(shape, dtype=np.float64)
        self.window = np.zeros(shape, dtype=np.int64)
        self.phase_requests = np.zeros(2, dtype=np.int64)
        self.completed_requests = 0
        self._lock = RLock()

    def begin_request(self):
        return RequestRoutes(self.identity)

    def complete_request(self, request, *, processed_prefill_tokens, committed_decode_tokens,
                         valid_pre_eos=True):
        exposures = [_integer(processed_prefill_tokens, 'processed_prefill_tokens'),
                     _integer(committed_decode_tokens, 'committed_decode_tokens')]
        if not any(exposures):
            raise ValueError('empty/no-work requests do not advance the completion counter')
        with self._lock, request._lock:
            if request.identity != self.identity or request._consumed:
                raise ValueError('wrong profile identity or duplicate completion')
            if any(not n and np.any(request.presence[phase]) for phase, n in enumerate(exposures)):
                raise ValueError('routed phase requires a positive token exposure')
            request._consumed = True
            if valid_pre_eos is not True:
                # Drop failed/forced post-EOS work; do not silently re-use it.
                return None
            for phase, n in enumerate(exposures):
                if not n:
                    continue
                value = request.presence[phase].astype(np.float64) / n
                # Bias-correct the zero-initialized exponential mean. Seeding a
                # 200-request EMA directly from request 1 would let that single
                # request dominate the first 50-request placement.
                n = int(self.phase_requests[phase]) + 1
                for state, half_life in ((self.fast, self.fast_half_life), (self.slow, self.slow_half_life)):
                    decay = math.log(2.) / half_life
                    rate = -math.expm1(-decay) / -math.expm1(-n * decay)
                    state[phase] += rate * (value - state[phase])
                self.phase_requests[phase] += 1
            self.window += request.presence
            self.completed_requests += 1
            if self.completed_requests % self.update_every:
                return None
            payload = self._boundary_bytes()
            self.window.fill(0)
            return ProfileSnapshot(payload, self.completed_requests, self.identity.sha256)

    def scores(self, phase='decode', *, stable_weight=.8):
        if phase not in PHASES or not math.isfinite(stable_weight) or not 0 <= stable_weight <= 1:
            raise ValueError('invalid phase/blend')
        with self._lock:
            i = PHASES.index(phase)
            return (stable_weight * self.slow[i] + (1 - stable_weight) * self.fast[i]).copy()

    def drift_js(self, phase='decode'):
        """Mean layer JS divergence in nats; a diagnostic, not a validated gate."""
        if phase not in PHASES:
            raise ValueError('invalid phase')
        with self._lock:
            i = PHASES.index(phase)
            a, b = self.fast[i].copy(), self.slow[i].copy()
        a /= np.maximum(a.sum(1, keepdims=True), 1e-300)
        b /= np.maximum(b.sum(1, keepdims=True), 1e-300)
        middle = (a + b) / 2
        result = np.zeros_like(a)
        for values in (a, b):
            mask = values > 0
            result[mask] += .5 * values[mask] * np.log(values[mask] / middle[mask])
        return float(result.sum(1).mean())

    def _aggregate_value(self):
        # Explicit allowlist: no serialization of request objects or arbitrary
        # user dictionaries, prompts, IDs, timestamps or semantic embeddings.
        value = {'schema': SCHEMA, 'semantics': SEMANTICS, 'identity': self.identity.as_metadata(),
                 'identity_sha256': self.identity.sha256, 'completed_requests': self.completed_requests,
                 'update_every': self.update_every,
                 'fast_half_life': self.fast_half_life, 'slow_half_life': self.slow_half_life,
                 'phase_requests': self.phase_requests.tolist(), 'fast': self.fast.tolist(),
                 'slow': self.slow.tolist(),
                 'ewma_estimator': 'bias_corrected_request_exponential_mean_v1',
                 'normalization': 'per_request_then_per_processed_prefill_or_committed_decode_token'}
        return value

    @staticmethod
    def _encode(value):
        return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()

    def _boundary_bytes(self):
        value = self._aggregate_value()
        value.update(kind='boundary', trained_through_request=self.completed_requests,
                     last_completed_window_presence=self.window.tolist())
        return self._encode(value)

    def save_state(self, path):
        """Atomically persist partial progress, never a placement candidate.

        Call after each completed request for a one-request-per-process CLI.
        The caller must coordinate an external single writer across processes.
        In-flight RequestRoutes are deliberately excluded; replay an interrupted
        request after restart rather than counting its partial statistics.
        """
        with self._lock:
            value = self._aggregate_value()
            remainder = self.completed_requests % self.update_every
            value.update(kind='working', open_window_presence=self.window.tolist(),
                         open_window_completed_requests=remainder,
                         last_candidate_boundary_request=self.completed_requests - remainder)
            _atomic_save(self._encode(value), path)

    @classmethod
    def load(cls, path, expected_identity):
        """Load a boundary candidate; working-state files are rejected."""
        return cls._load(path, expected_identity, kind='boundary')

    @classmethod
    def load_state(cls, path, expected_identity):
        """Restore working progress, including requests 1..49 of a window."""
        return cls._load(path, expected_identity, kind='working')

    @classmethod
    def _load(cls, path, expected_identity, *, kind):
        with Path(path).open('rb') as stream:
            raw = stream.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise ValueError('profile exceeds bounded metadata size')
        value = json.loads(raw)
        allowed = {'schema', 'kind', 'semantics', 'identity', 'identity_sha256', 'completed_requests',
                   'update_every', 'fast_half_life', 'slow_half_life', 'phase_requests', 'fast',
                   'slow', 'normalization', 'ewma_estimator'}
        allowed |= ({'trained_through_request', 'last_completed_window_presence'} if kind == 'boundary'
                    else {'open_window_presence', 'open_window_completed_requests', 'last_candidate_boundary_request'})
        if (not isinstance(value, dict) or set(value) != allowed or value['schema'] != SCHEMA
                or value['semantics'] != SEMANTICS or value['kind'] != kind):
            raise ValueError('unknown profile schema/fields/semantics')
        identity = ProfileIdentity(**value['identity'])
        if identity != expected_identity or value['identity_sha256'] != identity.sha256:
            raise ValueError('model/quantization/Engram/execution fingerprint mismatch')
        result = cls(identity, update_every=value['update_every'], fast_half_life=value['fast_half_life'],
                     slow_half_life=value['slow_half_life'])
        count = _integer(value['completed_requests'], 'completed_requests', 1 if kind == 'boundary' else 0)
        remainder = count % result.update_every
        if kind == 'boundary':
            if remainder or _integer(value['trained_through_request'], 'trained_through_request') != count:
                raise ValueError('profile must represent a completed request boundary')
            last_window = _int64_array(value['last_completed_window_presence'], result.window.shape)
        else:
            if (_integer(value['open_window_completed_requests'], 'open_window_completed_requests') != remainder
                    or _integer(value['last_candidate_boundary_request'], 'last_candidate_boundary_request') != count - remainder):
                raise ValueError('working-state window accounting mismatch')
            result.window = _int64_array(value['open_window_presence'], result.window.shape)
            if not remainder and np.any(result.window):
                raise ValueError('completed working-state window must be empty')
            last_window = result.window
        for name in ('fast', 'slow'):
            array = np.asarray(value[name], dtype=np.float64)
            if array.shape != result.fast.shape or not np.isfinite(array).all() or np.any(array < 0):
                raise ValueError('invalid aggregate profile arrays')
            setattr(result, name, array)
        phase_requests = _int64_array(value['phase_requests'], (2,))
        if (np.any(phase_requests > count)
                or any(not n and (np.any(result.fast[i]) or np.any(result.slow[i]) or np.any(last_window[i]))
                       for i, n in enumerate(phase_requests))):
            raise ValueError('invalid aggregate phase/window accounting')
        if value['normalization'] != 'per_request_then_per_processed_prefill_or_committed_decode_token':
            raise ValueError('profile normalization mismatch')
        if value['ewma_estimator'] != 'bias_corrected_request_exponential_mean_v1':
            raise ValueError('profile EWMA estimator mismatch')
        result.phase_requests = phase_requests
        result.completed_requests = count
        return result
