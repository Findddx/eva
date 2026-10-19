"""Bounded metadata for exact same-layer next-chunk disk reads; no Torch import."""
from contextlib import contextmanager
from dataclasses import dataclass
import threading


MIB = 1024**2
GIB = 1024**3
COUNTERS = (
    "submitted_reads", "submitted_bytes", "completed_reads", "completed_bytes",
    "timely_reads", "timely_bytes", "late_reads", "late_bytes",
    "unused_reads", "unused_bytes", "cancelled_reads", "cancelled_bytes",
    "skip_budget", "skip_headroom", "skip_resident", "skip_pending",
    "peak_charge_bytes", "submit_s", "cleanup_s",
)


@dataclass(frozen=True)
class ChunkPrefetchConfig:
    enabled: bool = False
    max_records: int = 4
    max_charge_bytes: int = 64 * MIB

    @classmethod
    def from_env(cls, env):
        flag = env.get("DS41_NEXT_CHUNK_PREFETCH", "0")
        if flag not in ("0", "1"):
            raise ValueError("DS41_NEXT_CHUNK_PREFETCH must be 0 or 1")
        try:
            records = int(env.get("DS41_NEXT_CHUNK_PREFETCH_RECORDS", "4"))
            mib = int(env.get("DS41_NEXT_CHUNK_PREFETCH_MIB", "64"))
        except (TypeError, ValueError) as exc:
            raise ValueError("next-chunk prefetch limits must be integers") from exc
        if not 1 <= records <= 16 or not 1 <= mib <= 512:
            raise ValueError("next-chunk prefetch requires records1..16 and MiB1..512")
        return cls(flag == "1", records, mib * MIB)

    def as_dict(self):
        return dict(enabled=self.enabled, max_records=self.max_records,
                    max_charge_bytes=self.max_charge_bytes,
                    minimum_available_bytes=8 * GIB, policy="exact_next_span_disk_only_v1",
                    scope="explicit_actual_decode_only", speculative_ram_admission=False,
                    extra_route_readbacks=0)


@dataclass
class PendingOrigin:
    nbytes: int
    charge: int
    claimed: bool = False
    ready_at_demand: bool = False


class ChunkPrefetchState:
    def __init__(self, config):
        self.config = config
        self.entries = {}
        self.charge = 0
        self._active = False
        self._owner = None

    @property
    def active(self):
        if self._active and threading.get_ident() != self._owner:
            raise RuntimeError("next-chunk prefetch requires the decode owner thread")
        return self._active

    @contextmanager
    def scope(self, cleanup):
        if self._active or self.entries:
            raise RuntimeError("next-chunk prefetch scope is nested or has pending reads")
        self._owner, self._active = threading.get_ident(), True
        try:
            yield
        finally:
            try:
                cleanup()
            finally:
                self._active, self._owner = False, None

    def pop(self, key):
        entry = self.entries.pop(key, None)
        if entry is not None:
            self.charge -= entry.charge
        return entry


def validate_next_ids(ids):
    """Only consume the already available CPU routes; never convert tensors."""
    if not isinstance(ids, (list, tuple)) or not 1 <= len(ids) <= 384:
        raise ValueError("next chunk routes must be an existing CPU list of1..384 IDs")
    if any(type(value) is not int or not 0 <= value < 384 for value in ids):
        raise ValueError("next chunk expert IDs must be integers in0..383")
    return list(dict.fromkeys(ids))
