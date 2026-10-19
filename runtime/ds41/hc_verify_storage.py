"""Opt-in building block for external HC verification-graph I/O storage.

No production hook, environment flag, graph capture or GPU work runs on import.
This arena owns storage only; callers must retain the documented serial HC
consumer order. Coefficients are borrowed and may be overwritten by later
layers, shapes or forwards. Two pre buffers are required by Block.forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import threading


MAX_ROWS = 128
MAX_STORAGE_BYTES = 64 * 1024**2
LARGE_ROWS = (512, 1024)
LARGE_MAX_STORAGE_BYTES = 96 * 1024**2


@dataclass(frozen=True)
class HCVerifyBuffers:
    hidden: object
    pre: object
    post: object
    comb: object


def _row_lengths(lengths):
    if not isinstance(lengths, (tuple, list)):
        raise TypeError("HC arena lengths must be an explicit tuple/list")
    values = tuple(lengths)
    if len(values) == 1 and type(values[0]) is int and values[0] in LARGE_ROWS:
        return values
    if (not 1 <= len(values) <= 4 or len(set(values)) != len(values)
            or any(type(n) is not int or not 2 <= n <= MAX_ROWS for n in values)):
        raise ValueError("HC arena requires1..4 distinct exact lengths in2..128 or a single512/1024; M1 is separate")
    return values


class HCVerifyIOArena:
    """One hidden per dtype and separate attention/FFN coefficients.

    CUDA allocation must occur outside capture. All captures use one explicitly
    bound capture stream. Replay binds a single CPU thread/current CUDA stream,
    separately from the capture stream. The caller must check replay ownership
    before copying input/replaying and keep all consumers on that same stream.
    CPU device is only for structural tests, never CUDA numerical evidence.
    """
    def __init__(self, lengths=(32, 64, 128), *, device="cuda", hidden_dtypes=None,
                 dim=5120, hc_mult=4, torch_module=None):
        if torch_module is None:
            import torch as torch_module
        torch = torch_module
        self.torch = torch
        self.lengths = _row_lengths(lengths)
        self.max_rows = max(self.lengths)
        if type(dim) is not int or not 1 <= dim <= 16384 or hc_mult != 4 or type(hc_mult) is not int:
            raise ValueError("HC arena requiresHC4 and bounded dim1..16384")
        self.dim, self.hc_mult = dim, hc_mult
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("HC arena supports explicit CUDA or CPU structural-test device")
        if self.device.type == "cuda":
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("HC arena external allocations must precede graph capture")
        dtypes = (torch.float16, torch.bfloat16) if hidden_dtypes is None else tuple(hidden_dtypes)
        if (not 1 <= len(dtypes) <= 2 or len(set(dtypes)) != len(dtypes)
                or any(dtype not in (torch.float16, torch.bfloat16) for dtype in dtypes)):
            raise ValueError("HC hidden dtypes must be unique Half/BFloat16, at most two")
        self.hidden_dtypes = dtypes
        hidden_bytes = len(dtypes) * self.max_rows * hc_mult * dim * 2
        coefficient_bytes = 2 * self.max_rows * (2 * hc_mult + hc_mult**2) * 4
        self.external_bytes = hidden_bytes + coefficient_bytes
        self.storage_budget_bytes = (LARGE_MAX_STORAGE_BYTES if self.max_rows in LARGE_ROWS
                                     else MAX_STORAGE_BYTES)
        if self.external_bytes > self.storage_budget_bytes:
            raise ValueError("HC arena exceeds bounded external storage byte budget")
        self._owner_thread = threading.current_thread()
        self._capture_stream = None
        self._capture_bound = False
        self._replay_stream = None
        self._replay_bound = False
        self._active_rows = None
        self._failed = False
        self._hidden = {dtype: torch.empty(1, self.max_rows, hc_mult, dim,
                                          device=self.device, dtype=dtype) for dtype in dtypes}
        self._pre = tuple(torch.empty(1, self.max_rows, hc_mult, device=self.device, dtype=torch.float32)
                          for _ in range(2))
        self._post = tuple(torch.empty_like(value) for value in self._pre)
        self._comb = tuple(torch.empty(1, self.max_rows, hc_mult, hc_mult,
                                      device=self.device, dtype=torch.float32) for _ in range(2))
        # Precreate all exact-shape views outside capture. Prefix views remain
        # contiguous for B1 and retain their max-row external backing storage.
        hidden_views = {(rows, dtype): value[:, :rows] for dtype, value in self._hidden.items()
                        for rows in self.lengths}
        self._views = {(rows, kind, dtype): HCVerifyBuffers(
            hidden_views[rows, dtype], self._pre[kind][:, :rows],
            self._post[kind][:, :rows], self._comb[kind][:, :rows])
            for rows in self.lengths for kind in (0, 1) for dtype in dtypes}

    def _check_thread(self):
        if threading.current_thread() is not self._owner_thread:
            raise RuntimeError("HC arena belongs to its construction CPU thread")
        if self._failed:
            raise RuntimeError("HC arena forward failed; discard this arena and captured graphs")

    def buffers(self, rows, kind, dtype):
        """Borrow exact-shape external views; never allocate or clear on use."""
        self._check_thread()
        if type(rows) is not int or type(kind) is not int or kind not in (0, 1):
            raise ValueError("HC arena requires exact configured rows and kind0/1")
        if self._active_rows is not None and rows != self._active_rows:
            raise RuntimeError("Cannot borrow another HC shape during an active forward")
        try:
            return self._views[rows, kind, dtype]
        except KeyError as exc:
            raise ValueError("Unconfigured HC arena rows/dtype; no silent padding/fallback") from exc

    def check_capture_owner(self, stream=None):
        """Bind one capture stream; caller supplies its existing stream waits."""
        self._check_thread()
        if self._active_rows is not None or self._replay_bound:
            raise RuntimeError("HC arena recapture after replay/live forward is unsupported")
        if self.device.type == "cuda":
            if (stream is None or stream.device != self.device
                    or type(stream.cuda_stream) is not int):
                raise ValueError("HC arena capture requires one matching explicit CUDA stream")
        elif stream is not None:
            raise ValueError("CPU HC arena has no CUDA capture stream")
        if self._capture_bound and stream is not self._capture_stream:
            raise RuntimeError("HC arena captures must retain the same stream object")
        self._capture_bound, self._capture_stream = True, stream

    def _bind_replay_stream(self):
        self._check_thread()
        if not self._capture_bound:
            raise RuntimeError("Bind the arena capture owner before replay")
        stream = (self.torch.cuda.current_stream(self.device)
                  if self.device.type == "cuda" else None)
        token = None if stream is None else (stream.device, stream.cuda_stream)
        if self._replay_bound and token != self._replay_stream:
            raise RuntimeError("HC arena replay/consumers require their original CUDA stream")
        self._replay_bound, self._replay_stream = True, token

    @contextmanager
    def replay_scope(self, rows):
        """Cover the complete forward through final head-pre consumption.

        Exiting means consumers have been enqueued, not device-synchronized.
        The next scope must use the same CUDA stream so their work is ordered.
        """
        self._check_thread()
        if self._active_rows is not None:
            raise RuntimeError("HC arena forbids nested/interleaved complete forwards")
        if type(rows) is not int or rows not in self.lengths:
            raise ValueError("HC arena replay scope requires an exact configured length")
        self._bind_replay_stream()
        self._active_rows = rows
        try:
            yield self
        except BaseException:
            self._failed = True
            raise
        finally:
            self._active_rows = None

    def check_replay_owner(self, rows=None):
        """Check before input copy/replay inside the complete forward scope."""
        self._check_thread()
        if self._active_rows is None or (rows is not None and rows != self._active_rows):
            raise RuntimeError("HC replay requires its matching complete-forward scope")
        self._bind_replay_stream()

    def metadata(self):
        """No addresses, CUDA synchronization or tensor value readback."""
        return {
            "kind": "external_hc_verify_io_arena_v1", "production_integrated": False,
            "GPU_math_validated": False, "performance_measured": False,
            "device": str(self.device), "lengths": list(self.lengths), "max_rows": self.max_rows,
            "dim": self.dim, "hc_mult": self.hc_mult,
            "hidden_dtypes": [str(value) for value in self.hidden_dtypes],
            "hidden_allocations": len(self._hidden), "pre_allocations": 2,
            "post_allocations": 2, "comb_allocations": 2,
            "external_bytes": self.external_bytes, "external_storage_budget_bytes": self.storage_budget_bytes,
            "capture_owner_bound": self._capture_bound, "replay_owner_bound": self._replay_bound,
            "active_forward_rows": self._active_rows, "complete_forward_scope_required": True,
            "failed": self._failed,
            "M1_storage_shared": False, "state_token_padding": False,
            "output_lifetime": "Borrowed until later same-kind HC write; consume serially before next same-kind replay",
            "contract": "Ordinary complete B1 target forward only; no interleaved forwards, suspended stacks, retained pre references, cross-stream consumers or live recapture",
            "memory_scope": "External tensors only, excluding graph-private scratch, graph executable/library workspace and allocator rounding",
        }
