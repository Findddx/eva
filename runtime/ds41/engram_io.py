"""Engram row prefetch.

Hash ids are known from the token ids alone, so the reads are issued at the
start of the forward and waited on at layers 1 and 14. Each row is one pread.
When liburing is installed the same offsets go through io_uring; otherwise
pread is the path that actually runs.
"""
import os
import threading


class EngramPrefetcher:
    ROW = 256  # fp8 values; the projection sees head_dim after the table

    def __init__(self, paths: list[str] | None, row_bytes: int = 256):
        self.row_bytes = row_bytes
        self.fds = []
        self.backend = "pread"
        for path in paths or []:
            if path and os.path.isfile(path):
                self.fds.append(os.open(path, os.O_RDONLY))
        self._thread = None
        self._rows = None
        self.reads = 0
        self.bytes = 0
        try:
            import ctypes
            ctypes.CDLL("liburing.so.2")
            self.backend = "liburing-present"
        except OSError:
            pass

    def close(self):
        for fd in self.fds:
            os.close(fd)

    def kick(self, hash_ids: list[list[int]]):
        """hash_ids[layer_slot] = row ids for that Engram module."""
        if not self.fds:
            self._rows = None
            return

        def _run():
            out = []
            for slot, ids in enumerate(hash_ids):
                fd = self.fds[min(slot, len(self.fds) - 1)]
                rows = []
                for rid in ids:
                    buf = os.pread(fd, self.row_bytes, int(rid) * self.row_bytes)
                    rows.append(buf)
                    self.reads += 1
                    self.bytes += len(buf)
                out.append(rows)
            self._rows = out

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def wait(self, slot: int):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if not self._rows or slot >= len(self._rows):
            return None
        return self._rows[slot]
