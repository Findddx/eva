"""Optional Linux 4K expert payload reads into the caller's existing buffer.

No Torch, allocation padding, metadata reads, write access or pack mutation.
Unsupported paths fall back with counters; malformed buffers/EOF/EIO raise.
"""
import ctypes
import errno
import os
from pathlib import Path
import stat
import struct
import sys
import threading


ALIGNMENT = 4096
DIRECT_READ_STATS = tuple("direct_io_" + name for name in (
    "requested_reads", "attempt_reads", "complete_reads", "syscalls", "bytes",
    "fallback_reads", "fallback_unavailable", "fallback_alignment", "fallback_errno",
    "fallback_partial", "buffered_bytes", "errors"))
_UNSUPPORTED = {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP}


def requested(environ=None):
    value = (os.environ if environ is None else environ).get("DS41_EXPERT_DIRECT_IO", "0")
    if value not in ("0", "1"):
        raise ValueError("DS41_EXPERT_DIRECT_IO must be 0 or 1")
    return value == "1"


def mount_filesystem(fdinfo, mountinfo, device):
    """Match the opened fd's exact mount ID, including bind/stacked mounts."""
    mount_id = next((line.split(":", 1)[1].strip() for line in fdinfo.splitlines()
                     if line.startswith("mnt_id:")), None)
    if mount_id is None:
        return None
    for line in mountinfo.splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        fields, extra = left.split(), right.split()
        if len(fields) >= 6 and len(extra) >= 3 and fields[0] == mount_id and fields[2] == device:
            return {"mount_id": mount_id, "type": extra[0]}
    return None


def _mount(fd, st_dev):
    with open(f"/proc/self/fdinfo/{fd}", encoding="ascii") as handle:
        fdinfo = handle.read()
    with open("/proc/self/mountinfo", encoding="utf-8", errors="surrogateescape") as handle:
        mountinfo = handle.read()
    return mount_filesystem(fdinfo, mountinfo, f"{os.major(st_dev)}:{os.minor(st_dev)}")


def statx_dio_alignment(fd):
    """Linux UAPI statx is 256B; require its returned DIOALIGN mask.

    The older offset alignment is conservative if a newer kernel separately
    reports a smaller read alignment. No unset/zero field means unrestricted.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "statx", None)
    if function is None:
        return None
    function.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p)
    function.restype = ctypes.c_int
    target = ctypes.create_string_buffer(256)
    if function(fd, b"", 0x1000, 0x2000, target) != 0:  # AT_EMPTY_PATH, STATX_DIOALIGN
        code = ctypes.get_errno()
        if code in _UNSUPPORTED:
            return None
        raise OSError(code, os.strerror(code))
    mask = struct.unpack_from("=I", target.raw, 0)[0]
    memory, offset = struct.unpack_from("=II", target.raw, 152)
    if not mask & 0x2000 or not memory or not offset:
        return None
    return {"memory": memory, "offset": offset, "statx_mask": mask}


class DirectExpertReader:
    """Own only a direct fd; the cache owns its original buffered metadata fd.

    Four pread workers may overlap. close waits for reads before closing the
    direct fd. The immutable pack and no-fork-while-reading are caller contracts.
    """
    def __init__(self, path, buffered_fd, records, *, pack_version=1, enabled=True):
        self.buffered_fd = buffered_fd
        self.requested = bool(enabled)
        self.pid = os.getpid()
        self._condition = threading.Condition()
        self._counts = {name: 0 for name in DIRECT_READ_STATS}
        self._inflight = 0
        self._closed = False
        self._direct_fd = None
        self.disabled_reason = "disabled" if not enabled else None
        self.filesystem = self.dio_alignment = self.file_identity = None
        if not enabled:
            return
        if not sys.platform.startswith("linux") or not hasattr(os, "O_DIRECT") or not hasattr(os, "preadv"):
            self.disabled_reason = "unsupported_platform"
            return
        source = os.fstat(buffered_fd)
        self.file_identity = {name: int(getattr(source, "st_" + name)) for name in ("dev", "ino", "size")}
        if not stat.S_ISREG(source.st_mode):
            self.disabled_reason = "not_regular_file"
            return
        if pack_version != 1:
            self.disabled_reason = "unsupported_pack_version"
            return
        spans = sorted((int(record[0]), int(record[1])) for record in records)
        if (not spans or any(off < 0 or size <= 0 or off % ALIGNMENT or size % ALIGNMENT
                            or off + size > source.st_size for off, size in spans)
                or any(off < prev + size for (prev, size), (off, _) in zip(spans, spans[1:]))):
            self.disabled_reason = "unaligned_or_invalid_pack_records"
            return
        try:
            self.filesystem = _mount(buffered_fd, source.st_dev)
        except OSError:
            self.disabled_reason = "filesystem_unknown"
            return
        if self.filesystem is None or self.filesystem["type"] not in ("ext4", "xfs"):
            self.disabled_reason = "unsupported_filesystem"
            return
        self.dio_alignment = statx_dio_alignment(buffered_fd)
        if (self.dio_alignment is None or any(ALIGNMENT % self.dio_alignment[key]
                                              for key in ("memory", "offset"))):
            self.disabled_reason = "dio_alignment_unknown_or_over_4k"
            return
        direct_fd = None
        try:
            # Reopen without changing the metadata fd's open-file-description flags.
            direct_fd = os.open(str(Path(path).resolve(strict=True)),
                                os.O_RDONLY | os.O_DIRECT | getattr(os, "O_CLOEXEC", 0))
            actual = os.fstat(direct_fd)
            if (actual.st_dev, actual.st_ino, actual.st_size) != (source.st_dev, source.st_ino, source.st_size):
                self.disabled_reason = "reopened_file_identity_changed"
                return
            self._direct_fd, direct_fd = direct_fd, None
        except OSError as exc:
            if exc.errno not in _UNSUPPORTED:
                raise
            self.disabled_reason = "open_errno_" + errno.errorcode.get(exc.errno, str(exc.errno))
        finally:
            if direct_fd is not None:
                os.close(direct_fd)

    def status(self):
        with self._condition:
            return {"requested": self.requested, "active": self._direct_fd is not None and not self.disabled_reason and not self._closed,
                    "scope": "_read_pinned_timed_payload_only", "policy_alignment": ALIGNMENT,
                    "disabled_reason": self.disabled_reason, "filesystem": self.filesystem,
                    "statx_dio_alignment": self.dio_alignment, "file_identity": self.file_identity,
                    "fd_open": self._direct_fd is not None, "closed": self._closed,
                    "inflight": self._inflight, "counters_including_startup": dict(self._counts)}

    @staticmethod
    def _buffered_exact(fd, view, offset, counters):
        done = 0
        while done < len(view):
            try:
                got = os.preadv(fd, [view[done:]], offset + done)
            except InterruptedError:
                continue
            if not 0 < got <= len(view) - done:
                raise RuntimeError("short/invalid buffered expert read")
            counters["direct_io_buffered_bytes"] += got
            done += got

    def read_into(self, target, offset):
        view = memoryview(target)
        if view.readonly or not view.c_contiguous:
            raise ValueError("expert target must be a writable contiguous buffer")
        view = view.cast("B")
        if type(offset) is not int or offset < 0 or not len(view):
            raise ValueError("expert record offset/size is invalid")
        if self.file_identity and offset + len(view) > self.file_identity["size"]:
            raise ValueError("expert record extends outside the opened pack")
        if os.getpid() != self.pid:
            raise RuntimeError("DirectExpertReader cannot be used after fork")
        counters = {name: 0 for name in DIRECT_READ_STATS}
        counters["direct_io_requested_reads"] = int(self.requested)
        with self._condition:
            if self._closed:
                raise RuntimeError("expert reader is closed")
            self._inflight += 1
            fd = self._direct_fd if not self.disabled_reason else None
        try:
            reason = None
            address = ctypes.addressof(ctypes.c_ubyte.from_buffer(view))
            if fd is None:
                reason = "unavailable"
            elif address % ALIGNMENT or offset % ALIGNMENT or len(view) % ALIGNMENT:
                reason = "alignment"
            else:
                counters["direct_io_attempt_reads"] = 1
                done = 0
                while done < len(view):
                    # Every continuation must preserve all three constraints.
                    if ((address + done) % ALIGNMENT or (offset + done) % ALIGNMENT
                            or (len(view) - done) % ALIGNMENT):
                        reason = "partial"
                        break
                    counters["direct_io_syscalls"] += 1
                    try:
                        got = os.preadv(fd, [view[done:]], offset + done)
                    except InterruptedError:
                        continue
                    except OSError as exc:
                        if exc.errno not in _UNSUPPORTED:
                            raise
                        reason = "errno"
                        # Leave this fd open until all concurrent reads drain.
                        with self._condition:
                            self.disabled_reason = "read_errno_" + errno.errorcode.get(exc.errno, str(exc.errno))
                        break
                    if not 0 < got <= len(view) - done:
                        raise RuntimeError("short/invalid direct expert read")
                    counters["direct_io_bytes"] += got
                    done += got
                if reason is None:
                    counters["direct_io_complete_reads"] = 1
                    return counters
            if self.requested:
                counters["direct_io_fallback_reads"] = 1
                counters["direct_io_fallback_" + reason] = 1
            # Restart the entire record, replacing any direct-read prefix.
            self._buffered_exact(self.buffered_fd, view, offset, counters)
            return counters
        except BaseException:
            counters["direct_io_errors"] += 1
            raise
        finally:
            with self._condition:
                for name, value in counters.items():
                    self._counts[name] += value
                self._inflight -= 1
                self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            while self._inflight:
                self._condition.wait()
            fd, self._direct_fd = self._direct_fd, None
        if fd is not None:
            os.close(fd)
