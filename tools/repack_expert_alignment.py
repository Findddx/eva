#!/usr/bin/env python3
"""Copy a complete DS41PK01 v1 pack to a new, 4K-aligned artifact.

Only record-offset uint64 fields change. No torch/numpy, O_DIRECT, checkpoint
download, in-place patch, workspace allocation or inference-path modification.
"""
import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import sys
import tempfile
import time


ALIGNMENT = 4096
MAGIC = b"DS41PK01"
HEADER = struct.Struct("<8sIIII")
SLOT = struct.Struct("<QIHH" + "HHHHIIIiIII" * 3)
LAYERS, EXPERTS, SLOT_BYTES = 40, 384, 128
TABLE_BYTES = LAYERS * EXPERTS * SLOT_BYTES
TABLE_END = HEADER.size + TABLE_BYTES
MAX_CHUNK = 16 * 1024**2
DEFAULT_CHUNK = 8 * 1024**2
DEFAULT_RESERVE = 1024**3
DEFAULT_MAX_OUTPUT = 512 * 1024**3
SPACE_CHECK_INTERVAL = 64 * 1024**2
RECEIPT_RESERVE = 64 * 1024


class PackError(ValueError):
    pass


@dataclass(frozen=True)
class Layout:
    path: Path
    fingerprint: dict
    path_fingerprint: dict
    header: bytes
    table: bytes
    padding: bytes
    payload_start: int
    size: int
    records: tuple

    @property
    def payload_bytes(self):
        return self.size - self.payload_start


def _fingerprint(value):
    return {name: int(getattr(value, "st_" + name))
            for name in ("dev", "ino", "size", "mtime_ns", "ctime_ns", "mode")}


def _assert_stable(handle, layout):
    if (_fingerprint(os.fstat(handle.fileno())) != layout.fingerprint
            or _fingerprint(layout.path.stat()) != layout.path_fingerprint):
        raise PackError("source identity/size/mtime/ctime changed during the operation")


def _read_exact(handle, count):
    result = bytearray()
    while len(result) < count:
        data = handle.read(min(MAX_CHUNK, count - len(result)))
        if not data:
            raise PackError(f"short read: need {count} bytes, received {len(result)}")
        result.extend(data)
    return bytes(result)


def _write_all(handle, data):
    view = memoryview(data)
    while view:
        count = handle.write(view)
        if count is None or count <= 0 or count > len(view):
            raise OSError("short/invalid output write")
        view = view[count:]


def _parameters(chunk_bytes, reserve_bytes=0, max_output_bytes=DEFAULT_MAX_OUTPUT):
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= MAX_CHUNK:
        raise ValueError("chunk_bytes must be an integer in 1..16MiB")
    if type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("reserve_bytes must be a nonnegative integer")
    if type(max_output_bytes) is not int or max_output_bytes < 1:
        raise ValueError("max_output_bytes must be a positive integer")


def _read_layout(handle, path):
    before = os.fstat(handle.fileno())
    path_before = _fingerprint(path.stat())
    # Windows Python builds can expose different ctime meanings through fstat
    # and path stat. Freeze each independently, and require shared identity/data
    # fields to agree before reading. Never compare ctime across the two APIs.
    handle_before = _fingerprint(before)
    if any(handle_before[key] != path_before[key]
           for key in ("dev", "ino", "size", "mtime_ns", "mode")):
        raise PackError("source path does not match opened file identity/data")
    if not stat.S_ISREG(before.st_mode):
        raise PackError("source must be a regular file")
    handle.seek(0)
    header = _read_exact(handle, HEADER.size)
    magic, version, layers, experts, slot = HEADER.unpack(header)
    if magic != MAGIC or version != 1:
        raise PackError("only DS41PK01 version 1 is supported")
    if (layers, experts, slot) != (LAYERS, EXPERTS, SLOT_BYTES):
        raise PackError("pack dimensions must be 40 layers, 384 experts, 128-byte slots")
    table = _read_exact(handle, TABLE_BYTES)
    records = []
    for index in range(LAYERS * EXPERTS):
        fields = SLOT.unpack_from(table, index * SLOT_BYTES)
        offset, size, first_k, ready = fields[:4]
        if ready != 1 or not first_k or not size or size % ALIGNMENT:
            raise PackError(f"slot {index}: incomplete metadata or non-4K record length")
        spans = []
        for projection in range(3):
            base = 4 + 11 * projection
            inn, out, k, _pad, su, sv, tr, _mul, a, b, c = fields[base:base + 11]
            if not inn or not out or not k or not su or not sv or not tr:
                raise PackError(f"slot {index}: incomplete projection metadata")
            for start, count in ((a, su), (b, sv), (c, tr)):
                if start % 2 or count % 2 or start + count > size:
                    raise PackError(f"slot {index}: tensor range outside/alignment of record")
                spans.append((start, start + count))
        spans.sort()
        if any(right[0] < left[1] for left, right in zip(spans, spans[1:])):
            raise PackError(f"slot {index}: overlapping tensor ranges")
        if offset < TABLE_END or offset + size > before.st_size:
            raise PackError(f"slot {index}: record offset outside file payload")
        records.append((index, offset, size))
    ordered = sorted(records, key=lambda record: record[1])
    start = ordered[0][1]
    aligned_start = (TABLE_END + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    if start not in (TABLE_END, aligned_start):
        raise PackError("unsupported payload start; expected original or minimally aligned v1 layout")
    cursor = start
    for index, offset, size in ordered:
        if offset != cursor:
            raise PackError(f"slot {index}: overlapping records or gap in payload")
        cursor += size
    if cursor != before.st_size:
        raise PackError("unexpected trailing bytes or incomplete payload")
    handle.seek(TABLE_END)
    padding = _read_exact(handle, start - TABLE_END)
    layout = Layout(path, handle_before, path_before, header, table, padding, start, before.st_size, tuple(records))
    _assert_stable(handle, layout)
    return layout


def _destination(path):
    path = Path(path).absolute()
    # lexists also rejects dangling symlinks. Never follow one for publication.
    if os.path.lexists(path):
        raise FileExistsError(f"refusing existing output: {path}")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("output parent must be an existing directory")
    return parent / path.name


def _paths(source, destination, receipt=None):
    source = Path(source).resolve(strict=True)
    if Path(destination).resolve(strict=False) == source:
        raise ValueError("source and destination resolve to the same path")
    destination = _destination(destination)
    receipt = _destination(receipt if receipt is not None else str(destination) + ".receipt.json")
    if receipt == destination or receipt.resolve(strict=False) == source:
        raise ValueError("source, destination and receipt must be distinct")
    if receipt.parent != destination.parent:
        raise ValueError("receipt must use the destination directory")
    return source, destination, receipt


def _space(parent, remaining, reserve):
    free = shutil.disk_usage(parent).free
    required = remaining + reserve + RECEIPT_RESERVE
    if free < required:
        raise OSError(errno.ENOSPC, f"need {required} free bytes including reserve; available {free}", str(parent))
    return free


def _new_table(layout):
    new_start = (TABLE_END + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    delta = new_start - layout.payload_start
    table = bytearray(layout.table)
    for index, offset, _size in layout.records:
        struct.pack_into("<Q", table, index * SLOT_BYTES, offset + delta)
    padding = layout.padding if delta == 0 else bytes(new_start - TABLE_END)
    return bytes(table), padding, delta


def _metadata_hash(table):
    digest = hashlib.sha256()
    for offset in range(0, TABLE_BYTES, SLOT_BYTES):
        digest.update(table[offset + 8:offset + SLOT_BYTES])
    return digest.hexdigest()


def plan(source, destination, *, reserve_bytes=DEFAULT_RESERVE,
         max_output_bytes=DEFAULT_MAX_OUTPUT, receipt=None):
    _parameters(DEFAULT_CHUNK, reserve_bytes, max_output_bytes)
    source, destination, receipt = _paths(source, destination, receipt)
    with source.open("rb", buffering=0) as handle:
        layout = _read_layout(handle, source)
        _table, _padding, delta = _new_table(layout)
        size = layout.size + delta
        if size > max_output_bytes:
            raise ValueError("planned output exceeds max_output_bytes")
        free = _space(destination.parent, size, reserve_bytes)
        return dict(kind="expert_alignment_plan", source=str(source), destination=str(destination),
                    receipt=str(receipt), source_stat=layout.fingerprint, source_bytes=layout.size,
                    destination_bytes=size, payload_bytes=layout.payload_bytes,
                    old_payload_start=layout.payload_start, new_payload_start=layout.payload_start + delta,
                    inserted_bytes=delta, record_count=len(layout.records), source_version=1,
                    reserve_bytes=reserve_bytes, available_bytes=free, max_output_bytes=max_output_bytes,
                    payload_sha256=None, payload_checksum_computed=False)


def _hash_payload(handle, layout, chunk_bytes, *, progress=None, phase="verify"):
    payload, whole = hashlib.sha256(), hashlib.sha256(layout.header + layout.table + layout.padding)
    handle.seek(layout.payload_start)
    done = 0
    while done < layout.payload_bytes:
        _assert_stable(handle, layout)
        data = handle.read(min(chunk_bytes, layout.payload_bytes - done))
        if not data:
            raise PackError("short read while hashing payload")
        payload.update(data)
        whole.update(data)
        done += len(data)
        if progress is not None:
            progress(dict(phase=phase, bytes=done, total=layout.payload_bytes))
    _assert_stable(handle, layout)
    return payload.hexdigest(), whole.hexdigest()


def _verify_destination(source_layout, destination, payload_sha, expected_file_sha, chunk_bytes, progress=None):
    table, padding, delta = _new_table(source_layout)
    with Path(destination).open("rb", buffering=0) as handle:
        layout = _read_layout(handle, Path(destination))
        if (layout.header != source_layout.header or layout.table != table or layout.padding != padding
                or layout.size != source_layout.size + delta or layout.payload_start % ALIGNMENT):
            raise PackError("destination header/table/padding/offsets differ from expected transformation")
        actual_payload, actual_file = _hash_payload(handle, layout, chunk_bytes, progress=progress)
        if actual_payload != payload_sha or (expected_file_sha is not None and actual_file != expected_file_sha):
            raise PackError("destination payload/file SHA-256 mismatch")
        return dict(payload_sha256=actual_payload, destination_file_sha256=actual_file,
                    slot_metadata_without_offsets_sha256=_metadata_hash(layout.table),
                    destination_table_sha256=hashlib.sha256(layout.table).hexdigest(),
                    destination_bytes=layout.size, destination_payload_start=layout.payload_start,
                    all_record_offsets_aligned=all(offset % ALIGNMENT == 0 for _, offset, _ in layout.records),
                    verified_record_count=len(layout.records))


def verify(source, destination, *, chunk_bytes=DEFAULT_CHUNK, progress=None):
    """Independently hash both complete payloads and compare every slot byte."""
    _parameters(chunk_bytes)
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve(strict=True)
    if source == destination or os.path.samefile(source, destination):
        raise ValueError("verification requires distinct source and destination files")
    with source.open("rb", buffering=0) as handle:
        layout = _read_layout(handle, source)
        payload, whole = _hash_payload(handle, layout, chunk_bytes, progress=progress, phase="hash_source")
        result = _verify_destination(layout, destination, payload, None, chunk_bytes, progress)
        _assert_stable(handle, layout)
        return dict(kind="expert_alignment_verification", source=str(source), destination=str(destination),
                    source_file_sha256=whole, source_payload_sha256=payload, source_stat=layout.fingerprint,
                    payload_bytes=layout.payload_bytes, slot_metadata_byte_equal_except_offsets=True, **result)


def _publish_noreplace(source, destination):
    # Plain POSIX os.rename may overwrite a racing destination. Do not use it.
    if os.name == "nt":
        os.rename(source, destination)
    elif sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameat2", None)
        if rename is None:
            raise RuntimeError("atomic no-replace renameat2 is unavailable")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(destination))
    else:
        raise RuntimeError("atomic no-replace publication is supported only on Linux and Windows")


def _fsync_parent(path):
    if os.name != "posix":
        return False
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def repack(source, destination, *, chunk_bytes=DEFAULT_CHUNK, reserve_bytes=DEFAULT_RESERVE,
           max_output_bytes=DEFAULT_MAX_OUTPUT, keep_partial=False, progress=None, receipt=None):
    """Write, fsync, fully verify, then exclusively publish a new pack + receipt."""
    _parameters(chunk_bytes, reserve_bytes, max_output_bytes)
    source, destination, receipt = _paths(source, destination, receipt)
    temp = receipt_temp = None
    published = False
    began = time.perf_counter()
    try:
        with source.open("rb", buffering=0) as handle:
            layout = _read_layout(handle, source)
            table, padding, delta = _new_table(layout)
            new_size = layout.size + delta
            if new_size > max_output_bytes:
                raise ValueError("planned output exceeds max_output_bytes")
            initial_free = _space(destination.parent, new_size, reserve_bytes)
            fd, name = tempfile.mkstemp(prefix=destination.name + ".partial-", suffix=".tmp", dir=destination.parent)
            temp = Path(name)
            source_sha = hashlib.sha256(layout.header + layout.table + layout.padding)
            output_sha = hashlib.sha256(layout.header + table + padding)
            payload_sha = hashlib.sha256()
            with os.fdopen(fd, "wb", buffering=0) as output:
                if progress is not None:
                    progress(dict(phase="created", temporary_path=str(temp), bytes=0, total=layout.payload_bytes))
                _write_all(output, layout.header)
                _write_all(output, table)
                _write_all(output, padding)
                handle.seek(layout.payload_start)
                done = checked = 0
                while done < layout.payload_bytes:
                    _assert_stable(handle, layout)
                    data = handle.read(min(chunk_bytes, layout.payload_bytes - done))
                    if not data:
                        raise PackError("short read during payload copy")
                    _write_all(output, data)
                    source_sha.update(data)
                    output_sha.update(data)
                    payload_sha.update(data)
                    done += len(data)
                    if done - checked >= SPACE_CHECK_INTERVAL or done == layout.payload_bytes:
                        _space(destination.parent, layout.payload_bytes - done, reserve_bytes)
                        checked = done
                    if progress is not None:
                        progress(dict(phase="copy", bytes=done, total=layout.payload_bytes, temporary_path=str(temp)))
                _assert_stable(handle, layout)
                os.fsync(output.fileno())
            verified = _verify_destination(layout, temp, payload_sha.hexdigest(), output_sha.hexdigest(), chunk_bytes, progress)
            _assert_stable(handle, layout)
            result = dict(schema=1, kind="expert_alignment_repack_receipt", status="verified",
                          created_at_utc=datetime.now(timezone.utc).isoformat(), source=str(source), destination=str(destination),
                          receipt=str(receipt), source_stat=layout.fingerprint,
                          source_path_stat=layout.path_fingerprint, source_version=1,
                          layers=LAYERS, experts_per_layer=EXPERTS, slot_bytes=SLOT_BYTES,
                          source_bytes=layout.size, source_payload_start=layout.payload_start,
                          inserted_bytes=delta, payload_bytes=layout.payload_bytes,
                          source_file_sha256=source_sha.hexdigest(), source_payload_sha256=payload_sha.hexdigest(),
                          source_header_sha256=hashlib.sha256(layout.header).hexdigest(),
                          source_table_sha256=hashlib.sha256(layout.table).hexdigest(),
                          slot_metadata_byte_equal_except_offsets=True, complete_destination_readback=True,
                          chunk_bytes=chunk_bytes, reserve_bytes=reserve_bytes, initial_available_bytes=initial_free,
                          max_output_bytes=max_output_bytes, elapsed_before_publish_s=time.perf_counter() - began,
                          checksum_scope="SHA-256 covers all payload bytes including record padding; table hashes are metadata only",
                          source_stability_scope="Observed fstat/path identity, size, mtime and ctime; no filesystem snapshot or writer lock",
                          publication="atomic no-replace rename per file; pack and receipt are not one atomic transaction",
                          O_DIRECT_implemented=False, performance_claim=False, **verified)
            receipt_raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
            if len(receipt_raw) > RECEIPT_RESERVE:
                raise PackError("receipt exceeds its reserved size")
            receipt_fd, receipt_name = tempfile.mkstemp(prefix=receipt.name + ".partial-", suffix=".tmp", dir=receipt.parent)
            receipt_temp = Path(receipt_name)
            with os.fdopen(receipt_fd, "wb", buffering=0) as output:
                _write_all(output, receipt_raw)
                os.fsync(output.fileno())
            _assert_stable(handle, layout)
            _publish_noreplace(temp, destination)
            published = True
            _fsync_parent(destination)
            _publish_noreplace(receipt_temp, receipt)
            _fsync_parent(receipt)
            return dict(result, receipt_file_sha256=hashlib.sha256(receipt_raw).hexdigest())
    except BaseException as exc:
        retained = []
        cleanup_errors = []
        for path in (temp, receipt_temp):
            if path is not None and path.exists():
                if keep_partial or published:
                    retained.append(str(path))
                else:
                    try:
                        path.unlink()
                    except OSError as cleanup_error:
                        retained.append(str(path))
                        cleanup_errors.append(str(cleanup_error))
        exc.repack_failure = dict(destination_published=published, destination=str(destination),
                                  partial_policy="retain" if keep_partial or published else "remove",
                                  retained_temporary_files=retained, cleanup_errors=cleanup_errors)
        raise


def _gib(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("GiB values must be finite and nonnegative")
    return int(value * 1024**3)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true", help="Read metadata and check space; do not read/hash payload or create files")
    action.add_argument("--verify-only", action="store_true", help="Independently compare source and an existing destination")
    parser.add_argument("--receipt", help="New receipt path in the destination directory")
    parser.add_argument("--chunk-mib", type=int, default=8, choices=range(1, 17))
    parser.add_argument("--reserve-gib", type=_gib, default=DEFAULT_RESERVE)
    parser.add_argument("--max-output-gib", type=_gib, default=DEFAULT_MAX_OUTPUT)
    parser.add_argument("--keep-partial", action="store_true", help="Retain named temporary files on failure/interruption")
    args = parser.parse_args(argv)
    last = [0.]
    def progress(event):
        now = time.monotonic()
        if event["phase"] == "created" or now - last[0] >= 5 or event["bytes"] == event["total"]:
            print(json.dumps(event), file=sys.stderr, flush=True)
            last[0] = now
    try:
        if args.verify_only:
            if args.receipt:
                parser.error("--verify-only prints evidence and does not write a receipt")
            result = verify(args.source, args.destination, chunk_bytes=args.chunk_mib * 1024**2, progress=progress)
        elif args.plan:
            result = plan(args.source, args.destination, reserve_bytes=args.reserve_gib,
                          max_output_bytes=args.max_output_gib, receipt=args.receipt)
        else:
            result = repack(args.source, args.destination, chunk_bytes=args.chunk_mib * 1024**2,
                            reserve_bytes=args.reserve_gib, max_output_bytes=args.max_output_gib,
                            keep_partial=args.keep_partial, receipt=args.receipt, progress=progress)
        print(json.dumps(result, indent=2))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps(dict(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                              error=type(exc).__name__, message=str(exc),
                              **getattr(exc, "repack_failure", {}))), file=sys.stderr, flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
