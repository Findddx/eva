"""Sparse, lossless RANMA MXFP8 Engram rows; no llama.cpp dependency.

Only GGUF headers and requested 264-byte rows are read. GGML type 320 is a
RANMA extension: 256 E4M3 bytes followed by eight E8M0 scale bytes. It is not
NVFP4, a rotation, or a lossy quantization of the original FP8 checkpoint.
"""
import bisect
import os
import re
import struct
import threading
from collections import OrderedDict

MXFP8 = 320
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
            6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_header(stream):
    """Read a GGUF v2/v3 header from a seekable binary stream."""
    def take(n):
        if n < 0 or n > 16 * 1024 * 1024:
            raise ValueError("unreasonable GGUF header length")
        data = stream.read(n)
        if len(data) != n:
            raise ValueError("truncated GGUF header")
        return data

    def unpack(fmt):
        return struct.unpack("<" + fmt, take(struct.calcsize("<" + fmt)))[0]

    def string():
        return take(unpack("Q")).decode("utf-8")

    def value(kind):
        if kind == 8:
            return string()
        if kind == 9:
            itemtype, count = unpack("I"), unpack("Q")
            if count > 1_000_000 or itemtype == 9:
                raise ValueError("unsupported GGUF metadata array")
            return [value(itemtype) for _ in range(count)]
        if kind not in _SCALARS:
            raise ValueError(f"unsupported GGUF metadata type {kind}")
        return unpack(_SCALARS[kind])

    if take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = unpack("I")
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    ntensors, nkv = unpack("Q"), unpack("Q")
    if ntensors > 100_000 or nkv > 100_000:
        raise ValueError("unreasonable GGUF header counts")
    metadata = {}
    for _ in range(nkv):
        key = string()
        metadata[key] = value(unpack("I"))
    tensors = []
    for _ in range(ntensors):
        name = string()
        ndim = unpack("I")
        if ndim > 4:
            raise ValueError("unsupported Engram tensor rank")
        shape = [unpack("Q") for _ in range(ndim)]
        tensors.append(dict(name=name, shape=shape, type=unpack("I"), offset=unpack("Q")))
    alignment = int(metadata.get("general.alignment", 32))
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("invalid GGUF alignment")
    data_offset = (stream.tell() + alignment - 1) // alignment * alignment
    return dict(version=version, metadata=metadata, tensors=tensors, data_offset=data_offset)


def _fnv1a(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


class GGUFEngramTable:
    def __init__(self, rows, dim, pieces, row_cache_bytes=8 * 1024**2):
        self.rows, self.dim = int(rows), int(dim)
        if self.dim != 256:
            raise ValueError("RANMA MXFP8 Engram rows must have width 256")
        self.row_bytes = 264
        self.pieces = pieces
        self.starts = [piece[0] for piece in pieces]
        self.cache_limit = max(0, int(row_cache_bytes) // self.row_bytes)
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.reads = self.bytes = self.cache_hits = 0

    def close(self):
        with self.lock:
            for _, _, fd, _ in self.pieces:
                os.close(fd)
            self.pieces = []
            self.cache.clear()

    def _raw_row(self, row):
        if not 0 <= row < self.rows:
            raise IndexError(f"Engram row {row} outside [0,{self.rows})")
        if row in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(row)
            return self.cache[row]
        piece = self.pieces[bisect.bisect_right(self.starts, row) - 1]
        start, count, fd, offset = piece
        if row >= start + count:
            raise ValueError("Engram piece coverage gap")
        raw = os.pread(fd, self.row_bytes, offset + (row - start) * self.row_bytes)
        if len(raw) != self.row_bytes:
            raise IOError("short MXFP8 Engram row read")
        self.reads += 1
        self.bytes += len(raw)
        if self.cache_limit:
            self.cache[row] = raw
            while len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
        return raw

    def lookup(self, ids):
        import torch

        flat = ids.reshape(-1).tolist()
        if not flat:
            return torch.empty(*ids.shape, self.dim, dtype=torch.bfloat16)
        unique = list(dict.fromkeys(flat))
        with self.lock:
            joined = bytearray().join(self._raw_row(row) for row in unique)
        raw = torch.frombuffer(joined, dtype=torch.uint8).reshape(-1, self.row_bytes)
        values = raw[:, :256].contiguous().view(torch.float8_e4m3fn).float().reshape(-1, 8, 32)
        scales_raw = raw[:, 256:].to(torch.int32)
        scales = torch.exp2(scales_raw.float() - 127)
        scales = torch.where(scales_raw == 255, torch.full_like(scales, torch.nan), scales)
        decoded = (values * scales.unsqueeze(-1)).reshape(-1, 256).to(torch.bfloat16)
        # Preserve reference FP8 NaNs, rather than the RANMA generic gather's
        # NaN-to-zero convention. Finite checkpoint values are identical.
        order = {row: index for index, row in enumerate(unique)}
        indices = torch.tensor([order[row] for row in flat], dtype=torch.long)
        return decoded[indices].reshape(*ids.shape, self.dim)


def from_index(index_path, expected_rows=(384006168, 384016682), head_dim=256,
               expected_source_revision="1bae0e1", row_cache_bytes=8 * 1024**2):
    """Validate a complete RANMA table set and return tables in index order.

    Structural checks cannot distinguish pre-1017 and 1017 table contents.
    Source revision is checked explicitly in addition to each piece's FNV check.
    Download integrity should additionally use the publisher's SHA-256 manifest.
    """
    with open(index_path, "rb") as f:
        index = read_header(f)
    meta = index["metadata"]
    if meta.get("general.architecture") != "deepseek41" or meta.get("general.type") != "engram_tables" or meta.get("engram_tables.type") != "MXFP8":
        raise ValueError("expected a RANMA MXFP8 Engram table index")
    if meta.get("engram_tables.version") != 1 or meta.get("engram_tables.kind") != "engram":
        raise ValueError("unsupported RANMA Engram table set version/kind")
    if expected_source_revision:
        revision = str(meta.get("engram_tables.source_revision", ""))
        if not revision.startswith(expected_source_revision):
            raise ValueError(f"Engram source revision {revision!r} does not match {expected_source_revision!r}")
    names = meta["engram_tables.names"]
    rows = list(meta["engram_tables.rows"])
    parts = list(meta["engram_tables.parts"])
    part_rows = list(meta["engram_tables.part_rows"])
    if names != ["blk.1.engram_embd", "blk.14.engram_embd"] or rows != list(expected_rows):
        raise ValueError("Engram table names/rows do not match the model config")
    checks = list(meta["engram_tables.piece_check"])
    if len(parts) != 2 or len(part_rows) != 2 or len(checks) != sum(parts):
        raise ValueError("inconsistent Engram table layout metadata")
    match = re.match(r"^(.*)-00001-of-(\d{5})\.gguf$", os.fspath(index_path))
    count = int(meta.get("split.count", 1))
    if count != sum(parts) + 1 or match is None or int(match.group(2)) != count:
        raise ValueError("expected a split MXFP8 index ending in -00001-of-NNNNN.gguf")
    if index["tensors"] or int(meta.get("split.no", 0)) != 0:
        raise ValueError("split Engram index must be metadata only")
    tables, opened = [], []
    try:
        piece_no = 0
        for table_id, name in enumerate(names):
            pieces = []
            cursor = 0
            for part in range(parts[table_id]):
                piece_no += 1
                path = f"{match.group(1)}-{piece_no + 1:05d}-of-{count:05d}.gguf"
                with open(path, "rb") as f:
                    hdr = read_header(f)
                if int(hdr["metadata"].get("split.no", -1)) != piece_no or int(hdr["metadata"].get("split.count", -1)) != count:
                    raise ValueError("Engram piece split metadata mismatch")
                if len(hdr["tensors"]) != 1:
                    raise ValueError("expected exactly one MXFP8 tensor per piece")
                tensor = hdr["tensors"][0]
                count_rows = min(part_rows[table_id], rows[table_id] - cursor)
                if tensor["name"] != f"{name}.p{part}.weight" or tensor["type"] != MXFP8 or tensor["shape"] != [head_dim, count_rows]:
                    raise ValueError("Engram piece tensor name/type/shape mismatch")
                fd = os.open(path, os.O_RDONLY)
                opened.append(fd)
                offset = hdr["data_offset"] + tensor["offset"]
                nbytes = count_rows * 264
                if count_rows <= 0 or os.fstat(fd).st_size < offset + nbytes:
                    raise ValueError("Engram piece file is too short")
                ncheck = min(4096, nbytes)
                edges = os.pread(fd, ncheck, offset) + os.pread(fd, ncheck, offset + nbytes - ncheck)
                if _fnv1a(edges) != checks[piece_no - 1]:
                    raise ValueError("Engram piece integrity check failed")
                pieces.append((cursor, count_rows, fd, offset))
                cursor += count_rows
            if cursor != rows[table_id]:
                raise ValueError("Engram pieces do not cover all rows")
            tables.append(GGUFEngramTable(rows[table_id], head_dim, pieces, row_cache_bytes))
        return tables
    except Exception:
        for fd in opened:
            os.close(fd)
        raise
