"""Stream non-backbone-expert tensors into an independent local safetensors cache.

Pure Python/CPU, bounded copy buffers, no torch import and no tensor decoding.
The source model is read-only. This is an opt-in tool; importing it does no I/O.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import time
from pathlib import Path


ROUTED = re.compile(r"^layers\.\d+\.ffn\.experts\.")


def read_header(path):
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"truncated safetensors prefix: {path}")
        size = struct.unpack("<Q", prefix)[0]
        if size > 128 * 1024 * 1024:
            raise ValueError(f"unreasonable safetensors header: {path}")
        raw = f.read(size)
        if len(raw) != size:
            raise ValueError(f"truncated safetensors header: {path}")
    return 8 + size, json.loads(raw)


def plan(model_dir, max_shard_bytes):
    index_path = Path(model_dir) / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    wm = json.loads(index_bytes)["weight_map"]
    headers, records = {}, []
    for name, shard in wm.items():
        if ROUTED.match(name):
            continue
        source = Path(shard) if os.path.isabs(shard) else Path(model_dir) / shard
        source = source.resolve()
        if source not in headers:
            headers[source] = read_header(source)
        data_offset, header = headers[source]
        meta = header[name]
        begin, end = meta["data_offsets"]
        if begin < 0 or end < begin or data_offset + end > source.stat().st_size:
            raise ValueError(f"invalid tensor range: {name}")
        records.append(dict(name=name, source=str(source), offset=data_offset + begin,
                            size=end - begin, dtype=meta["dtype"], shape=meta["shape"]))
    # Source order is also destination order, avoiding random NAS reads.
    records.sort(key=lambda r: (r["source"], r["offset"]))
    shards, current, used = [], [], 0
    for record in records:
        if current and used + record["size"] > max_shard_bytes:
            shards.append(current)
            current, used = [], 0
        current.append(record)
        used += record["size"]
    if current:
        shards.append(current)
    return shards, hashlib.sha256(index_bytes).hexdigest()


def _header_for(records, source_digest):
    header = {"__metadata__": {"format": "pt", "ds41_source_index_sha256": source_digest}}
    offset = 0
    for r in records:
        header[r["name"]] = dict(dtype=r["dtype"], shape=r["shape"], data_offsets=[offset, offset + r["size"]])
        offset += r["size"]
    data = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    data += b" " * ((-len(data)) % 8)
    return struct.pack("<Q", len(data)) + data


def pack(model_dir, destination, *, max_shard_bytes=4 * 1024**3, buffer_bytes=8 * 1024**2, plan_only=False):
    shards, digest = plan(model_dir, max_shard_bytes)
    total_size = sum(r["size"] for shard in shards for r in shard)
    summary = dict(source_index_sha256=digest, tensors=sum(map(len, shards)),
                   tensor_bytes=total_size, shard_count=len(shards))
    print(json.dumps(summary), flush=True)
    if plan_only:
        return summary
    destination = Path(destination).resolve()
    source_root = Path(model_dir).resolve()
    if destination == source_root or source_root in destination.parents:
        raise ValueError("dense cache must be outside the read-only source model directory")
    destination.mkdir(parents=True, exist_ok=True)
    final_index = destination / "model.safetensors.index.json"
    if final_index.exists():
        existing = json.loads(final_index.read_text(encoding="utf-8"))
        if existing.get("metadata", {}).get("source_index_sha256") != digest:
            raise FileExistsError("destination contains a cache for a different source index")
    source_fds, weight_map, artifacts = {}, {}, []
    start = time.perf_counter()
    try:
        for number, records in enumerate(shards, 1):
            name = f"dense-{digest[:12]}-{number:05d}-of-{len(shards):05d}.safetensors"
            target = destination / name
            header = _header_for(records, digest)
            expected_size = len(header) + sum(r["size"] for r in records)
            if target.exists():
                if target.stat().st_size != expected_size:
                    raise ValueError(f"existing cache shard has incorrect length: {target}")
                with target.open("rb") as f:
                    if f.read(len(header)) != header:
                        raise ValueError(f"existing cache shard header differs: {target}")
                print(f"reuse {target.name}", flush=True)
            else:
                temporary = target.with_name(target.name + f".{os.getpid()}.partial")
                checksum = hashlib.sha256()
                try:
                    with temporary.open("xb") as out:
                        out.write(header)
                        checksum.update(header)
                        for r in records:
                            fd = source_fds.get(r["source"])
                            if fd is None:
                                fd = os.open(r["source"], os.O_RDONLY)
                                source_fds[r["source"]] = fd
                            offset, remaining = r["offset"], r["size"]
                            while remaining:
                                if hasattr(os, "pread"):
                                    data = os.pread(fd, min(buffer_bytes, remaining), offset)
                                else:
                                    os.lseek(fd, offset, os.SEEK_SET)
                                    data = os.read(fd, min(buffer_bytes, remaining))
                                if not data:
                                    raise IOError(f"short source read: {r['name']}")
                                out.write(data)
                                checksum.update(data)
                                offset += len(data)
                                remaining -= len(data)
                        out.flush()
                        os.fsync(out.fileno())
                    if temporary.stat().st_size != expected_size:
                        raise IOError("incomplete output safetensors shard")
                    os.replace(temporary, target)
                    artifacts.append(dict(file=name, size=expected_size, sha256=checksum.hexdigest()))
                    print(f"wrote {name} {expected_size / 1024**3:.3f} GiB", flush=True)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            weight_map.update({r["name"]: name for r in records})
        index = dict(metadata={"total_size": total_size, "source_index_sha256": digest,
                               "ds41_cache_kind": "dense-mtp-vision-excluding-backbone-routed"}, weight_map=weight_map)
        temporary_index = final_index.with_name(final_index.name + f".{os.getpid()}.partial")
        temporary_index.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_index, final_index)
        (destination / "copy-manifest.json").write_text(json.dumps(dict(summary, copied_shards=artifacts), indent=2) + "\n", encoding="utf-8")
    finally:
        for fd in source_fds.values():
            os.close(fd)
    summary["elapsed_seconds"] = time.perf_counter() - start
    summary["index_path"] = str(final_index)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--shard-gib", type=float, default=4)
    parser.add_argument("--buffer-mib", type=int, default=8)
    parser.add_argument("--plan", action="store_true", help="read only tensor headers; do not copy data")
    args = parser.parse_args()
    if args.shard_gib <= 0 or args.buffer_mib < 1:
        parser.error("positive shard size and buffer >= 1 MiB required")
    result = pack(args.model, args.destination, max_shard_bytes=int(args.shard_gib * 1024**3),
                  buffer_bytes=args.buffer_mib * 1024**2, plan_only=args.plan)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
