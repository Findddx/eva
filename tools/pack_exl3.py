#!/usr/bin/env python3
"""Pack routed EXL3 experts by streaming each NAS shard once.

A sequential pass writes each tensor to an aligned local expert record.
"""
import json
import os
import struct
import time

BASE = os.environ.get("DS41_MODEL", "")
OUT_DIR = os.environ.get("DS41_PACK", "")
ALIGN = 4096
MAGIC = b"DS41PK01"
N_LAYERS = 40
N_EXPERTS = 384
PROJS = ("w1", "w2", "w3")
KINDS = ("suh", "svh", "trellis")


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def align_up(n):
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def parse_expert(name):
    if not name.startswith("layers.") or ".ffn.experts." not in name:
        return None
    parts = name.split(".")
    try:
        layer = int(parts[1])
        ei = parts.index("experts")
        expert = int(parts[ei + 1])
        proj = parts[ei + 2]
        kind = parts[ei + 3]
    except (ValueError, IndexError):
        return None
    if proj not in PROJS or kind not in ("mul1",) + KINDS:
        return None
    if not (0 <= layer < N_LAYERS and 0 <= expert < N_EXPERTS):
        return None
    return layer, expert, proj, kind


def main():
    if not BASE or not OUT_DIR:
        raise ValueError("Set DS41_MODEL to the checkpoint directory and DS41_PACK to a new output directory")
    if os.path.exists(os.path.join(OUT_DIR, "experts.bin")):
        raise FileExistsError("Expert pack already exists; choose a new output directory")
    os.makedirs(OUT_DIR, exist_ok=True)
    bin_path = os.path.join(OUT_DIR, "experts.bin")
    print("headers", flush=True)
    index = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))
    wm = index["weight_map"]
    shard_meta = {}
    for shard in sorted(set(wm.values())):
        data_off, hdr = read_header(os.path.join(BASE, shard))
        shard_meta[shard] = (data_off, hdr)
        print("header", shard, flush=True)

    # tensor[(layer, expert, proj, kind)] = (shard, abs_off, nbytes, shape)
    tensors = {}
    for name, shard in wm.items():
        parsed = parse_expert(name)
        if not parsed:
            continue
        data_off, hdr = shard_meta[shard]
        meta = hdr[name]
        a, b = meta["data_offsets"]
        tensors[parsed] = (shard, data_off + a, b - a, meta.get("shape"))

    slot = struct.Struct("<QIHH" + "HHHHIIIiIII" * 3)
    SLOT = 128
    assert slot.size <= SLOT
    index_bytes = N_LAYERS * N_EXPERTS * SLOT
    payload_at = 8 + 16 + index_bytes

    copies = []
    mul1_patches = []  # (shard, src, layer, expert, proj_index)
    cursor = payload_at
    slot_blobs_meta = []
    for layer in range(N_LAYERS):
        for expert in range(N_EXPERTS):
            rec_off = cursor
            rel = 0
            proj_fields = []
            k0 = 0
            for pi, proj in enumerate(PROJS):
                suh = tensors[(layer, expert, proj, "suh")]
                svh = tensors[(layer, expert, proj, "svh")]
                tre = tensors[(layer, expert, proj, "trellis")]
                mul = tensors[(layer, expert, proj, "mul1")]
                in_f = suh[3][0]
                out_f = svh[3][0]
                k = tre[3][-1] // 16
                if pi == 0:
                    k0 = k
                rel_suh, rel_svh, rel_trellis = rel, rel + suh[2], rel + suh[2] + svh[2]
                copies.append((suh[0], suh[1], rec_off + rel_suh, suh[2]))
                copies.append((svh[0], svh[1], rec_off + rel_svh, svh[2]))
                copies.append((tre[0], tre[1], rec_off + rel_trellis, tre[2]))
                rel = rel_trellis + tre[2]
                proj_fields.append((in_f, out_f, k, suh[2], svh[2], tre[2], rel_suh, rel_svh, rel_trellis))
                mul1_patches.append((mul[0], mul[1], layer, expert, pi))
            padded = align_up(rel)
            slot_blobs_meta.append((rec_off, padded, k0, proj_fields))
            cursor = rec_off + padded

    print(f"payload {cursor/1024**3:.2f} GiB  copies {len(copies)}", flush=True)
    out = os.open(bin_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
    os.ftruncate(out, cursor)
    os.write(out, MAGIC + struct.pack("<IIII", 1, N_LAYERS, N_EXPERTS, SLOT))

    by_shard = {}
    for item in copies:
        by_shard.setdefault(item[0], []).append(item)
    for item in mul1_patches:
        by_shard.setdefault(item[0], []).append(("mul1",) + item)

    t0 = time.time()
    moved = 0
    # mul1 values collected then patched into the index
    mul1_val = [[[0, 0, 0] for _ in range(N_EXPERTS)] for _ in range(N_LAYERS)]
    for shard, items in by_shard.items():
        path = os.path.join(BASE, shard)
        size = os.path.getsize(path)
        fd = os.open(path, os.O_RDONLY)
        st = time.time()
        # Merge into a sequential sweep.
        spans = []
        for item in items:
            if item[0] == "mul1":
                _, _, src, layer, expert, pi = item
                spans.append((src, src + 4, ("mul1", layer, expert, pi)))
            else:
                _, src, dst, nbytes = item
                spans.append((src, src + nbytes, ("copy", dst)))
        spans.sort()
        chunk_n = 64 * 1024 * 1024
        pos = 0
        si = 0
        os.lseek(fd, 0, os.SEEK_SET)
        while si < len(spans):
            buf = os.read(fd, chunk_n)
            if not buf:
                raise RuntimeError(f"eof {shard} with {len(spans) - si} spans left")
            buf_lo = pos
            buf_hi = pos + len(buf)
            pos = buf_hi
            while si < len(spans) and spans[si][0] < buf_hi:
                start, end, op = spans[si]
                if start < buf_lo:
                    raise RuntimeError(f"span overlap {shard}")
                if end > buf_hi:
                    # Span crosses this chunk. Pull the tail with one more read.
                    tail = os.read(fd, end - buf_hi)
                    if len(tail) != end - buf_hi:
                        raise RuntimeError(f"short tail {shard}")
                    payload = buf[start - buf_lo :] + tail
                    pos = end
                    buf_hi = end
                    buf = b""
                    buf_lo = end
                else:
                    payload = buf[start - buf_lo : end - buf_lo]
                if op[0] == "copy":
                    os.pwrite(out, payload, op[1])
                    moved += len(payload)
                else:
                    _, layer, expert, pi = op
                    mul1_val[layer][expert][pi] = struct.unpack("<i", payload)[0]
                    moved += 4
                si += 1
        os.close(fd)
        dt = time.time() - st
        elapsed = time.time() - t0
        print(
            f"{shard} {size/1024**3:.2f} GiB in {dt:.1f}s  "
            f"useful {moved/1024**3:.2f} GiB  avg {moved/1024**3/elapsed:.2f} GiB/s",
            flush=True,
        )

    # Write the index now that mul1 scalars are known.
    blobs = []
    for layer in range(N_LAYERS):
        for expert in range(N_EXPERTS):
            rec_off, padded, k0, proj_fields = slot_blobs_meta[layer * N_EXPERTS + expert]
            fields = [rec_off, padded, k0, 1]
            for pi, (in_f, out_f, k, suh_b, svh_b, tr_b, rel_suh, rel_svh, rel_trellis) in enumerate(proj_fields):
                fields.extend(
                    [in_f, out_f, k, 0, suh_b, svh_b, tr_b, mul1_val[layer][expert][pi], rel_suh, rel_svh, rel_trellis]
                )
            blobs.append(slot.pack(*fields).ljust(SLOT, b"\x00"))
    os.pwrite(out, b"".join(blobs), 8 + 16)
    os.fsync(out)
    os.close(out)
    print(f"done {bin_path} {cursor/1024**3:.2f} GiB in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
