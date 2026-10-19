"""Sequential reads of the dense EXL3 tensors. Routed experts stay in the pack."""
import json
import hashlib
import os
import struct

import torch

_DTYPE = {
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I16": torch.int16,
    "I8": torch.int8,
    "I32": torch.int32,
    "I64": torch.int64,
    "BF16": torch.bfloat16,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E4M3FN": torch.float8_e4m3fn,
}


class WeightStore:
    def __init__(self, model_dir: str):
        self.dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json"), "rb") as f:
            source_index = f.read()
        index = json.loads(source_index)
        self.wm = index["weight_map"]
        local_dir = os.environ.get("DS41_DENSE_CACHE")
        if local_dir:
            local_dir = os.path.abspath(local_dir)
            with open(os.path.join(local_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
                local_index = json.load(f)
            if local_index.get("metadata",{}).get("source_index_sha256") != hashlib.sha256(source_index).hexdigest():
                raise ValueError("Dense cache was not packed from this source index")
            local = local_index["weight_map"]
            unknown = set(local) - set(self.wm)
            if unknown:
                raise ValueError(f"Dense cache has {len(unknown)} tensors absent from the source checkpoint")
            self.wm.update({name: os.path.join(local_dir, shard) for name, shard in local.items()})
        self.headers = {}
        self.cache = {}

    def has(self, name: str) -> bool:
        return name in self.wm

    def _header(self, shard: str):
        if shard not in self.headers:
            path = os.path.join(self.dir, shard)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            self.headers[shard] = (8 + n, hdr)
        return self.headers[shard]

    def meta(self, name: str):
        shard = self.wm[name]
        data_off, hdr = self._header(shard)
        meta = hdr[name]
        a, b = meta["data_offsets"]
        return shard, data_off + a, b - a, meta["dtype"], meta["shape"]

    def preload(self, names):
        pending = []
        for name in names:
            if name in self.cache or name not in self.wm:
                continue
            shard, off, nbytes, dtype, shape = self.meta(name)
            pending.append((shard, off, name, nbytes, dtype, shape))
        pending.sort()
        fds = {}
        try:
            moved = 0
            for shard, off, name, nbytes, dtype, shape in pending:
                fd = fds.get(shard)
                if fd is None:
                    fd = os.open(os.path.join(self.dir, shard), os.O_RDONLY)
                    fds[shard] = fd
                tensor = torch.empty(shape, dtype=_DTYPE[dtype])
                if tensor.numel() * tensor.element_size() != nbytes:
                    raise ValueError(f"Tensor byte length inconsistent with dtype/shape: {name}")
                target = memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B")
                moved_tensor = 0
                while moved_tensor < nbytes:
                    got = os.preadv(fd, [target[moved_tensor:min(nbytes,moved_tensor+8*1024*1024)]], off+moved_tensor)
                    if not got:
                        raise RuntimeError(f"short read {name}")
                    moved_tensor += got
                self.cache[name] = tensor
                moved += nbytes
            return moved
        finally:
            for fd in fds.values():
                os.close(fd)

    def get(self, name: str) -> torch.Tensor:
        if name not in self.cache:
            self.preload([name])
        return self.cache[name]

    def release(self):
        """Drop CPU copies once their GPU owners have completed their transfers.

        The checkpoint remains the source of truth and get() can reload a tensor.
        Keeping these copies consumes several GiB that are more useful for experts.
        """
        nbytes = sum(t.numel() * t.element_size() for t in self.cache.values())
        self.cache.clear()
        return nbytes


def _decode(buf: bytes, dtype: str, shape) -> torch.Tensor:
    torch_dtype = _DTYPE[dtype]
    t = torch.frombuffer(bytearray(buf), dtype=torch_dtype).clone()
    return t.reshape(shape)
