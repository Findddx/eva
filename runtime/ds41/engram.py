"""Exact token hashes and sparse, scaled FP8 Engram table reads.

The hash algorithm is the MIT-licensed DeepSeek reference implementation in
ref/inference/engram.py. Tables are read by row; they never enter WeightStore's
whole-tensor cache. Missing tables fail before loading the dense backbone.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch

from ds41.engram_hash import EngramLayout, NgramHashState


def missing_engram_tensors(args, store):
    return [
        f"layers.{layer}.engram.embed.{suffix}"
        for layer in args.engram_layer_ids
        for suffix in ("weight", "scale")
        if not store.has(f"layers.{layer}.engram.embed.{suffix}")
    ]


class _TokenizerAdapter:
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.backend_tokenizer = Tokenizer.from_file(path)

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


class EngramTable:
    """A table descriptor references raw tensors or offsets into safetensors.

    Manifest entries keyed by layer id contain weight_path, scale_path,
    weight_offset/scale_offset (default 0), scale_dtype (default F8_E8M0),
    num_embeddings and head_dim. Relative paths resolve against the manifest.
    """
    def __init__(self, entry, expected_rows, head_dim):
        self.rows = int(entry["num_embeddings"])
        self.dim = int(entry["head_dim"])
        if (self.rows, self.dim) != (expected_rows, head_dim):
            raise ValueError("Engram table dimensions do not match the model config")
        self.block = int(entry.get("block_size", 32))
        self.scale_dtype = entry.get("scale_dtype", "F8_E8M0")
        self.weight_offset = int(entry.get("weight_offset", 0))
        self.scale_offset = int(entry.get("scale_offset", 0))
        self.scale_itemsize = {"F32": 4, "BF16": 2, "F16": 2}.get(self.scale_dtype, 1)
        self.scale_width = self.dim // self.block
        self.weight_fd = os.open(entry["weight_path"], os.O_RDONLY)
        self.scale_fd = os.open(entry["scale_path"], os.O_RDONLY)
        if os.fstat(self.weight_fd).st_size < self.weight_offset + self.rows * self.dim:
            self.close()
            raise ValueError("Engram weight file is shorter than its declared table")
        if os.fstat(self.scale_fd).st_size < self.scale_offset + self.rows * self.scale_width * self.scale_itemsize:
            self.close()
            raise ValueError("Engram scale file is shorter than its declared table")

    def close(self):
        for name in ("weight_fd", "scale_fd"):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)

    def _row(self, row):
        if not 0 <= row < self.rows:
            raise IndexError(f"Engram row {row} outside [0,{self.rows})")
        values = os.pread(self.weight_fd, self.dim, self.weight_offset + row * self.dim)
        nscale = self.scale_width * self.scale_itemsize
        scales = os.pread(self.scale_fd, nscale, self.scale_offset + row * nscale)
        if len(values) != self.dim or len(scales) != nscale:
            raise IOError("short Engram table row read")
        v = torch.frombuffer(bytearray(values), dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        if self.scale_dtype in ("F8_E8M0", "F8_E8M0FNU", "E8M0", "UE8M0"):
            raw = torch.frombuffer(bytearray(scales), dtype=torch.uint8).to(torch.int32)
            # E8M0 byte e denotes 2**(e-127); 255 is NaN.
            s = torch.exp2(raw.float() - 127)
            s = torch.where(raw == 255, torch.full_like(s, torch.nan), s)
        else:
            dtype = {"F8_E4M3": torch.float8_e4m3fn, "F8_E4M3FN": torch.float8_e4m3fn,
                     "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}[self.scale_dtype]
            s = torch.frombuffer(bytearray(scales), dtype=dtype).float()
        return (v.reshape(-1, self.block) * s.unsqueeze(-1)).flatten().to(torch.bfloat16)

    def lookup(self, ids):
        flat = ids.reshape(-1).tolist()
        # The same n-gram may repeat many times in code; coalesce within the
        # request without retaining an unbounded RAM cache.
        unique = {row: self._row(row) for row in dict.fromkeys(flat)}
        return torch.stack([unique[row] for row in flat]).reshape(*ids.shape, self.dim)


class EngramRuntime:
    def __init__(self, args, store):
        self.args = args
        self.layout = EngramLayout.from_args(args)
        self.tables = []
        self.pool = ThreadPoolExecutor(max_workers=max(1, len(args.engram_layer_ids)))
        self.jobs = []
        manifest_path = os.environ.get("DS41_ENGRAM_MANIFEST")
        manifest = None
        if manifest_path:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        try:
            gguf = manifest is not None and manifest.get("format") == "ranma_mxfp8_gguf"
            if gguf:
                from ds41.engram_gguf import from_index
                index_path = manifest["index_path"]
                if not os.path.isabs(index_path):
                    index_path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), index_path)
                self.tables = from_index(index_path, expected_rows=self.layout.num_embeddings,
                                         head_dim=self.layout.head_dim)
            for slot, layer in enumerate(() if gguf else args.engram_layer_ids):
                if manifest is not None:
                    entry = dict(manifest.get("tables", manifest)[str(layer)])
                    for key in ("weight_path", "scale_path"):
                        if not os.path.isabs(entry[key]):
                            entry[key] = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), entry[key])
                else:
                    p = f"layers.{layer}.engram.embed."
                    wsh, woff, _, wdt, wshape = store.meta(p + "weight")
                    ssh, soff, _, sdt, sshape = store.meta(p + "scale")
                    if wdt not in ("F8_E4M3", "F8_E4M3FN") or sshape != [wshape[0], wshape[1] // 32]:
                        raise ValueError("unsupported Engram checkpoint table layout")
                    entry = dict(weight_path=os.path.join(store.dir, wsh), scale_path=os.path.join(store.dir, ssh),
                                 weight_offset=woff, scale_offset=soff, scale_dtype=sdt,
                                 num_embeddings=wshape[0], head_dim=wshape[1])
                self.tables.append(EngramTable(entry, self.layout.num_embeddings[slot], self.layout.head_dim))
            with torch.device("cpu"):
                self.hash = NgramHashState(args, self.layout, _TokenizerAdapter(os.path.join(store.dir, "tokenizer.json")))
        except Exception:
            self.close()
            raise

    @classmethod
    def from_store(cls, args, store):
        if not args.engram_layer_ids:
            return None
        missing = missing_engram_tensors(args, store)
        if missing and not os.environ.get("DS41_ENGRAM_MANIFEST"):
            message = "Missing required Engram tensors: " + ", ".join(missing)
            if os.environ.get("DS41_ALLOW_MISSING_ENGRAM") == "1":
                print("ABLATION: Engram DISABLED. " + message, flush=True)
                return None
            raise RuntimeError(message + ". Supply DS41_ENGRAM_MANIFEST or explicitly set DS41_ALLOW_MISSING_ENGRAM=1 for an ablation.")
        return cls(args, store)

    def kick(self, input_ids, pos0):
        for job in self.jobs:
            job.result()
        ids = self.hash(input_ids.detach().to(device="cpu", dtype=torch.long), pos0)
        self.jobs = [self.pool.submit(table.lookup, ids[:, :, slot].clone()) for slot, table in enumerate(self.tables)]

    def wait(self, slot):
        return self.jobs[slot].result()

    def close(self):
        self.pool.shutdown(wait=True)
        for table in self.tables:
            table.close()
