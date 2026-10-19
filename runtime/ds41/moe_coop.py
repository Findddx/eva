"""Decode MoE: one fused EXL3 launch for the six routed experts of a layer.

exllamav3's coop kernel reads expert addresses from GPU pointer tables, so a
resident expert is a table update rather than a new LinearEXL3.
"""
import os

import torch


def coop_ctr_len(slots: int, rows: int, inter: int, hidden_out: int) -> int:
    return slots * (inter // 128) + rows * (hidden_out // 128) + 2 + (slots + 1) + slots


class CoopRunner:
    def __init__(self, device, topk: int, hidden: int, inter: int, rows: int = 1):
        # Automatic geometry changes at multi-row thresholds and can change
        # FP16 rounding. Keep target and verification arithmetic identical by
        # default; the explicit opt-out is for separately validated tuning.
        if os.environ.get("DS41_STRICT_BATCH", "1") == "1":
            os.environ.setdefault("EXL3_MOE_COOP_WIDE", "0")
            os.environ.setdefault("EXL3_MOE_COOP_KSPLIT", "1")
        self.topk = topk
        self.rows = rows
        slots = topk * rows
        if rows < 1 or slots > 256:
            raise ValueError("Cooperative EXL3 supports 1..256 routed slots")
        self.hi = hidden
        self.inter = inter
        self.ho = hidden
        opt16 = dict(device=device, dtype=torch.float16)
        self.had_g = torch.empty(slots, hidden, **opt16)
        self.had_u = torch.empty(slots, hidden, **opt16)
        self.gu_g = torch.empty(slots, inter, **opt16)
        self.gu_u = torch.empty(slots, inter, **opt16)
        self.act_out = torch.empty(slots, inter, **opt16)
        self.d_out = torch.empty(slots, hidden, device=device, dtype=torch.float32)
        self.out = torch.empty(rows, hidden, device=device, dtype=torch.float32)
        self.rw = torch.empty(rows, topk, **opt16)
        self.ctr = torch.zeros(coop_ctr_len(slots, rows, inter, hidden), device=device, dtype=torch.int32)

    def run(self, x, sel, weights, packed, k, limit) -> torch.Tensor:
        from exllamav3.ext import exllamav3_ext as ext

        self.ctr.zero_()
        self.rw.copy_(weights.reshape(self.rows, self.topk).to(dtype=torch.float16))
        ext.exl3_moe_coop(
            x.reshape(self.rows, self.hi).contiguous(),
            sel.reshape(self.rows, self.topk).to(dtype=torch.long).contiguous(),
            self.rw,
            -1,
            0,
            self.hi,
            packed[0], packed[1], packed[2],
            packed[3], packed[4], packed[5],
            packed[6], packed[7], packed[8],
            None, None, None,
            float(k), float(k), float(k),
            False, True,
            0,
            float(limit),
            True,
            self.had_g, self.had_u,
            self.gu_g, self.gu_u,
            self.act_out, self.d_out,
            self.ctr, self.out,
            None, None,
        )
        return self.out
