"""Experimental cross-layer router prediction; actual expert selection is untouched.

Predictions are used only for IO scheduling. The default records recall without
issuing reads, so bandwidth waste can be measured before enabling prefetch.
"""
import time

import torch
import torch.nn.functional as F


class RouterLookahead:
    def __init__(self, model, horizons=(1, 2, 4), candidates=12, prefetch=False,
                 byte_budget=64 * 1024**2, pending_limit=12):
        self.model, self.cache = model, model.cache
        self.horizons = tuple(sorted(set(int(h) for h in horizons)))
        if not self.horizons or min(self.horizons) < 1 or candidates < 1:
            raise ValueError("lookahead requires positive horizons and candidates")
        self.candidates, self.prefetch = candidates, prefetch
        self.byte_budget, self.pending_limit = int(byte_budget), int(pending_limit)
        self.original = self.cache.moe
        self.predictions = {}
        self.position = None
        self.remaining_bytes = 0
        self.stats = dict(calls=0, prediction_ms=0., submitted_reads=0, submitted_bytes=0,
                          headroom_skips=0)
        self.recall = {h: dict(actual=0, predicted=0, matched=0, cold=0, cold_matched=0,
                               cold_bytes=0, cold_matched_bytes=0) for h in self.horizons}

    def attach(self):
        self.cache.moe = self.moe

    def detach(self):
        self.cache.moe = self.original

    def begin(self, position):
        self.position = int(position)
        self.predictions.clear()
        self.remaining_bytes = self.byte_budget

    def _resident(self, key):
        return key in self.cache.frozen or key in self.cache.vram or key in self.cache.ram

    def _score_actual(self, layer, actual):
        cold = [expert for expert in actual if not self._resident((layer, expert))]
        for horizon in self.horizons:
            prediction = self.predictions.pop((layer, horizon), None)
            if prediction is None:
                continue
            stats = self.recall[horizon]
            stats["actual"] += len(actual)
            stats["predicted"] += len(prediction)
            stats["matched"] += len(set(actual) & prediction)
            stats["cold"] += len(cold)
            stats["cold_matched"] += sum(e in prediction for e in cold)
            for expert in cold:
                nbytes = self.cache._fields(layer, expert)[1]
                stats["cold_bytes"] += nbytes
                stats["cold_matched_bytes"] += nbytes * (expert in prediction)

    def _submit(self, layer, ids):
        if not self.prefetch:
            return
        for expert in ids:
            key = (layer, int(expert))
            if self._resident(key) or key in self.cache._pending:
                continue
            nbytes = self.cache._fields(*key)[1]
            if nbytes > self.remaining_bytes or len(self.cache._pending) >= self.pending_limit:
                break
            try:
                self.cache._submit_read(key)
            except MemoryError:
                self.stats["headroom_skips"] += 1
                break
            self.remaining_bytes -= nbytes
            self.stats["submitted_reads"] += 1
            self.stats["submitted_bytes"] += nbytes

    def moe(self, layer, x, indices, weights, k, limit, inter):
        if self.position is None or x.reshape(-1, x.shape[-1]).shape[0] != 1:
            return self.original(layer, x, indices, weights, k, limit, inter)
        actual = list(dict.fromkeys(int(e) for e in indices.detach().reshape(-1).tolist()))
        self._score_actual(layer, actual)
        start = time.perf_counter()
        predictions, targets = [], []
        current = self.model.layers[layer]
        flat = x.reshape(1, -1).float()
        # Undo only the current learned RMS scale, then apply the future scale.
        # Future residual evolution remains unknown: this is a measured predictor.
        scale = current.ffn_norm.float()
        unscaled = torch.where(scale.abs() > 1e-6, flat / scale, flat)
        for horizon in self.horizons:
            target = layer + horizon
            if target >= len(self.model.layers):
                continue
            block = self.model.layers[target]
            scores = F.softplus(F.linear(unscaled * block.ffn_norm.float(), block.gate_w.float())).sqrt()
            scores = scores + block.gate_b.float()
            count = min(self.candidates, scores.shape[-1])
            predictions.append(scores.topk(count, dim=-1).indices)
            targets.append((target, horizon))
        if predictions:
            # A single device-to-host rendezvous for all future layers.
            host = torch.cat(predictions, dim=0).detach().tolist()
            for (target, horizon), ids in zip(targets, host):
                self.predictions[(target, horizon)] = set(ids)
                self._submit(target, ids)
        self.stats["calls"] += 1
        self.stats["prediction_ms"] += (time.perf_counter() - start) * 1000
        return self.original(layer, x, indices, weights, k, limit, inter)

    def report(self):
        bins = {}
        for horizon, stats in self.recall.items():
            bins[str(horizon)] = dict(stats,
                recall=stats["matched"] / stats["actual"] if stats["actual"] else None,
                precision=stats["matched"] / stats["predicted"] if stats["predicted"] else None,
                cold_recall=stats["cold_matched"] / stats["cold"] if stats["cold"] else None,
                cold_byte_recall=stats["cold_matched_bytes"] / stats["cold_bytes"] if stats["cold_bytes"] else None)
        return dict(self.stats, horizons=bins, candidates=self.candidates,
                    mode="prefetch" if self.prefetch else "diagnostic", byte_budget=self.byte_budget)
