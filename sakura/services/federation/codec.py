"""Codec, FlatState, and anchor utilities for sakura federation."""

from __future__ import annotations

import hashlib
import math
from typing import List, Optional, Tuple

import numpy as np
import torch

BUCKET = 4096
TOPK_DENSITY = 0.1  # default kept fraction per bucket for the "topk" codec


def quantize_int8(
    x: torch.Tensor, ef: Optional[torch.Tensor]
) -> Tuple[bytes, torch.Tensor, float]:
    """Bucketed symmetric int8 with error feedback.

    x: fp32 CPU flat delta. ef: residual carried from the previous round
    (same shape) or None. Returns (payload, new_ef, contraction δ̂).
    payload = n_buckets fp32 scales || int8 data (padded to bucket multiple).
    """
    if ef is not None:
        x = x + ef
    if not torch.isfinite(x).all():
        raise FloatingPointError("non-finite delta entering quantizer")
    n = x.numel()
    pad = (-n) % BUCKET
    xp = torch.nn.functional.pad(x, (0, pad)) if pad else x
    b = xp.view(-1, BUCKET)
    scale = b.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = torch.clamp(torch.round(b / scale[:, None]), -127, 127).to(torch.int8)
    deq = q.float() * scale[:, None]
    ef_new = (b - deq).reshape(-1)[:n].contiguous()
    xn = float(x.norm()) or 1e-30
    contraction = 1.0 - float(ef_new.norm()) ** 2 / xn**2
    payload = scale.numpy().tobytes() + q.numpy().tobytes()
    return payload, ef_new, contraction


def dequantize_int8(payload: bytes, numel: int) -> torch.Tensor:
    n_buckets = (numel + BUCKET - 1) // BUCKET
    sbytes = n_buckets * 4
    scale = torch.frombuffer(bytearray(payload[:sbytes]), dtype=torch.float32)
    q = torch.frombuffer(bytearray(payload[sbytes:]), dtype=torch.int8).view(
        n_buckets, BUCKET
    )
    # Two separate elementwise ops on CPU — the bit-identity contract.
    deq = q.to(torch.float32)
    deq = deq.mul_(scale[:, None])
    return deq.reshape(-1)[:numel].contiguous()


def quantize_int4(
    x: torch.Tensor, ef: Optional[torch.Tensor]
) -> Tuple[bytes, torch.Tensor, float]:
    """Bucketed symmetric int4 (nibble-packed, 2 values/byte) with error
    feedback — same contract as quantize_int8 but ~8× vs fp32. Levels are
    [-7, 7]; payload = n_buckets fp32 scales || packed nibbles."""
    if ef is not None:
        x = x + ef
    if not torch.isfinite(x).all():
        raise FloatingPointError("non-finite delta entering quantizer")
    n = x.numel()
    pad = (-n) % BUCKET
    xp = torch.nn.functional.pad(x, (0, pad)) if pad else x
    b = xp.view(-1, BUCKET)
    scale = b.abs().amax(dim=1).clamp_min(1e-12) / 7.0
    q = torch.clamp(torch.round(b / scale[:, None]), -7, 7).to(torch.int8)
    deq = q.float() * scale[:, None]
    ef_new = (b - deq).reshape(-1)[:n].contiguous()
    xn = float(x.norm()) or 1e-30
    contraction = 1.0 - float(ef_new.norm()) ** 2 / xn**2
    qf = (q.reshape(-1).numpy().astype(np.uint8)) & 0x0F  # low nibble (two's-comp)
    packed = (qf[0::2] | (qf[1::2] << 4)).astype(np.uint8)  # BUCKET even -> exact
    payload = scale.numpy().tobytes() + packed.tobytes()
    return payload, ef_new, contraction


def dequantize_int4(payload: bytes, numel: int) -> torch.Tensor:
    n_buckets = (numel + BUCKET - 1) // BUCKET
    sbytes = n_buckets * 4
    scale = torch.frombuffer(bytearray(payload[:sbytes]), dtype=torch.float32)
    packed = np.frombuffer(bytearray(payload[sbytes:]), dtype=np.uint8)
    u = np.empty(packed.size * 2, dtype=np.uint8)
    u[0::2] = packed & 0x0F
    u[1::2] = (packed >> 4) & 0x0F
    q = u.astype(np.int8)
    q[q >= 8] -= 16  # nibble two's-complement -> signed [-8, 7]
    deq = torch.from_numpy(q).view(n_buckets, BUCKET).to(torch.float32)
    deq = deq.mul_(scale[:, None])
    return deq.reshape(-1)[:numel].contiguous()


def _topk_k(density: float) -> int:
    return max(1, math.ceil(density * BUCKET))


def quantize_topk(
    x: torch.Tensor, ef: Optional[torch.Tensor], density: float
) -> Tuple[bytes, torch.Tensor, float]:
    """Bucket-local top-k magnitude sparsification + int8 values, with full
    error feedback (the dropped + quantization residual is carried forward —
    Deep Gradient Compression). Indices are position-in-bucket (uint16, since
    BUCKET=4096 < 2^16), keeping the all-gather local to a bucket.
    payload = uint32 k || nb fp32 scales || nb·k uint16 indices || nb·k int8 vals."""
    if ef is not None:
        x = x + ef
    if not torch.isfinite(x).all():
        raise FloatingPointError("non-finite delta entering quantizer")
    n = x.numel()
    pad = (-n) % BUCKET
    xp = torch.nn.functional.pad(x, (0, pad)) if pad else x
    b = xp.view(-1, BUCKET)
    nb = b.shape[0]
    k = _topk_k(density)
    idx = torch.topk(b.abs(), k, dim=1).indices  # (nb, k) positions in bucket
    gathered = torch.gather(b, 1, idx)
    scale = gathered.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = torch.clamp(torch.round(gathered / scale[:, None]), -127, 127).to(torch.int8)
    deq_sparse = torch.zeros_like(b)
    deq_sparse.scatter_(1, idx, q.float() * scale[:, None])
    ef_new = (b - deq_sparse).reshape(-1)[:n].contiguous()
    xn = float(x.norm()) or 1e-30
    contraction = 1.0 - float(ef_new.norm()) ** 2 / xn**2
    payload = (
        np.uint32(k).tobytes()
        + scale.numpy().tobytes()
        + idx.to(torch.int32).numpy().astype(np.uint16).tobytes()
        + q.numpy().tobytes()
    )
    return payload, ef_new, contraction


def dequantize_topk(payload: bytes, numel: int) -> torch.Tensor:
    k = int(np.frombuffer(bytearray(payload[:4]), dtype=np.uint32)[0])
    nb = (numel + BUCKET - 1) // BUCKET
    off = 4
    scale = torch.frombuffer(bytearray(payload[off : off + nb * 4]), dtype=torch.float32)
    off += nb * 4
    idx = np.frombuffer(
        bytearray(payload[off : off + nb * k * 2]), dtype=np.uint16
    ).reshape(nb, k)
    off += nb * k * 2
    q = np.frombuffer(bytearray(payload[off : off + nb * k]), dtype=np.int8).reshape(nb, k)
    deq = torch.zeros(nb, BUCKET, dtype=torch.float32)
    vals = torch.from_numpy(q.astype(np.float32)) * scale[:, None]
    deq.scatter_(1, torch.from_numpy(idx.astype(np.int64)), vals)
    return deq.reshape(-1)[:numel].contiguous()


def _topk_density(kind: str) -> float:
    return float(kind.split(":", 1)[1]) if ":" in kind else TOPK_DENSITY


def codec_encode(
    x: torch.Tensor, ef: Optional[torch.Tensor], kind: str
) -> Tuple[bytes, torch.Tensor, float]:
    if kind == "int8":
        return quantize_int8(x, ef)
    if kind == "int4":
        return quantize_int4(x, ef)
    if kind.startswith("topk"):
        return quantize_topk(x, ef, _topk_density(kind))
    if kind == "fp32raw":  # lossless reference rung; EF stays zero
        if not torch.isfinite(x).all():
            raise FloatingPointError("non-finite delta")
        return x.numpy().tobytes(), torch.zeros_like(x), 1.0
    raise ValueError(kind)


def codec_decode(payload: bytes, numel: int, kind: str) -> torch.Tensor:
    if kind == "int8":
        return dequantize_int8(payload, numel)
    if kind == "int4":
        return dequantize_int4(payload, numel)
    if kind.startswith("topk"):
        return dequantize_topk(payload, numel)
    if kind == "fp32raw":
        return torch.frombuffer(bytearray(payload), dtype=torch.float32)[:numel].clone()
    raise ValueError(kind)


class FlatState:
    """Canonical flat views of a model: float params (sorted state_dict keys)
    on the quantized path; BN running_mean/var as a raw fp32 sideband;
    num_batches_tracked excluded entirely."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        sd = model.state_dict(keep_vars=True)
        pkeys, bkeys = [], []
        for k in sorted(sd.keys()):
            v = sd[k]
            if k.endswith("num_batches_tracked"):
                continue
            if isinstance(v, torch.nn.Parameter):
                if v.dtype.is_floating_point:
                    pkeys.append(k)
            elif k.endswith(("running_mean", "running_var")):
                bkeys.append(k)
        self.param_keys, self.bn_keys = pkeys, bkeys
        self.params = [sd[k] for k in pkeys]
        self.bn_bufs = [sd[k] for k in bkeys]
        self.offsets = []
        off = 0
        for p in self.params:
            self.offsets.append(off)
            off += p.numel()
        self.numel = off
        self.bn_offsets = []
        boff = 0
        for b in self.bn_bufs:
            self.bn_offsets.append(boff)
            boff += b.numel()
        self.bn_numel = boff
        self.device = self.params[0].device if self.params else torch.device("cpu")
        self._gpu_stage: Optional[torch.Tensor] = None

    # --- training-thread ops (touch live GPU tensors) ---
    def snapshot_params_to(self, flat_cpu: torch.Tensor) -> None:
        for p, off in zip(self.params, self.offsets):
            flat_cpu[off : off + p.numel()].copy_(
                p.detach().reshape(-1), non_blocking=True
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def add_params_from(self, flat_cpu: torch.Tensor) -> None:
        """θ += flat (one H2D then per-param add_; preserves memory format)."""
        with torch.no_grad():
            if self.device.type == "cuda":
                if self._gpu_stage is None:
                    self._gpu_stage = torch.empty(
                        self.numel, dtype=torch.float32, device=self.device
                    )
                self._gpu_stage.copy_(flat_cpu, non_blocking=True)
                src = self._gpu_stage
            else:
                src = flat_cpu
            for p, off in zip(self.params, self.offsets):
                p.add_(src[off : off + p.numel()].view(p.shape))
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

    def set_params_from(self, flat_cpu: torch.Tensor) -> None:
        with torch.no_grad():
            for p, off in zip(self.params, self.offsets):
                p.copy_(flat_cpu[off : off + p.numel()].view(p.shape))

    def bn_vec(self) -> torch.Tensor:
        out = torch.empty(self.bn_numel, dtype=torch.float32)
        for b, off in zip(self.bn_bufs, self.bn_offsets):
            out[off : off + b.numel()].copy_(b.detach().reshape(-1))
        return out

    def set_bn_from(self, vec: torch.Tensor) -> None:
        with torch.no_grad():
            for b, off in zip(self.bn_bufs, self.bn_offsets):
                b.copy_(vec[off : off + b.numel()].view(b.shape).to(b.device))

    def bn_var_mask(self) -> torch.Tensor:
        m = torch.zeros(self.bn_numel, dtype=torch.bool)
        for k, b, off in zip(self.bn_keys, self.bn_bufs, self.bn_offsets):
            if k.endswith("running_var"):
                m[off : off + b.numel()] = True
        return m


def merge_bn_pooled(
    bn_vecs: List[torch.Tensor],
    weights: List[float],
    mean_mask: torch.Tensor,
    var_mask: torch.Tensor,
) -> torch.Tensor:
    """Pooled-moment merge: m̄=Σwᵢmᵢ ; v̄=Σwᵢ(vᵢ+mᵢ²)−m̄², clamped ≥1e-10."""
    W = sum(weights)
    weights = [w / W for w in weights]
    mbar = torch.zeros(int(mean_mask.sum()))
    second = torch.zeros_like(mbar)
    for vec, w in zip(bn_vecs, weights):
        m = vec[mean_mask]
        v = vec[var_mask]
        mbar += w * m
        second += w * (v + m * m)
    vbar = (second - mbar * mbar).clamp_min_(1e-10)
    out = torch.empty_like(bn_vecs[0])
    out[mean_mask] = mbar
    out[var_mask] = vbar
    return out


def f32_from(buf: "bytes | bytearray") -> torch.Tensor:
    """Decode a float32 byte payload; empty-safe.

    ``torch.frombuffer`` raises on zero-length buffers, but a model with no
    BatchNorm layers (any pure transformer) legitimately produces an empty
    BN vector - the wire still carries the (empty) frame.
    """
    if not buf:
        return torch.empty(0, dtype=torch.float32)
    return torch.frombuffer(bytearray(buf), dtype=torch.float32)


def anchor_hash16(params_flat: torch.Tensor, bn_flat: torch.Tensor) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(params_flat.numpy().tobytes())
    h.update(bn_flat.numpy().tobytes())
    return h.digest()


def apply_round_to_anchor(
    anchor: torch.Tensor,
    payload: bytes,
    kind: str,
    lo: int = 0,
    hi: Optional[int] = None,
) -> torch.Tensor:
    """THE shared anchor mutation: identical bytes -> identical CPU fp32 ops on
    every node (leader included). Returns the dequantized update vector.
    [lo, hi) selects the shard slice for streaming partial sync; the default
    is the full vector."""
    hi = anchor.numel() if hi is None else hi
    upd = codec_decode(payload, hi - lo, kind)
    anchor[lo:hi].add_(upd)
    return upd


def shard_bounds(rnd: int, numel: int, shards: int) -> Tuple[int, int]:
    """Round r syncs shard (r-1) mod K of the flat param vector."""
    per = -(-numel // shards)
    s = (rnd - 1) % shards
    lo = s * per
    return lo, min(numel, lo + per)
