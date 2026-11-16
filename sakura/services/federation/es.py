"""Low-rank (EGGROLL) evolution-strategies primitives for federation.

EGGROLL (arXiv 2511.16652) structures each ES perturbation of a weight matrix
``W`` (``m x n``) as a rank-``r`` product ``E = A Bᵀ / sqrt(r)`` with ``A``
(``m x r``) and ``B`` (``n x r``) i.i.d. standard normal drawn from a seed.
Every entry of ``E`` still has unit variance, but generating it costs
``(m + n) r`` random numbers instead of ``m n`` and the perturbed forward
``x (W + σE)ᵀ`` can be computed as ``x Wᵀ + σ (x B) Aᵀ / sqrt(r)``.

For sakura the property that matters is the wire: a population member is
fully described by ``(seed, fitness)``. Workers evaluate perturbed copies of
the shared anchor locally and ship only seeds and fitness scalars; any node
holding the anchor regenerates the perturbations from the seeds and rebuilds
the same ES gradient bit-for-bit (same torch build, CPU generator).

Conventions used here:

  * Antithetic pairs: member ``2k`` is ``+E_k`` and member ``2k+1`` is
    ``-E_k``; a pair shares one seed, so a pair costs 8 B (seed) + 2 x 4 B
    (fitness) = 16 B on the wire.
  * 1-D tensors (biases, norm affine params) get a dense Gaussian vector from
    the same generator — they are tiny and have no low-rank structure.
    Tensors with more than two dims are perturbed as ``shape[0] x rest``.
  * Fitness is shaped with centered ranks in ``[-0.5, 0.5]`` across the whole
    population before the gradient estimate (standard OpenAI-ES / EGGROLL
    practice; makes the step invariant to the fitness scale).

Pure torch + struct; no dependency on the Rust wire module, so it loads in the
CPU simulator and in tests standalone.
"""
from __future__ import annotations

import math
import struct
from typing import List, Sequence, Tuple

import torch

Shape = Tuple[int, ...]

_MASK64 = (1 << 64) - 1
# Per-message header: round (u32), worker (u16), pair count (u16).
_HDR = struct.Struct("<IHH")
# Per-pair record: seed (u64), fitness of +E (f32), fitness of -E (f32).
_PAIR = struct.Struct("<Qff")
PAIR_BYTES = _PAIR.size  # 16


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def member_seed(base_seed: int, rnd: int, worker: int, pair: int) -> int:
    """Deterministic 63-bit seed for antithetic pair ``pair`` of ``worker`` in
    round ``rnd``. Distinct (rnd, worker, pair) give independent streams; the
    top bit is cleared so it fits ``torch.Generator.manual_seed``."""
    s = _splitmix64(base_seed & _MASK64)
    for v in (rnd, worker, pair):
        s = _splitmix64(s ^ (v & _MASK64))
    return s & ((1 << 63) - 1)


def lowrank_perturbation(seed: int, shapes: Sequence[Shape], rank: int) -> torch.Tensor:
    """Flat fp32 perturbation for a list of tensor shapes from one seed.

    Each matrix-shaped entry is ``A Bᵀ / sqrt(r)`` (rank ``min(r, m, n)``),
    each vector a dense standard normal. Concatenated in ``shapes`` order,
    matching ``FlatState`` param order when ``shapes`` comes from it."""
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    g = torch.Generator().manual_seed(int(seed))
    parts: List[torch.Tensor] = []
    for shape in shapes:
        numel = math.prod(shape)
        if len(shape) < 2:
            parts.append(torch.randn(numel, generator=g))
            continue
        m = shape[0]
        n = numel // m
        a = torch.randn(m, rank, generator=g)
        b = torch.randn(n, rank, generator=g)
        parts.append((a @ b.T).reshape(-1) / math.sqrt(rank))
    if not parts:
        return torch.empty(0)
    return torch.cat(parts)


def antithetic_perturbation(
    seed: int, sign: int, shapes: Sequence[Shape], rank: int
) -> torch.Tensor:
    """``sign * E(seed)`` with ``sign`` in ``{+1, -1}``."""
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    e = lowrank_perturbation(seed, shapes, rank)
    return e if sign == 1 else e.neg_()


def centered_ranks(f: torch.Tensor) -> torch.Tensor:
    """Map fitness values to centered ranks in ``[-0.5, 0.5]`` (ties broken by
    order, which is deterministic for a deterministic input)."""
    n = f.numel()
    if n < 2:
        return torch.zeros_like(f, dtype=torch.float32)
    ranks = torch.empty(n, dtype=torch.float32)
    ranks[torch.argsort(f.reshape(-1), stable=True)] = torch.arange(n, dtype=torch.float32)
    return (ranks / (n - 1) - 0.5).reshape(f.shape)


def es_gradient(
    pairs: Sequence[Tuple[int, float, float]],
    shapes: Sequence[Shape],
    rank: int,
    sigma: float,
    shape_fitness: bool = True,
) -> torch.Tensor:
    """Reconstruct the ES ascent direction from ``(seed, f_plus, f_minus)``.

    ``g = 1 / (2 P σ) · Σ_k (f⁺_k − f⁻_k) E_k`` over ``P`` antithetic pairs,
    with fitness optionally replaced by population-wide centered ranks.
    Deterministic in its inputs: every node that sees the same seeds and
    fitness values computes the same tensor."""
    if not pairs:
        raise ValueError("es_gradient needs at least one pair")
    f = torch.tensor([[fp, fm] for _, fp, fm in pairs], dtype=torch.float32)
    if shape_fitness:
        f = centered_ranks(f)
    numel = sum(math.prod(s) for s in shapes)
    g = torch.zeros(numel)
    for (seed, _, _), (fp, fm) in zip(pairs, f.tolist()):
        w = fp - fm
        if w != 0.0:
            g.add_(lowrank_perturbation(seed, shapes, rank), alpha=w)
    return g.div_(2.0 * len(pairs) * sigma)


def pack_es_payload(rnd: int, worker: int, pairs: Sequence[Tuple[int, float, float]]) -> bytes:
    """Wire form of one worker's population: header + 16 B per pair."""
    out = bytearray(_HDR.pack(rnd, worker, len(pairs)))
    for seed, fp, fm in pairs:
        out += _PAIR.pack(seed, fp, fm)
    return bytes(out)


def unpack_es_payload(buf: bytes) -> Tuple[int, int, List[Tuple[int, float, float]]]:
    rnd, worker, n = _HDR.unpack_from(buf, 0)
    if len(buf) != _HDR.size + n * _PAIR.size:
        raise ValueError(f"ES payload length {len(buf)} != header count {n}")
    pairs = [_PAIR.unpack_from(buf, _HDR.size + i * _PAIR.size) for i in range(n)]
    return rnd, worker, [(int(s), float(a), float(b)) for s, a, b in pairs]
