"""TDD for e8: magnitude compression codecs (int4, bucket-local top-k) with
error feedback. Loads the REAL shipping federation modules as a synthetic
`fed` package so we do not need the Rust `sakura_wire` native built."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")

_BASE = pathlib.Path(__file__).resolve().parents[2] / "sakura/services/federation"


def _codec():
    if "fed.codec" in sys.modules:
        return sys.modules["fed.codec"]
    pkg = types.ModuleType("fed")
    pkg.__path__ = [str(_BASE)]
    sys.modules["fed"] = pkg
    spec = importlib.util.spec_from_file_location("fed.codec", _BASE / "codec.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["fed.codec"] = m
    spec.loader.exec_module(m)
    return m


def _heavy_delta(seed: int = 0):
    """Gaussian bulk + sparse heavy outliers — the realistic delta shape the
    int8 test in experiments/test_fed_local.py also uses."""
    c = _codec()
    n = c.BUCKET * 7 + 123
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g) * 1e-3
    idx = torch.randperm(n, generator=g)[:50]
    x[idx] *= 300.0
    return x


def test_int8_baseline_roundtrip_through_harness():
    """Sanity: the existing int8 codec still works through our synthetic loader."""
    c = _codec()
    x = _heavy_delta()
    payload, ef, contraction = c.codec_encode(x, None, "int8")
    deq = c.codec_decode(payload, x.numel(), "int8")
    assert torch.allclose(ef, x - deq, atol=1e-7)
    assert 0.0 < contraction <= 1.0
    ratio = (x.numel() * 4) / len(payload)
    assert 3.5 < ratio < 4.1, ratio  # int8 ≈ 4× vs fp32


def test_int4_roundtrip_ef_and_ratio():
    c = _codec()
    x = _heavy_delta()
    payload, ef, contraction = c.codec_encode(x, None, "int4")
    deq = c.codec_decode(payload, x.numel(), "int4")
    assert deq.numel() == x.numel()
    assert 0.0 < contraction <= 1.0
    # error feedback must equal the true residual (telescoping correctness)
    assert torch.allclose(ef, x - deq, atol=1e-6)
    # int4 packs 2 values/byte -> ~8× vs fp32
    ratio = (x.numel() * 4) / len(payload)
    assert ratio > 7.0, ratio


def _telescope_err(kind: str, K: int) -> float:
    """K rounds of a constant true delta d through EF; return the relative
    cumulative bias ‖Σ deqᵢ − K·d‖ / ‖K·d‖. The EF identity is exact:
    Σ deqᵢ = K·d − ef_K, so this error must decay ~1/K."""
    c = _codec()
    g = torch.Generator().manual_seed(1)
    d = torch.randn(c.BUCKET * 3, generator=g) * 1e-2
    d[::997] *= 100
    applied = torch.zeros_like(d)
    ef = None
    for _ in range(K):
        payload, ef, _ = c.codec_encode(d.clone(), ef, kind)
        applied += c.codec_decode(payload, d.numel(), kind)
    return float((applied - K * d).norm()) / float((K * d).norm())


def test_topk_keeps_largest_per_bucket():
    """Bucket-local top-k must preserve exactly the k largest-magnitude
    entries in each 4096-bucket (and zero the rest)."""
    c = _codec()
    g = torch.Generator().manual_seed(3)
    x = torch.zeros(c.BUCKET)
    perm = torch.randperm(c.BUCKET, generator=g)
    x[perm] = torch.arange(1, c.BUCKET + 1, dtype=torch.float32)  # distinct |x|
    density = 0.1
    k = -(-c.BUCKET * 10 // 100)  # ceil(0.1*BUCKET) = 410
    payload, ef, _ = c.codec_encode(x, None, f"topk:{density}")
    deq = c.codec_decode(payload, x.numel(), "topk")
    kept = set((deq != 0).nonzero().flatten().tolist())
    true_topk = set(torch.topk(x.abs(), k).indices.tolist())
    assert kept == true_topk
    assert torch.allclose(ef, x - deq, atol=1e-5)  # EF = full residual (DGC)


def test_topk_compression_ratio_scales_with_density():
    c = _codec()
    x = _heavy_delta()
    p10, _, _ = c.codec_encode(x, None, "topk:0.1")
    p01, _, _ = c.codec_encode(x, None, "topk:0.01")
    r10 = (x.numel() * 4) / len(p10)
    r01 = (x.numel() * 4) / len(p01)
    assert r10 > 10.0, r10  # ~13× at 10% density
    assert r01 > 40.0, r01  # ~100×+ at 1% density
    assert r01 > r10


def test_topk_ef_telescopes_and_decays_like_one_over_k():
    """DGC property: dropped-and-accumulated mass is eventually applied, so the
    cumulative update is unbiased and its error decays ~1/K."""
    err30 = _telescope_err("topk:0.1", 30)
    err120 = _telescope_err("topk:0.1", 120)
    assert err30 < 0.15, err30
    assert err120 < err30 * 0.6, (err30, err120)


def test_int4_ef_telescopes_and_decays_like_one_over_k():
    """EF makes int4 unbiased over rounds: cumulative bias is bounded and
    shrinks ~1/K (the residual ef_K is ~constant while K·d grows linearly).
    int4 is ~18× coarser than int8, so the constant is larger — what matters
    is the decay, which is the proof EF is carrying the residual correctly."""
    err30 = _telescope_err("int4", 30)
    err120 = _telescope_err("int4", 120)
    assert err30 < 0.10, err30
    assert err120 < err30 * 0.6, (err30, err120)  # ~1/K decay (4× longer ⇒ <0.6×)


# --- SparseLoCo (#88): int2 and topk2 ---------------------------------------


def test_int2_roundtrip_ef_and_ratio():
    c = _codec()
    x = _heavy_delta()
    payload, ef, contraction = c.codec_encode(x, None, "int2")
    deq = c.codec_decode(payload, x.numel(), "int2")
    assert deq.numel() == x.numel()
    assert 0.0 < contraction <= 1.0
    assert torch.allclose(ef, x - deq, atol=1e-6)  # EF = true residual
    # symmetric ternary levels {-1, 0, +1}·scale per bucket
    nb = -(-x.numel() // c.BUCKET)
    scale = torch.frombuffer(bytearray(payload[: nb * 4]), dtype=torch.float32)
    pad = (-x.numel()) % c.BUCKET
    levels = torch.nn.functional.pad(deq, (0, pad)).view(nb, c.BUCKET) / scale[:, None]
    assert set(levels.unique().tolist()) <= {-1.0, 0.0, 1.0}
    # int2 packs 4 values/byte -> ~16× vs fp32
    ratio = (x.numel() * 4) / len(payload)
    assert ratio > 14.0, ratio


def test_int2_crumb_packing_is_exact():
    """Every level, at every crumb position and across a partial last bucket,
    survives pack/unpack bit-exactly."""
    c = _codec()
    n = c.BUCKET + 5
    x = torch.tensor([-1.0, 0.0, 1.0, 0.0, 1.0, -1.0, -1.0] * (n // 7 + 1))[:n]
    payload, ef, _ = c.codec_encode(x, None, "int2")
    deq = c.codec_decode(payload, n, "int2")
    assert torch.equal(deq, x)
    assert float(ef.abs().max()) == 0.0


def test_int2_ef_telescopes_and_decays_like_one_over_k():
    """Ternary levels are coarse (measured err30≈0.26, err120≈0.085,
    err480≈0.023), so the constant is ~4× int4's; the 1/K decay is the proof
    EF carries the residual."""
    err30 = _telescope_err("int2", 30)
    err120 = _telescope_err("int2", 120)
    assert err30 < 0.35, err30
    assert err120 < err30 * 0.6, (err30, err120)


def test_topk2_keeps_largest_per_bucket_and_ef_is_full_residual():
    c = _codec()
    g = torch.Generator().manual_seed(3)
    x = torch.zeros(c.BUCKET * 2 + 17)
    perm = torch.randperm(x.numel(), generator=g)
    x[perm] = torch.arange(1, x.numel() + 1, dtype=torch.float32)
    x[::2] *= -1
    payload, ef, contraction = c.codec_encode(x, None, "topk2:0.02")
    deq = c.codec_decode(payload, x.numel(), "topk2:0.02")
    assert deq.numel() == x.numel()
    assert 0.0 < contraction <= 1.0
    assert torch.allclose(ef, x - deq, atol=1e-4)  # dropped + quant residual
    k = -(-c.BUCKET * 2 // 100)  # ceil(0.02*BUCKET) = 82
    xp = torch.nn.functional.pad(x, (0, (-x.numel()) % c.BUCKET)).view(-1, c.BUCKET)
    dp = torch.nn.functional.pad(deq, (0, (-x.numel()) % c.BUCKET)).view(-1, c.BUCKET)
    for xb, db in zip(xp, dp):
        top = set(torch.topk(xb.abs(), k).indices.tolist())
        kept = set((db != 0).nonzero().flatten().tolist())
        assert kept <= top  # only top-k positions may be transmitted
        assert kept  # the bucket max always survives 2-bit rounding
        # kept values are ±scale with the sign of the original entry
        nz = db != 0
        assert torch.equal(torch.sign(db[nz]), torch.sign(xb[nz]))
        assert torch.allclose(db[nz].abs(), xb.abs().max().expand(int(nz.sum())))


def test_topk2_bytes_per_round_at_2pct_at_least_4x_below_int8():
    """Acceptance (#88): topk2 at density 0.02 ships >=4x fewer bytes/round
    than int8 (it is ~20x)."""
    c = _codec()
    x = _heavy_delta()
    p8, _, _ = c.codec_encode(x, None, "int8")
    p2, _, _ = c.codec_encode(x, None, "topk2:0.02")
    assert len(p8) >= 4 * len(p2), (len(p8), len(p2))
    # and strictly smaller than the int8-valued topk at the same density
    pk, _, _ = c.codec_encode(x, None, "topk:0.02")
    assert len(p2) < len(pk)


def test_topk2_default_density_is_sparseloco_regime():
    c = _codec()
    x = _heavy_delta()
    p_default, _, _ = c.codec_encode(x, None, "topk2")
    p_explicit, _, _ = c.codec_encode(x, None, f"topk2:{c.TOPK2_DENSITY}")
    assert p_default == p_explicit
    assert 0.01 <= c.TOPK2_DENSITY <= 0.03


def test_topk2_ef_telescopes_and_decays_like_one_over_k():
    err30 = _telescope_err("topk2:0.02", 30)
    err120 = _telescope_err("topk2:0.02", 120)
    assert err30 < 0.35, err30  # measured ≈0.26
    assert err120 < err30 * 0.6, (err30, err120)


@pytest.mark.parametrize("kind", ["int2", "topk2", "topk2:0.03"])
def test_new_kinds_dispatch_and_are_lossy(kind):
    c = _codec()
    assert c.is_lossy(kind)
    x = _heavy_delta()
    payload, ef, _ = c.codec_encode(x, None, kind)
    assert torch.allclose(c.codec_decode(payload, x.numel(), kind), x - ef, atol=1e-5)


@pytest.mark.parametrize("kind", ["int3", "topk3:0.1", "int8:0.5", "topk2:0", "topk2:1.5"])
def test_unknown_or_malformed_kind_rejected(kind):
    c = _codec()
    with pytest.raises(ValueError):
        c.codec_encode(torch.zeros(8), None, kind)


def test_fp32raw_not_lossy():
    assert not _codec().is_lossy("fp32raw")
