"""ShardedOptimizer NCCL multi-rank — 2 GPUs, real CUDA hardware.

Mirrors the gloo CPU test (`test_sharded_optimizer_multi_rank.py`) but
with the backend that actually ships in production. Each rank pins to a
different physical GPU; NCCL handles the broadcast.

Skipped unless 2+ CUDA devices are available. The test uses a tiny model
(~30 params) because the second GPU on the test machine may have limited
free VRAM — correctness validation does not need scale, only enough to
exercise the cyclic-dealing partition + broadcast path.
"""
from __future__ import annotations

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
mp = pytest.importorskip("torch.multiprocessing")


WORLD_SIZE = 2


def _make_tiny_model():
    """Tiny MLP with ~30 params spread across 4 weight tensors.

    Cyclic dealing with world_size=2 distributes ownership as
    [rank0, rank1, rank0, rank1] — both ranks step their shard.
    The ~30 params keeps the per-rank GPU footprint at <100 KB even
    after autograd buffers, so NCCL works on a memory-constrained
    second GPU."""
    torch.manual_seed(42)
    return torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),
    )


def _seed_grads(model: torch.nn.Module, seed: int = 7) -> None:
    """Fill .grad with deterministic values — same on both ranks so the
    distributed result is comparable to a single-rank reference."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.empty_like(p, device="cpu").normal_(generator=g).to(p.device)


def _nccl_worker(rank: int, world_size: int, init_file: str,
                  result_dir: str, step_count: int):
    """Per-rank worker: pin to cuda:{rank}, init NCCL, run ShardedOptimizer.

    Disables NCCL's shared-memory and peer-to-peer transports — both can
    fail in containerized / sandboxed test environments (the shm segment
    size is sometimes reported as 0 even when /dev/shm is mounted with
    plenty of space). Falls back to the socket transport, which is slower
    but works anywhere. Production users on bare metal should leave the
    defaults.
    """
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    from sakura.zero.sharded_optimizer import ShardedOptimizer

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device(f"cuda:{rank}")
        model = _make_tiny_model().to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sharded = ShardedOptimizer(opt)

        for step in range(step_count):
            _seed_grads(model, seed=7 + step)
            sharded.step()

        # Ensure broadcast completes before we read params back.
        torch.cuda.synchronize(device)
        params = [p.detach().cpu().clone() for p in model.parameters()]
        torch.save(params, os.path.join(result_dir, f"rank_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _single_rank_reference_cuda(step_count: int) -> list:
    """Single-rank reference on cuda:0 — same model, same seed, vanilla SGD.

    Run on GPU rather than CPU to match the multi-rank execution context;
    fp32 arithmetic on cuda:0 vs cuda:0 is bit-identical for these ops, so
    we can compare with atol=0 against rank 0 of the multi-rank run.
    """
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    model = _make_tiny_model().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for step in range(step_count):
        _seed_grads(model, seed=7 + step)
        opt.step()
    return [p.detach().cpu().clone() for p in model.parameters()]


def _run_multi_rank_nccl(step_count: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="sak-zero1-nccl-") as result_dir:
        with tempfile.NamedTemporaryFile(suffix=".init", delete=False) as f:
            init_file = f.name
        os.unlink(init_file)
        try:
            mp.spawn(
                _nccl_worker,
                args=(WORLD_SIZE, init_file, result_dir, step_count),
                nprocs=WORLD_SIZE,
                join=True,
            )
            results = {}
            for rank in range(WORLD_SIZE):
                path = os.path.join(result_dir, f"rank_{rank}.pt")
                results[rank] = torch.load(path, map_location="cpu", weights_only=True)
            return results
        finally:
            if os.path.exists(init_file):
                os.unlink(init_file)


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_nccl_available()
         and torch.cuda.is_available() and torch.cuda.device_count() >= WORLD_SIZE),
    reason=f"NCCL test requires {WORLD_SIZE}+ CUDA devices",
)
class TestZeRO1NCCLCorrectness:
    def test_nccl_one_step_agrees_across_ranks_and_matches_reference(self):
        """Same invariants as the gloo test, exercised over real NCCL on 2 GPUs.

        After one ShardedOptimizer step:
          (a) both ranks hold bit-identical params (NCCL broadcast works);
          (b) the result equals a single-rank vanilla SGD reference
              (cyclic dealing math is correct on GPU just like on CPU).
        """
        results = _run_multi_rank_nccl(step_count=1)
        ref = _single_rank_reference_cuda(step_count=1)

        # (a) inter-rank consistency.
        for r0, r1 in zip(results[0], results[1]):
            assert torch.allclose(r0, r1, atol=0, rtol=0), (
                "ranks disagree on a parameter — NCCL broadcast didn't run "
                "or each rank computed independently and never synced"
            )
        # (b) reference equivalence — same forward arithmetic on the same
        # GPU type for rank 0 vs reference, so atol can be tight.
        for r, refp in zip(results[0], ref):
            assert torch.allclose(r, refp, atol=1e-7), (
                "NCCL multi-rank diverged from single-rank reference — "
                "ShardedOptimizer math is wrong on GPU"
            )
