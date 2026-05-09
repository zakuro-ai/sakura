"""ShardedOptimizer multi-rank correctness — mp.spawn(2) over gloo.

Verifies the ZeRO stage-1 sharding implementation:
  1. After step(), every rank holds identical parameters (broadcast worked).
  2. The result matches what a single-rank step on the same gradients would
     produce — the cyclic dealing + per-shard step + broadcast is equivalent
     to a non-sharded step.
  3. Multi-step training converges to within tolerance of the single-rank
     reference, exercising the persistence of optimizer state across steps.

These tests are CPU-only (gloo backend) and self-contained — no real
DDP, just `torch.distributed.init_process_group("gloo")` per spawned
worker. They'll skip cleanly if torch.distributed is unavailable.
"""
from __future__ import annotations

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
mp = pytest.importorskip("torch.multiprocessing")


WORLD_SIZE = 2


# Module-level model factory: must be picklable for mp.spawn.
def _make_model():
    """Tiny model with 5 parameters (4 + 1 + 2 + 1 = 8 tensors).

    The first dimension chosen so cyclic dealing actually splits ownership
    across both ranks (rank 0 owns even-indexed params, rank 1 owns odd).
    """
    torch.manual_seed(42)
    return torch.nn.Sequential(
        torch.nn.Linear(4, 3),  # weight + bias = 2 params (rank 0, rank 1)
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2),  # weight + bias = 2 params (rank 0, rank 1)
    )


def _seed_grads(model: torch.nn.Module, seed: int = 7) -> None:
    """Fill .grad on every parameter with deterministic values.

    Used by both the multi-rank workers and the single-rank reference so
    every step compares the same input gradients.
    """
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.grad = torch.empty_like(p).normal_(generator=g)


def _zero1_worker(rank: int, world_size: int, init_file: str,
                   result_dir: str, step_count: int):
    """Worker entry point — spawned per rank.

    Sets up gloo PG, builds the model identically on each rank, seeds
    grads identically, runs `step_count` ShardedOptimizer steps, writes
    final params to `result_dir/rank_{rank}.pt` for the parent test to
    collect. File-based IPC is more robust across spawn boundaries than
    mp.Queue (the queue's listener socket sometimes vanishes under pytest).
    """
    from sakura.zero.sharded_optimizer import ShardedOptimizer

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _make_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sharded = ShardedOptimizer(opt)

        for step in range(step_count):
            _seed_grads(model, seed=7 + step)
            sharded.step()

        params = [p.detach().cpu().clone() for p in model.parameters()]
        torch.save(params, os.path.join(result_dir, f"rank_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _single_rank_reference(step_count: int) -> list:
    """Run the same training schedule on a single rank with a vanilla SGD.

    The reference is what ShardedOptimizer's distributed step *should*
    produce — same params, same lr, same grads, no sharding. ZeRO1 with
    cyclic dealing is mathematically equivalent because each parameter is
    updated by exactly one rank using the same SGD math, then broadcast.
    """
    model = _make_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for step in range(step_count):
        _seed_grads(model, seed=7 + step)
        opt.step()
    return [p.detach().cpu().clone() for p in model.parameters()]


def _run_multi_rank(step_count: int) -> dict:
    """Spawn `WORLD_SIZE` workers, collect results from `result_dir/rank_*.pt`.

    Returns a dict {rank: [param tensors]} once all workers have finished.
    """
    with tempfile.TemporaryDirectory(prefix="sak-zero1-test-") as result_dir:
        with tempfile.NamedTemporaryFile(suffix=".init", delete=False) as f:
            init_file = f.name
        os.unlink(init_file)  # PG file-store wants the path NOT to exist initially.
        try:
            mp.spawn(
                _zero1_worker,
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


@pytest.mark.skipif(not dist.is_available(),
                     reason="torch.distributed unavailable")
class TestZeRO1MultiRankCorrectness:
    def test_one_step_agrees_across_ranks_and_matches_reference(self):
        """One step: (a) ranks agree (broadcast worked), (b) result == single-rank ref.

        Combined into a single test to amortize the fixed ~6s cost of
        spawning 2 worker processes + initializing the gloo PG.
        """
        results = _run_multi_rank(step_count=1)
        ref = _single_rank_reference(step_count=1)
        # (a) inter-rank consistency.
        for r0, r1 in zip(results[0], results[1]):
            assert torch.allclose(r0, r1, atol=0, rtol=0), (
                "ranks disagree on a parameter — the broadcast didn't run, "
                "or each rank computed independently and never synced"
            )
        # (b) reference equivalence — ZeRO1 with cyclic dealing must be
        # mathematically identical to a vanilla SGD step on the same grads.
        for r, refp in zip(results[0], ref):
            assert torch.allclose(r, refp, atol=1e-7), (
                "multi-rank result diverged from single-rank reference — "
                "ShardedOptimizer is not equivalent to opt.step() for SGD"
            )

    def test_multi_step_convergence_matches_reference(self):
        """5 steps of training converge to identical params on both ranks
        and match the single-rank reference within float tolerance."""
        n_steps = 5
        ref = _single_rank_reference(step_count=n_steps)
        results = _run_multi_rank(step_count=n_steps)
        # Inter-rank consistency.
        for r0, r1 in zip(results[0], results[1]):
            assert torch.allclose(r0, r1, atol=0, rtol=0)
        # Reference equivalence — slightly looser tolerance to cover any
        # numerics from broadcast roundtrip (none for SGD on CPU, but
        # protect future backends like NCCL where reductions can re-order).
        for r, refp in zip(results[0], ref):
            assert torch.allclose(r, refp, atol=1e-6), (
                "5-step ZeRO1 trajectory diverged from single-rank — "
                "either the per-shard step skipped a parameter or the "
                "broadcast picked the wrong owner"
            )

    def test_owner_assignment_partitions_params_across_ranks(self):
        """Sanity: with 4 params and world_size=2, rank 0 owns indices 0,2
        and rank 1 owns 1,3. Verifies the cyclic-dealing math directly."""
        from sakura.zero.sharded_optimizer import ShardedOptimizer

        # Build single-process; we don't need a real PG for this check.
        model = _make_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        # Construct without an active PG: world_size defaults to 1.
        sharded = ShardedOptimizer(opt)
        # Force the partition table by manually patching world_size — the
        # _partition_params logic doesn't depend on dist state at compute time.
        sharded._world_size = 2
        owners = sharded._partition_params()
        ranks = list(owners.values())
        # 4 params, alternating 0, 1, 0, 1 by cyclic dealing.
        assert ranks == [0, 1, 0, 1]
