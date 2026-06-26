"""Multi-rank ShardedOptimizer test using torch.multiprocessing.spawn."""
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    from sakura.zero.sharded_optimizer import ShardedOptimizer

    torch.manual_seed(0)  # all ranks see the same initial weights
    model = torch.nn.Linear(8, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sharded = ShardedOptimizer(opt)

    # Set fake gradients (uniform; all ranks have the same).
    for p in model.parameters():
        p.grad = torch.ones_like(p)

    # Capture pre-step weights.
    before = [p.detach().clone() for p in model.parameters()]
    sharded.step()
    after = [p.detach().clone() for p in model.parameters()]

    # Weights moved.
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))

    # All ranks should have the SAME weights after step (broadcast).
    # Verify by gathering rank 0's weights and comparing.
    if rank == 0:
        # rank 0 holds even-indexed params (0, 2, ...). After broadcast, every
        # rank has every param.
        pass
    # Use all_reduce to verify everyone agrees.
    for p in model.parameters():
        snapshot = p.detach().clone()
        dist.all_reduce(p.detach(), op=dist.ReduceOp.SUM)
        # If all ranks have the same value, sum should be world_size * snapshot.
        _expected = snapshot * world_size
        assert torch.allclose(p.detach() / world_size, snapshot, atol=1e-5)
        # Restore for next iteration (no-op since this is end of test)
        p.data.copy_(snapshot)

    dist.barrier()
    dist.destroy_process_group()


def _find_free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_sharded_optimizer_2_ranks():
    """Spawn 2 processes, run ShardedOptimizer.step on a 4-param model."""
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")
    port = _find_free_port()
    mp.spawn(_worker, args=(2, port), nprocs=2, join=True)


def test_sharded_optimizer_world_size_1_passthrough():
    """Sanity: world_size=1 (no dist init) passes through to opt.step()."""
    from sakura.zero.sharded_optimizer import ShardedOptimizer

    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sharded = ShardedOptimizer(opt)

    for p in model.parameters():
        p.grad = torch.ones_like(p)
    before = [p.detach().clone() for p in model.parameters()]
    sharded.step()
    after = [p.detach().clone() for p in model.parameters()]
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))
