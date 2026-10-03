# Federation example — cross-region local-SGD

Runnable demo of `sakura.services.federation`: a CPU-only **leader** process
merges int8-compressed weight deltas from GPU **workers** that never stop
training. Workers train ResNet-50 on a synthetic batch, so no dataset is
needed. Each worker picks its own batch size and precision — heterogeneity
is the point. On the reference 2-region deployment (248 ms RTT WireGuard,
3 GPUs with 56:1 compute skew), this design retains 97.6% of the fleet's
physical throughput where synchronous DDP collapses to ~1 img/s.

## Requirements

- Linux host(s) with `torch` and `torchvision` installed (workers want a
  CUDA GPU; they fall back to CPU, which is slow but functional).
- The leader's TCP port reachable from every worker (WireGuard mesh, VPN,
  or loopback).

## Loopback smoke test (one machine, three terminals)

```bash
# terminal 1 — leader: 3 rounds, 10 s apart, then exit
python examples/federation/run_federation.py leader \
    --port 29500 --expected w0,w1 --codec int8 --gap 10 --rounds 3

# terminal 2 — worker w0
python examples/federation/run_federation.py worker \
    --leader-host 127.0.0.1 --port 29500 --id w0 --batch 8 --steps 300

# terminal 3 — worker w1
python examples/federation/run_federation.py worker \
    --leader-host 127.0.0.1 --port 29500 --id w1 --batch 8 --steps 300
```

On a CPU-only machine, drop `--batch` to 4 and expect ~1 s/step.

## Real cross-region run

Start the leader on a host reachable from both regions, then one worker per
GPU, each with its own batch/precision and a `--region` label:

```bash
# leader host (CPU-only; self-paced rounds at <=50% link occupancy)
python examples/federation/run_federation.py leader \
    --port 29500 --expected tokyo-a,paris-b --codec int8 --link-duty 0.5

# region A GPU
python examples/federation/run_federation.py worker \
    --leader-host <leader-host> --port 29500 --id tokyo-a \
    --batch 96 --amp bf16 --region tokyo

# region B GPU
python examples/federation/run_federation.py worker \
    --leader-host <leader-host> --port 29500 --id paris-b \
    --batch 64 --amp fp16 --region paris
```

## What to expect

Rounds start once every `--expected` worker has connected. Each committed
round prints one `LEADER_ROUND {...}` JSON line on the leader (participants,
per-worker sample weights, bytes up/down, `hash16`, `consensus_ok`) and one
`WORKER_ROUND {...}` line per worker (drift norm, upload/broadcast ms,
snapshot/apply stalls). `hash16` is the blake2b consensus hash of the
post-round anchor — it must be identical on the leader and every worker; a
mismatch is logged as `CONSENSUS HASH MISMATCH` and the worker is resynced.
On exit, the leader prints `LEADER_DONE` and workers print `WORKER_DONE`
with their aggregate img/s.

Full protocol and tuning notes: [docs/services/federation.md](../../docs/services/federation.md).
