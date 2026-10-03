# Federation — cross-region local-SGD

`sakura.services.federation` trains one model across GPUs in different
regions — sites connected by a WireGuard mesh or similar overlay, where
RTT is hundreds of milliseconds and bandwidth is a shared budget. Workers
train at full native speed on their own data; a CPU-only **leader** process
periodically gathers compressed weight deltas, merges them into a canonical
**anchor**, and broadcasts the consensus back. On our reference 2-region
deployment (248 ms RTT, ~11.5 MB/s), synchronous DDP over the same path
collapses to **1.03 img/s** on ResNet-50; the federation service retains
**2 363 img/s — 97.6% of the fleet's physical ceiling, a 2 294× speedup**.

**When to use it:** the fleet is geographically split *and* the workload is
compute-bound — large models or batches where no single GPU saturates the
problem. Do **not** use it inside a single region with fast interconnect:
plain DDP is simpler and correct there. And do not expect a win when one GPU
can crush the model on its own: in the E6 honesty check (CIFAR-10 /
ResNet-18), a solo RTX 4090 *beat* the federated 3-GPU fleet by ~2 pp at
equal wall-clock, because at 6 000+ img/s the solo GPU does ~58 epochs in
the 8-minute window and a 56:1-skewed fleet can't pay its staleness cost. Federation
pays off where the fleet is compute-bound (ResNet-50@224: +43% over the
solo 4090) and the gap grows with model and data scale.

## Quick start

One leader process (CPU-only, typically co-located with one region) plus one
worker process per GPU. The leader and all workers must construct the same
architecture and use the same codec.

**Leader process:**

```python
# leader_main.py — separate CPU-only process
from torchvision.models import resnet50
from sakura.services.federation import Leader

leader = Leader(
    model_fn=lambda: resnet50(),         # builds the CPU anchor
    port=29500,
    expected=["p620", "2080ti", "4090"], # every worker_id that may connect
    codec="int8",
    link_duty=0.5,                       # self-paced: round traffic <= 50% of the link
    persist_path="leader_state.pt",      # survive leader restarts
)
leader.serve_forever(warmup_s=5.0)       # blocks; rounds start when all workers are up
```

**Worker process** (one per GPU; each picks its own batch size and precision —
heterogeneity is the point). Wire the service through `SakuraRuntime` and an
adapter, exactly as in `tests/services/test_federation.py`:

```python
# worker_main.py — e.g. the 4090
import torch
from torchvision.models import resnet50
from sakura import SakuraRuntime
from sakura.adapters import DDPAdapter
from sakura.services.federation import FederationService

model = resnet50().cuda()
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

fed = FederationService(
    leader_host="leader.example.org", leader_port=29500,
    worker_id="4090",                 # must appear in the leader's `expected`
    batch_size=64,                    # samples per optimizer step (round weighting)
    region="site-a",
)

with SakuraRuntime() as rt:
    rt.install(fed)
    adapter = DDPAdapter(rt, rank=0, world_size=1)
    adapter.on_train_begin(model=model, optimizer=opt, train_loader=train_loader)
    for epoch in range(epochs):
        for step, (x, y) in enumerate(train_loader):
            adapter.on_train_step_begin(model=model, batch=(x, y), step=step)
            opt.zero_grad()
            loss = loss_fn(model(x.cuda()), y.cuda())
            loss.backward()
            adapter.on_optimizer_step(optimizer=opt)
            opt.step()
        adapter.on_epoch_end(epoch, model=model, optimizer=opt,
                             metrics={"loss": float(loss.item())})
```

The Lightning and HF adapters work the same way: install the service, attach
the adapter, train normally. All parameter mutation happens on the training
thread at step boundaries; the background thread only talks to the leader.

## How it works

**Anchor.** The leader owns a canonical fp32 copy of the parameters on CPU.
Each round, every worker uploads `(current params − anchor)` for its window;
the leader averages the deltas weighted by slow-EMA sample counts and adds
the result to the anchor. Workers apply the *same broadcast bytes* through
the same CPU function, so every node lands on bit-identical state.

**int8 + error feedback.** Deltas are compressed with bucketed symmetric
int8 (4 096-element buckets, one fp32 scale each — ~4× smaller than fp32).
The quantization residual is carried into the next round on both the worker
(upload) and leader (broadcast) sides, so compression error telescopes away
instead of accumulating. Measured contraction δ̂ ≈ 0.999.

**BatchNorm pooled moments.** BN `running_mean`/`running_var` can't be
delta-averaged (naive variance averaging can go negative). They travel as a
raw fp32 sideband and merge by pooled moments:
`m̄ = Σwᵢmᵢ`, `v̄ = Σwᵢ(vᵢ + mᵢ²) − m̄²`, clamped ≥ 1e-10.

**Consensus hash.** After applying a round, every node sends a blake2b-16
hash of its anchor + BN state. A mismatch is treated as corruption: the
worker is dropped and resynced from scratch. 44/44 WAN rounds green in the
benchmark campaign.

**Overlap.** In the default `mode="overlap"`, workers never stop training
during a round. The snapshot and apply are 9–46 ms training-thread pauses at
step boundaries; the upload/merge/broadcast happens in the background, at
the cost of one round of staleness. `mode="block"` pauses training until the
round commits (the fp32-blocking E3 rung; mostly useful for debugging).

**Streaming shards.** With `shards=K`, round *r* syncs only shard
`(r−1) mod K` of the flat parameter vector — K× less WAN traffic per round,
allowing K× faster cadence. E5s (K=4, T=8 s) cut consensus staleness p50
from ~9.6 s to 3.9 s fleet-wide (3.4 s for the leader-local workers)
for a 1.1 pp throughput cost.

**Self-pacing.** With `link_duty > 0` there is no configured round period:
the leader derives each gap from the *measured* gather+broadcast transfer
cost of the round just completed, so that round traffic occupies at most
that fraction of the inter-region link (gap clamped to [3 s, 120 s]; held
unchanged after a failed round). Nobody knows an arbitrary path's
RTT/bandwidth in advance — the framework measures instead of being
configured. `link_duty=0.5` is the multi-tenant good-citizen default choice.

## Configuration reference

### `FederationService` (worker side)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `leader_host` | `str` | required | Hostname of the leader process. |
| `leader_port` | `int` | required | Leader TCP port. |
| `worker_id` | `str` | required | Unique ID; must be in the leader's `expected` list or the handshake is rejected. |
| `batch_size` | `int` | required | Samples per optimizer step; accumulated into the round's sample count, which sets this worker's aggregation weight. |
| `mode` | `str` | `"overlap"` | `"overlap"`: keep training during rounds (one-round staleness). `"block"`: pause at the step boundary until the round commits. |
| `codec` | `str` | `"int8"` | `"int8"`: bucketed int8 + error feedback. `"fp32raw"`: lossless reference. Must match the leader. |
| `region` | `str` | `"default"` | First-class topology tag; reported in the handshake and in per-round records. |

### `Leader`

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `model_fn` | `Callable[[], torch.nn.Module]` | required | Factory building the same architecture as the workers; instantiated once on CPU to create the anchor. |
| `port` | `int` | required | TCP port to listen on (binds all interfaces). |
| `expected` | `List[str]` | required | Allowed `worker_id`s. Rounds begin once all are connected; afterwards, rounds proceed with whichever subset is alive. |
| `codec` | `str` | `"int8"` | Delta codec; must match the workers. |
| `gap_s` | `float` | `30.0` | Fixed gap between rounds. Ignored when `link_duty > 0`. |
| `persist_path` | `Optional[str]` | `None` | Atomic per-commit persistence of `{round, anchor, bn}`; a restarted leader resumes instead of re-initializing from seed. Strongly recommended in production. |
| `seed` | `int` | `1234` | Seeds torch before `model_fn()` so the initial anchor is deterministic. |
| `max_rounds` | `int` | `10**9` | Stop after this many committed rounds. |
| `shards` | `int` | `1` | Streaming shard count K; round r syncs shard `(r−1) mod K`. |
| `link_duty` | `float` | `0.0` | `> 0` enables self-pacing: next gap = `t_xfer × (1/duty − 1)`, clamped [3 s, 120 s]. `0` uses `gap_s`. |

`Leader.serve_forever(warmup_s=5.0)` waits for all expected workers, sleeps
`warmup_s`, then runs the round loop until `max_rounds` or `leader.stop` is set.

## Operational notes

- **Keep connections warm.** The service holds persistent striped TCP
  connections precisely because cold connections measured 6.5 MB/s vs
  11.3 MB/s warm on the reference path. Set
  `net.ipv4.tcp_slow_start_after_idle=0` on both ends so idle gaps between
  rounds don't reset the congestion window.
- **Never `setsockopt` SO_SNDBUF/SO_RCVBUF** on these connections in
  containers. With `rmem_max` capped (416 KB on our mesh) a manual buffer
  size kills kernel autotuning and throughput drops to 0.86 MB/s. The
  transport deliberately does not set these — don't add them.
- **WireGuard MTU must be 1280** on the mesh interface. With larger MTUs,
  bulk transfers blackhole (stalls of 30 s+) while small control traffic
  still flows, which looks exactly like a "leader hang".
- **Protocol version mismatch** (`PROTO_VER`, currently 2) fails at the
  frame header: a stale binary gets `WireDead: bad frame header` on its
  first frame and rejects at handshake instead of crash-looping every round.
  Upgrade leader and workers together.
- **Worker drop / rejoin.** Faults are confined per worker: a malformed or
  timed-out upload drops that worker and the round commits with the
  remaining subset. On reconnect the worker reports its last committed round
  and anchor hash; the leader picks one of three paths: nothing (already in
  sync), **retained resend** of the one missed round (only if the rejoiner's
  anchor hash matches the stored pre-round hash — otherwise replaying would
  double-apply), or **FULL_SYNC** (full fp32 anchor + BN, used for anything
  older and for every fresh worker at bootstrap). A FULL_SYNC discards the
  worker's uncommitted local progress — correct by protocol, and rare.
- **Leader restart.** With `persist_path` set, state is persisted atomically
  on every commit and a restarted leader resumes at its last committed
  round; workers reconnect (with backoff) and resync as above. Without it, a
  restarted leader re-initializes from `seed` and FULL_SYNCs a fresh random
  model over every trained worker — set `persist_path`.
- The leader sends heartbeats every 5 s (also mid-round), so workers detect
  a dead leader even during long transfers; receive paths skip them
  transparently.

## Measured performance

ResNet-50, 3×224×224, 3 GPUs (RTX 4090 bf16 b64 + RTX 2080 Ti fp16 b96 in
one region; Quadro P620 4 GB fp32 b16 in the other), 2 regions over a
WireGuard mesh via an LA relay — 248 ms RTT, ~11.5 MB/s warm striped TCP,
56:1 compute skew:

| Rung | Config | Aggregate img/s | % of ceiling | × baseline |
|---|---|---|---|---|
| baseline | sync DDP (gloo) over mesh | 1.03 | 0.04% | 1× |
| E3 | fp32 blocking, T=30 s | 1 619 | 66.8% | 1 572× |
| E4 | int8+EF blocking, T=30 s | 1 969 | 81.3% | 1 912× |
| **E5** | **int8 overlapped, T=30 s** | **2 363** | **97.6%** | **2 294×** |
| E5s | + streaming shards K=4, T=8 s | 2 337 | 96.5% | 2 269× |
| E5r2 | int8 overlapped, self-paced (`link_duty=0.5`) | 2 321 | 95.8% | 2 253× |

Ceiling = 2 422 img/s, the sum of each GPU's *concurrent* tuned solo
throughput. Per-worker retention in E5: P620 98.9%, 2080 Ti 95.6%,
4090 98.4%. Consensus hash green on every round of every run (44 WAN
rounds). Self-pacing settled at 5.0–7.7 s steady-state gaps (8.7 s first
measurement; mean 6.6 s) purely from measurement, within ~2 pp of the
hand-tuned T=30 run.

**The honest caveat:** these numbers measure *throughput retention*, not a
free lunch. The E6 convergence check (CIFAR-10, ResNet-18, 8 min wall-clock)
confirmed federation converges correctly — matching a solo-4090 control
early, within ~2 pp at 8 min, consensus green on all 21 rounds — but the
solo 4090 *won* that workload, as expected for a model one GPU crushes.
Reach for federation when your fleet is compute-bound; reach for one big GPU
(or single-region DDP) when it isn't.
