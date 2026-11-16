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
from ~9.6 s to 3.9 s fleet-wide (3.4 s for the leader-local the GPU host workers)
for a 1.1 pp throughput cost.

**Streaming DiLoCo (int4 shard + bounded overlap).** Shards combine with
`codec="int4"` (bucketed nibble-packed int4, error feedback on both the
worker and leader side) and, in overlap mode, an optional staleness bound
`max_stale_steps=τ`: once τ optimizer steps have run since a round's
snapshot, the trainer waits at the step boundary until the round commits
(`stale_wait_ms` in the round record). `None` (default) keeps the E5
behaviour of one round of staleness. Keep τ below the steps per round: at
τ = H/K the simulator lost 6–9 pp (likely because the apply's BN hard-set
lands right before the next snapshot and stalls the BN stats), while
τ < H/K showed no loss.

Measured with `python sim_fed.py --streaming` (CPU, shipping codec /
leader / apply primitives, 4 workers on 1:2:4:8 data shards, H=40 inner
steps per shard sync, 800 steps, 5 seeds; target = 0.95 × solo accuracy;
`results/streaming_sim/results.json`):

| Config | Final acc | Steps to target | Peak bytes/round/worker | Peak reduction |
|---|---|---|---|---|
| block, int8 | 0.7269 ± 0.002 | 64 ± 20 | 61 496 | 1.00× |
| stream K=4, int8, τ=5 | 0.7246 ± 0.004 | 80 ± 0 | 20 496 | 3.00× |
| stream K=4, int4, τ=0 | 0.7311 ± 0.005 | 80 ± 0 | 12 304 | 5.00× |
| **stream K=4, int4, τ=5** | **0.7263 ± 0.004** | **80 ± 0** | **12 304** | **5.00×** |

Accuracy is evaluated every 40 steps, so steps-to-target differs from block
mode by one evaluation interval (within the seed spread). The toy model is
26 k parameters, where the whole-model BN sideband is a large share of
each round; at ResNet-50 size (25.5 M) the per-worker round payload is
51.05 MB (block int8) vs 6.39 MB (stream int4, K=4): **7.99×** lower peak.
The loopback QUIC bench (`cargo bench -p sakura-wire --bench
transport_bench -- federation_round_payload`, ResNet-18-sized payload
echoed up and down, the GPU host) measured 115.1 ms per block-int8 round vs
14.9 ms per streamed int4 shard (7.7×) at the same ~190 MiB/s, so the
bandwidth a round needs at a fixed cadence drops by the payload ratio.

**Self-pacing.** With `link_duty > 0` there is no configured round period:
the leader derives each gap from the *measured* gather+broadcast transfer
cost of the round just completed, so that round traffic occupies at most
that fraction of the inter-region link (gap clamped to [3 s, 120 s]; held
unchanged after a failed round). Nobody knows an arbitrary path's
RTT/bandwidth in advance — the framework measures instead of being
configured. `link_duty=0.5` is the multi-tenant good-citizen default choice.

**Bootstrap link probe (SAK-0013).** With `codec="auto"` and/or
`shards=0` the leader picks the codec and K itself. It times the bootstrap
FULL_SYNC of every worker (plus the kernel RTT on Linux), logs each link
(`link probe: {...}` and a `LEADER_LINK_PLAN` stdout line: Mbit/s, RTT, BDP),
and walks (codec, K) rungs, `int8 -> int4 -> topk` x K in {1, 2, 4, 8, 16},
largest payload first. The first rung whose estimated round transfer
(`2 x RTT + (upload + broadcast bytes) / goodput`) fits `gap_s x link_duty`
(duty 0.5 when unset) on the **slowest** link wins. The broadcast uses that
one codec for the whole fleet (one byte string, SAK-0002); each worker
uploads with the least aggressive codec its own link can carry. Explicit
settings always win: a fixed codec only lets K move, a fixed K only lets the
codec move. The plan is chosen at bootstrap and again when a worker joins
(or rejoins with a FULL_SYNC) or is retired; a new plan takes effect at the
next round (announced in `ROUND_BEGIN`), and every change is logged with its
`reason` and `from_round` (`Leader.plan_history`). With `persist_path` the
plan is saved with the anchor, so a restarted leader whose workers are all in
sync resumes it instead of falling back to int8 / K=1 (the fallback when
nothing was ever measured). `Leader.link_plan.link_estimate()` returns the
largest-BDP link as a `LinkEstimate`; `.flow_control()` turns it into a
`sakura.wire.FlowControl` with BDP-scaled QUIC windows (SAK-0010).

## Hierarchical regions (SAK-0011)

With more than one worker per region, put a `RegionAggregator` in each
region between its workers and the root leader. The aggregator is a leader
for its region's workers (fast intra-region rounds on cheap links) and a
worker toward the root leader (inter-region rounds at the root's, usually
self-paced, cadence). Only one upload and one broadcast per region cross the
WAN per root round, instead of one per worker. Workers need no change: they
point `leader_host`/`leader_port` at their region's aggregator.

```python
# root: aggregator ids are its "workers"
root = Leader(model_fn=lambda: resnet50(), port=29500,
              expected=["agg-eu", "agg-us"], link_duty=0.5,
              persist_path="root_state.pt")
root.serve_forever()

# one per region, CPU-only, co-located with the region's workers
from sakura.services.federation import RegionAggregator
agg = RegionAggregator(model_fn=lambda: resnet50(), port=29501,
                       expected=["4090", "2080ti"], region="eu",
                       root_host="root.mesh", root_port=29500,
                       aggregator_id="agg-eu", gap_s=5.0)
agg.serve_forever()   # bootstraps from the root, then serves the region
```

The consensus hash is checked at both levels on every round. Workers ack the
regional anchor to their aggregator. Aggregators ack their copy of the root
anchor to the root. Regional round records carry `global_round` and
`global_hash16`. The root weighs regions by the samples they committed since
the last root round. Without `persist_path` a restarted aggregator FULL_SYNCs
from the root and its workers FULL_SYNC from it. With `persist_path` it
resumes from its file: one atomic snapshot of both tiers (the regional
anchor, round and optimizer buffers, plus its copy of the root anchor, the
root round, the pending correction, the upload error feedback and the samples
not yet reported), rewritten after every regional commit, root commit and
root resync. It then presents its root round to the root (nothing, resend or
FULL_SYNC, as for any worker) and its workers resume against the restored
regional anchor without a FULL_SYNC. Streaming shards (`shards=2` or more)
work at either level. `round_wire_bytes(rec)` gives a round's payload
bytes between a leader and its participants. On the root these are the WAN
bytes. See SAK-0011 for how the two anchors are reconciled.

Membership works at both levels (SAK-0014). `RegionAggregator(allow_join=...,
join_token=...)` admits regional workers outside its `expected`, exactly like
`Leader`. `root_join_token=` is a separate secret the aggregator presents in
its HELLO to the root, so an aggregator need not be listed in the root's
`expected` when the root has the same `join_token`. All three default to
`None`: unknown ids are rejected at both levels.

```python
root = Leader(..., expected=["agg-eu"], join_token=ROOT_TOKEN)
agg_us = RegionAggregator(..., expected=["c"], region="us",
                          aggregator_id="agg-us", join_token=REGION_TOKEN,
                          root_join_token=ROOT_TOKEN)
# worker "d" joins region us: FederationService(..., join_token=REGION_TOKEN)
```

Workers can fail over to another region's aggregator. With
`FederationService(fallback_hosts=[(host, port), ...], failover_after=3)` a
worker that fails `failover_after` connection attempts in a row moves to the
next endpoint of `[leader, *fallback_hosts]` (cyclic). The fallback aggregator
must admit it (`allow_join=` or `join_token=`), and FULL_SYNCs it because the
worker presents `last_committed=-1` to any endpoint whose anchor it does not
hold. From then on its samples count for the new region, at the regional and
at the root level; its uncommitted progress at the old region is dropped by
the FULL_SYNC, as for any rejoin. A worker does not fail back while its
current aggregator works. A restarted aggregator still waits for its whole
`expected` before its first regional round (as `Leader` does), so `retire()`
the workers that moved, or list only the ones that stay.

```python
fed = FederationService("agg-us.mesh", 29502, "d", 64, region="us",
                        join_token=REGION_TOKEN,          # eu admits it
                        fallback_hosts=[("agg-eu.mesh", 29501)])
```

## Configuration reference

### `FederationService` (worker side)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `leader_host` | `str` | required | Hostname of the leader process. |
| `leader_port` | `int` | required | Leader TCP port. |
| `worker_id` | `str` | required | Unique ID; must be in the leader's `expected` list (or admitted by its `allow_join` / `join_token`) or the handshake is rejected. |
| `batch_size` | `int` | required | Samples per optimizer step; accumulated into the round's sample count, which sets this worker's aggregation weight. |
| `mode` | `str` | `"overlap"` | `"overlap"`: keep training during rounds (one-round staleness). `"block"`: pause at the step boundary until the round commits. |
| `codec` | `str` | `"int8"` | `"int8"`: bucketed int8 + error feedback. `"int4"` / `"int2"`: same, nibble- / crumb-packed (about 8x / 16x vs fp32; int4 is the streaming-DiLoCo choice; int2 uses symmetric ternary levels). `"topk[:density]"`: bucket-local top-k + int8 values (default density 0.1). `"topk2[:density]"`: SparseLoCo, bucket-local top-k + int2 values (default density 0.02, about 20x smaller than int8). `"fp32raw"`: lossless reference. All lossy codecs carry error feedback; an unknown kind raises `ValueError`. Initial upload codec only: the worker follows the codecs the leader announces in every round. |
| `region` | `str` | `"default"` | First-class topology tag; reported in the handshake and in per-round records. |
| `max_stale_steps` | `Optional[int]` | `None` | Overlap mode only: at most this many optimizer steps between a round's snapshot and its apply; the trainer then waits for the round to commit. `None` = unbounded (one round of staleness). |
| `shard_size` | `Optional[int]` | `None` | Samples this worker sweeps per epoch, reported to the leader for the [scope advisory](#scope-advisory). `None` reads `len(train_loader.sampler)` at `on_train_begin`; unknown if that fails. |
| `join_token` | `Optional[str]` | `None` | Shared secret sent in the HELLO so a leader configured with the same `join_token` admits this worker even though it is not in `expected` (SAK-0014). |
| `rejoin_momentum` | `str` | `"noop"` | What a FULL_SYNC does to the inner optimizer's first-moment buffers (`momentum_buffer`, `exp_avg`): `"noop"` keeps them, `"reset"` zeroes them, `"average"` multiplies them by `rejoin_momentum_decay`. Second moments and step counts are never touched. Retained resends and normal rounds leave the optimizer alone. An unknown value raises `ValueError`. Measured in SAK-0014: no policy beat `noop` beyond seed noise. |
| `rejoin_momentum_decay` | `float` | `0.5` | Factor for `"average"`, in [0, 1]: 0.5 is the mean of the local buffer and a fresh (zero) one. |
| `reconnect_backoff_max_s` | `float` | `30.0` | Ceiling of the retry interval when the first connection or a reconnect to the leader fails (exponential from 2 s / 1 s). Lower it on a LAN, or where a restarted leader is back within seconds, so workers find it sooner. The wait is cut short by `on_train_end`. |
| `fallback_hosts` | `Optional[Sequence[Tuple[str, int]]]` | `None` | Other leaders or region aggregators to try, in order, when the current one keeps failing (see [Hierarchical regions](#hierarchical-regions-sak-0011)). Each must admit this `worker_id`. `None`: always the same leader. |
| `failover_after` | `int` | `3` | Consecutive failed connection attempts before moving to the next endpoint. Must be >= 1. |

### `Leader`

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `model_fn` | `Callable[[], torch.nn.Module]` | required | Factory building the same architecture as the workers; instantiated once on CPU to create the anchor. |
| `port` | `int` | required | TCP port to listen on (binds all interfaces). |
| `expected` | `List[str]` | required | Allowed `worker_id`s. Rounds begin once all are connected; afterwards, rounds proceed with whichever subset is alive. |
| `codec` | `str` | `"int8"` | Delta codec (see the worker table), announced to the workers each round. `"auto"`: chosen at bootstrap by the link probe. |
| `gap_s` | `float` | `30.0` | Fixed gap between rounds. Ignored when `link_duty > 0`. |
| `persist_path` | `Optional[str]` | `None` | Atomic per-commit persistence of `{round, anchor, bn}` plus the outer-optimizer state (EF residual, momentum, delayed-Nesterov buffer, per-shard step counts) and the link plan; a restarted leader resumes instead of re-initializing from seed. Strongly recommended in production. |
| `seed` | `int` | `1234` | Seeds torch before `model_fn()` so the initial anchor is deterministic. |
| `max_rounds` | `int` | `10**9` | Stop after this many committed rounds. |
| `shards` | `int` | `1` | Streaming shard count K; round r syncs shard `(r−1) mod K`. `0`: chosen at bootstrap by the link probe. |
| `link_duty` | `float` | `0.0` | `> 0` enables self-pacing: next gap = `t_xfer × (1/duty − 1)`, clamped [3 s, 120 s]. `0` uses `gap_s`. |
| `outer_opt` | `str` | `"plain"` | Outer step on the aggregated delta. `"plain"`: FedAvg (`lr·dbar`). `"sgdm"` / `"slowmo"` / `"nesterov"`: persistent outer-momentum buffer (DiLoCo). `"delayed_nesterov"`: buffered Nesterov, momentum refreshed every `outer_delay` updates (HALoS, #90; see *Outer optimizer and async aggregation* below). `"ef"`: SparseLoCo, no momentum buffer; the broadcast error-feedback residual is decayed by `outer_momentum` and acts as the momentum (pair with `topk2`; with `fp32raw` it is `plain`). An unknown kind raises `ValueError`. |
| `outer_lr` | `float` | `1.0` | Outer learning rate. |
| `outer_momentum` | `float` | `0.9` | μ for the momentum kinds (`SAKURA_MU_WARMUP=N` ramps it over N rounds); the EF decay β for `"ef"` (SparseLoCo uses 0.95). |
| `dataset_size` | `Optional[int]` | `None` | Training samples across all shards, for the [scope advisory](#scope-advisory). `None` sums the workers' `shard_size`. |
| `allow_join` | set of `str` or `Callable[[str], bool]` | `None` | Worker ids outside `expected` that may join while the leader runs (an explicit set, or a predicate; a predicate that raises denies). `None`: unknown ids are rejected. |
| `join_token` | `Optional[str]` | `None` | Shared secret; a worker presenting it (`FederationService(join_token=...)`) may join although its id is not in `expected`. `None`: no token admission. |
| `link_probe` | `Optional[Callable[[], Dict[str, LinkSample]]]` | `None` | Replaces the passive FULL_SYNC measurement (tests, or an external prober). Samples of unknown workers and observers are ignored. |
| `outer_delay` | `int` | `1` | `delayed_nesterov` only: N, the number of global updates buffered per momentum refresh (per streaming shard). N = 1 is plain `nesterov`. |

### `RegionAggregator`

Takes the `Leader` parameters above except `dataset_size`, `link_probe` and
`outer_delay`, plus:

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `region` | `str` | required | This region's tag, sent to the root in the HELLO. |
| `root_host` / `root_port` | `str` / `int` | required | The root leader. |
| `aggregator_id` | `Optional[str]` | `None` | Its worker id at the root (default `region-<region>`). Must be in the root's `expected` unless `root_join_token` is set. |
| `allow_join` / `join_token` | as `Leader` | `None` | Admission of regional workers outside `expected` (SAK-0014). |
| `root_join_token` | `Optional[str]` | `None` | Shared secret sent in the HELLO to the root; a root with the same `join_token` admits the aggregator although its id is not in the root's `expected`. Independent of the regional `join_token`. |
| `persist_path` | `Optional[str]` | `None` | Two-tier state file (regional `Leader` keys plus a `region` entry with the root-side state), rewritten atomically after every regional commit, root commit and root resync. A restart resumes from it; an unreadable or mismatched file is ignored (bootstrap from the root). `None`: bootstrap from the root on every start. |

`Leader.serve_forever(warmup_s=5.0)` waits for all expected workers, sleeps
`warmup_s`, then runs the round loop until `max_rounds` or `leader.stop` is set.

`Leader.admit(worker_id)` authorises another id at runtime;
`Leader.retire(worker_id)` drops a worker between rounds, removes it from the
roster and revokes its id-based authorisation.

### Outer optimizer and async aggregation (#90, HALoS slice)

`outer_opt.py` applies the outer step to the aggregated pseudo-gradient
`dbar`. `delayed_nesterov` is the buffered (Delayed) Nesterov step from HALoS
(arXiv 2506.04531) / Liu et al. 2024: every global update adds `dbar` to a
buffer `b`; on every N-th update the momentum is refreshed with the buffer
mean and the Nesterov look-ahead is applied, otherwise the update is a plain
`η_out·dbar` with momentum frozen:

```
b += dbar ; t += 1
t % N == 0:  m = μ·m + b/N ; b = 0 ; update = η_out·(dbar + μ·m)
otherwise:                           update = η_out·dbar
```

Staleness-weighted async aggregation: an async aggregator commits each upload
when it arrives as one global update. An upload computed from an anchor `τ`
updates old contributes `dbar = share_w · (1 + τ)^(−a) · Δ_w`, where
`share_w = ema_n_w / Σ ema_n` (so one fresh upload per worker sums to the sync
weighted mean). The discount is not renormalised: a stale upload moves the
anchor less, it does not take weight from the others. `a = 0` disables it;
`a = 0.5` is the default (`outer_opt.staleness_weight`,
`outer_opt.staleness_weighted_delta`).

The flat `Leader` is synchronous: every upload in a round shares the same
staleness, so the weight cancels and the leader uses only
`outer_opt="delayed_nesterov"` + `outer_delay`. The staleness functions are for
async aggregation, exercised today by `sim_fed.py --async-dn` (a 2-region
topology with a fast in-region LAN and a slow cross-region link). The
synchronous regional tier (#99, `RegionAggregator`) is unchanged; its root is a
plain `Leader`, so `outer_delay` applies there. Async regional aggregation
is simulated by `sim_fed.py --halos` (#91, below). Choose N ≈ the number of
workers, so one momentum step covers about one sync round of uploads.

Measured (`python sim_fed.py --async-dn`, CPU, synthetic 8-class task, int8,
4 workers on 2 regions: A = leader + 2 workers on a 1 GB/s LAN, B = 2 workers
on a 2 MB/s / 40 ms WAN link, worker speeds 100/60/100/40 steps/s; H = 10,
60 × 4 global updates, mean ± std of 3 seeds, best outer lr per arm; raw rows
in `results/e90_async_dn/results.json`):

| arm | η_out | final acc | sim time | mean / max staleness |
|---|---|---|---|---|
| sync `nesterov` (block rounds) | 0.3 | 0.7178 ± 0.008 | 23.83 s | 0 / 0 |
| async `plain`, a = 0.5 | 1.0 | 0.7186 ± 0.003 | 10.74 s | 2.98 / 9 |
| async `nesterov`, a = 0 | 0.5 | 0.5960 ± 0.097 | 10.74 s | 2.98 / 9 |
| async `nesterov`, a = 0.5 | 0.3 | 0.7103 ± 0.014 | 10.74 s | 2.98 / 9 |
| async `delayed_nesterov` N = 4, a = 0 | 0.7 | 0.7160 ± 0.009 | 10.74 s | 2.98 / 9 |
| async `delayed_nesterov` N = 4, a = 0.5 | 0.7 | 0.7186 ± 0.006 | 10.74 s | 2.98 / 9 |

Naive async + Nesterov loses 12 points and is unstable across seeds; the
delayed step with staleness weighting matches sync Nesterov at 2.2× less
simulated wall time (the slow-link workers no longer gate every round). On
this small task async `plain` FedAvg is equally good, so the result shows
that delayed Nesterov makes outer momentum safe under async, not that it
beats plain; a real-model run belongs with async regional aggregation (#91).

### Three-tier hierarchical async vs flat DiLoCo (#91, HALoS)

`python sim_fed.py --halos` compares flat DiLoCo with a three-tier
hierarchical async setup (HALoS, arXiv 2506.04531) on a 3-GPU skew fleet. The
sim models the three tiers like this:

1. **Workers** run time-paced local steps, like `FederationService`: a worker
   runs `speed × 0.1 s` steps per round. Speeds are 140 / 40 / 10 steps/s for
   the 4090 / 2080 Ti / P620, the 1:4:14 R50 skew. Data shards are
   proportional to speed.
2. **Regional aggregators** run synchronous block-mode rounds on the in-region
   LAN (1 GB/s, 0.5 ms), using the shipping `Leader` round math (int8 + error
   feedback, sample weights, plain outer step). They never wait for the WAN.
   After each regional round, if no drift is in flight, the aggregator uploads
   its drift `D = intended − g` to the root. When the reply arrives, it
   reconciles with the SAK-0011 rule `anchor += U − D`. The regional BN is left
   untouched, as in `region.py`.
3. **The root** sits in region A and commits each region's drift as soon as it
   arrives. The weight is `share_R · (1+τ)^(−a)`, where `share_R` is the
   region's throughput share and τ is the number of root versions since the
   region's base. The outer step is `delayed_nesterov` with N = 2 (the number
   of regions). The reply is the int8 encoding of `G − mirror_R`: whatever the
   region has not seen yet, so a broadcast residual is resent rather than
   lost. Between regions the link is 2 MB/s with 40 ms latency.

Flat DiLoCo is the star `Leader` in block mode on the same clock. Every round
waits for the slowest worker's WAN round trip.

The sim does not run the real `RegionAggregator` over sockets. Transport,
two-level consensus and rejoin are covered by
`tests/services/test_federation_region.py`, so this sim isolates the
algorithm.

Two layouts were run. In `fleet`, region A holds the 4090 and the 2080 Ti, and
region B holds the P620. In `split`, region A holds the 4090, and region B
holds the 2080 Ti and the P620.

Setup: 15 simulated seconds, evaluation every 0.25 s, 5 seeds (mean ± std),
best outer lr per arm from {0.3, 0.5, 0.7}. The target is 98 % of flat
DiLoCo's final accuracy (0.6958). "Rounds" counts outer commits: sync rounds
for flat, root commits for HALoS. Raw rows and per-seed histories are in
`results/e91_halos/results.json`.

| layout | arm | final acc (CV) | acc at flat's token budget | time → t98 | tokens → t98 | rounds → t98 | WAN MB |
|---|---|---|---|---|---|---|---|
| both | flat DiLoCo `nesterov` η 0.3 | 0.7100 ± 0.006 (0.008) | 0.7100 | 2.25 ± 0.57 s | 11.9k ± 3.2k | 9.8 ± 2.6 | 4.3 / 8.6 |
| fleet | HALoS root `plain`, a = 0.5 | 0.7065 ± 0.010 (0.014) | 0.7164 | 1.05 ± 0.37 s | 11.9k ± 4.5k | 14.2 ± 5.6 | 4.5 |
| fleet | HALoS root `nesterov`, a = 0.5 | 0.7030 ± 0.006 (0.008) | 0.6996 | 2.15 ± 0.34 s | 25.2k ± 4.2k | 30.6 ± 5.2 | 4.5 |
| fleet | HALoS root `delayed_nesterov`, a = 0 | 0.7143 ± 0.005 (0.007) | 0.7240 | **0.85 ± 0.12 s** | **9.4k ± 1.2k** | 11.2 ± 1.5 | 4.5 |
| fleet | HALoS root `delayed_nesterov`, a = 0.5 | **0.7208 ± 0.005** (0.007) | 0.7252 | 1.15 ± 0.49 s | 13.3k ± 6.0k | 16.0 ± 7.3 | 4.5 |
| split | HALoS root `plain`, a = 0.5 | 0.7035 ± 0.009 (0.012) | 0.7223 | 1.15 ± 0.49 s | 13.1k ± 6.0k | 16.0 ± 7.3 | 4.5 |
| split | HALoS root `nesterov`, a = 0.5 | 0.6948 ± 0.017 (0.025) | 0.6979 | 2.60 ± 1.00 s | 30.6k ± 12.1k | 37.6 ± 14.9 | 4.5 |
| split | HALoS root `delayed_nesterov`, a = 0 | 0.7138 ± 0.008 (0.011) | 0.7134 | **1.10 ± 0.20 s** | 12.3k ± 2.5k | 14.8 ± 2.9 | 4.5 |
| split | HALoS root `delayed_nesterov`, a = 0.5 | 0.7044 ± 0.013 (0.018) | 0.6788 | 1.95 ± 0.98 s | 22.8k ± 11.9k | 28.0 ± 14.7 | 4.5 |

The flat rows are identical in both layouts: the P620 gates every round
either way. Flat WAN traffic is 4.3 MB in `fleet` (one worker crosses the WAN)
and 8.6 MB in `split` (two cross).

Reading the table (CPU sim on a small synthetic task, not a real model):

- **Time to target: the hierarchy wins by 2.0 to 2.6×.** The regions never
  wait for the WAN, so they process about 2.1× more samples per simulated
  second than the star (180k vs 85k tokens in 15 s).
- **Tokens: mixed, and short of HALoS's claim in one layout.** Only `fleet`
  with `delayed_nesterov` a = 0 cuts tokens to t98 by more than 10 % (−21 %).
  `split` delayed_nesterov a = 0 uses +3 % tokens, and the a = 0.5 variants use
  +12 % (fleet) and +91 % (split). At the looser 95 % target, every HALoS arm
  uses 1.9 to 3.3× flat's tokens, because flat reaches it in about 3 rounds. On a
  task that saturates this quickly, the async tier buys wall time, not sample
  efficiency.
- **Rounds: more, not fewer.** Root commits to target are 11 to 16 versus 9.8
  flat rounds. Each root commit carries one region's drift, not the whole
  fleet's, so "rounds" counts more but cheaper events. The parent acceptance
  ("beats flat DiLoCo on rounds-to-target") holds in wall time, not in commit
  count.
- **Quality: no regression.** Final accuracy matches flat or beats it (the
  best is +1.1 pt in `fleet`). Measured at flat's own token budget, the
  delayed-Nesterov a = 0 arm is at or above flat in both layouts.
- **Naive async Nesterov degrades, delayed Nesterov recovers it.** Plain
  `nesterov` at the root needs 2.1 to 2.6× flat's tokens and has the worst
  CV. `delayed_nesterov` fixes this, matching #90.
- **The staleness discount hurts here.** Staleness is small (mean τ ≈ 1,
  max 2). With a = 0.5, each region's drift is cut by about 30 % with nothing
  gained in return, which slows the split layout. Use a = 0 when τ is this
  small.
- **WAN bytes** halve in `split`, because one aggregator replaces two workers
  on the WAN (8.6 → 4.5 MB). In `fleet` they are about equal (4.3 vs 4.5 MB):
  B pushes about every second regional round, roughly as often as flat runs
  rounds.

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
  worker's uncommitted local progress — correct by protocol, and rare. The
  inner optimizer keeps its momentum unless `rejoin_momentum` says otherwise.
- **Joining a running leader (SAK-0014).** By default the roster is fixed:
  ids outside `expected` are rejected. With `allow_join=` or `join_token=`
  on the leader, an authorised new worker gets a roster slot at handshake,
  is bootstrapped by the same FULL_SYNC as any fresh worker, and takes part
  in the next round. Leaving needs nothing special: when a worker shuts down
  it is dropped from the round in flight and later rounds commit without it.
  Anyone holding the join token can push updates into the anchor; protect it
  like the mesh keys.
- **Leader restart.** With `persist_path` set, state is persisted atomically
  on every commit and a restarted leader resumes at its last committed
  round; workers reconnect (with backoff) and resync as above. Without it, a
  restarted leader re-initializes from `seed` and FULL_SYNCs a fresh random
  model over every trained worker — set `persist_path`. See
  [Leader availability](#leader-availability) for the full story.
- The leader sends heartbeats every 5 s (also mid-round), so workers detect
  a dead leader even during long transfers; receive paths skip them
  transparently.

## Leader availability

There is **one leader, no standby and no election** (SAK-0015). Availability
comes from a **supervised restart from the atomic persisted anchor**:

1. Run the leader under a supervisor that restarts it on exit (systemd
   `Restart=always`, or a single-replica Kubernetes Deployment with
   `strategy: Recreate`). Never run two leaders for the same fleet, not even
   briefly during a rollout: two leaders broadcasting different bytes fork
   the anchor, which is the one failure this design cannot repair.
2. Set `persist_path` to storage that **survives the restart** (a host or
   persistent volume, not the container's writable layer). The file holds
   `{round, anchor, bn}`, the link plan (SAK-0013) and the leader's
   optimizer state as of that same round, and is written with tmp + `os.replace` after every commit, so it
   always holds the last fully acked round. The optimizer state is every
   buffer the configuration keeps: the broadcast error-feedback residual
   `ef_bcast` (lossy codecs), the outer momentum `outer_m` (`sgdm`,
   `slowmo`, `nesterov`, `delayed_nesterov`), the delayed-Nesterov
   pseudo-gradient buffer `outer_buf`, and the per-shard update counts
   (each streaming shard keeps its own window). **Size:** each buffer is
   one fp32 copy of the parameters, so the file is 1x the model with
   `fp32raw` + `plain`, 2x with a lossy codec + `plain`/`ef`, 3x with a
   lossy codec + momentum, and 4x with a lossy codec + `delayed_nesterov`
   (plus BN). That is written once per commit; size the volume and the
   per-round write bandwidth accordingly. Files written before the
   optimizer state was persisted still load: the anchor resumes and the
   missing buffers start at zero. A buffer the restarted configuration does
   not keep (for example after switching `outer_opt`) is ignored with a
   warning.
3. Restart on the same host/port with the same `model_fn`, `codec` and
   `expected`: workers have the leader address fixed.

While the leader is down, workers keep training solo (a round in flight is
dropped like an abort, which also releases `mode="block"`) and
reconnect with exponential backoff from 1 s, capped at 30 s, so they find the
restarted leader within ~30 s of it listening. On reconnect each worker's
`{last committed round, anchor hash}` goes through the normal rejoin path:

| Leader killed… | Persisted round | Worker state | Rejoin path |
|---|---|---|---|
| between rounds, or while gathering | r−1 | committed r−1, hash matches | nothing; the solo progress is their next delta |
| mid-broadcast (some workers applied r) | r−1 | applied r: committed r | FULL_SYNC back to r−1 |
| | | did not apply r: committed r−1 | nothing |
| after every ack, before the persist | r−1 | committed r | FULL_SYNC back to r−1 |

The restarted leader then re-runs round r from the persisted anchor. A
round is therefore never applied twice: a worker that applied the lost round
is rolled back by FULL_SYNC, never handed a second delta on top (the
retained-round resend is not persisted, so it cannot fire after a restart).
`tests/services/test_leader_restart.py` SIGKILLs a child-process leader at
each of these three points and checks that the restarted leader resumes at
the persisted round and that the next rounds commit with all three workers
and the consensus hash green, every worker ending on the leader's anchor
bit-exact. It also checks that the optimizer state in the file is the
snapshot of the last committed round (not of the lost one), that the
restarted leader loads it bit-exact, and that per-shard step counts carry on
(`delayed_nesterov` with two streaming shards).

**What a restart costs:** at most the one in-flight round; the local progress
of any worker that is FULL_SYNCed (it applied the lost round), including its
solo training during the outage. The leader's broadcast-side error-feedback
residual, outer momentum and delayed-Nesterov buffer are restored bit-exact
from the file, as of the persisted round, so the re-run round continues the
optimizer exactly where the last commit left it. The retained-round resend
is still not persisted.

**What it does not cover:** losing the leader's *host and its disk*. A leader
without its `persist_path` file re-initializes from `seed` and FULL_SYNCs an
untrained model over the fleet; recovering from that needs the file on
storage that outlives the host.

**When to revisit (would reopen SAK-0015):** a measured bound on how long
workers can train solo before the next merge hurts convergence that is
shorter than supervisor restart plus the 30 s reconnect cap; a deployment
that cannot give the leader durable storage; or an availability target for
the round cadence itself. Hierarchical aggregation (SAK-0011) does not
change the story: each region aggregator would be a leader of its own and
follow the same supervised-restart rule.

## Co-distillation round mode (SAK-0012, E8)

Weight averaging needs every worker on the same architecture. The
co-distillation mode (`sakura/services/federation/codistill.py`) exchanges
predictions instead: each worker evaluates its own model on a shared probe
batch and uploads the logits, the leader sends back the mean of the *other*
workers' logits, and each worker adds a KL term toward it. It runs on its own
port beside the weight-delta `Leader` and does not change it.

```python
from sakura.services.federation.codistill import (
    CodistillLeader, CodistillWorker, kd_loss, probe_indices)

leader = CodistillLeader(7600, ["a", "b"], probe_rows=512, num_classes=10,
                         probe_seed=0, gap_s=5.0).start()        # CPU, no model

cd = CodistillWorker("a", "leader-host", 7600, arch="resnet18").start()
probe_x = public_pool[probe_indices(cd.probe_seed, cd.probe_rows, len(public_pool))]
for step, (x, y) in enumerate(loader):
    loss = F.cross_entropy(model(x), y)
    tgt = cd.target()                       # latest ensemble target or None
    if tgt is not None and step >= burn_in_steps:
        rows = torch.randint(0, cd.probe_rows, (128,))
        loss = loss + 0.25 * kd_loss(model(probe_x[rows]), tgt[rows].to(dev))
    if step % 20 == 0:
        with torch.no_grad():
            cd.publish(model(probe_x))      # uploaded at the next round
    ...
```

| `CodistillLeader` parameter | Default | Meaning |
|---|---|---|
| `port` | required | TCP port; `0` binds an ephemeral port (read `leader.port` after `start()`). |
| `expected` | required | Allowed `worker_id`s. |
| `probe_rows`, `num_classes` | 512, 1000 | Shape of the logit matrix every worker publishes. |
| `probe_seed` | 0 | Handed to workers at handshake; every site derives the same probe from it. The probe itself never travels. |
| `gap_s` | 5.0 | Round period. Rounds never block training. |
| `target_codec` | `"int8"` | Downlink target encoding: per-row-scaled `"int8"` or `"fp16"`. |
| `max_stale_rounds` | 4 | A worker's last logits stay in the ensemble this many rounds after it stops refreshing them. |

**Wire cost** per worker per round for a 512x1000 probe, counted on the
socket: 1,024,016 B up (fp16 logits) + 514,081 B down (int8 target) =
**1,538,097 B** (2,048,049 B with an fp16 target), against about 25 MB for
an int8 ResNet-50 delta.

**Probe choice.** Use rows of a public, in-distribution, unlabelled pool
every site holds (`probe_indices`); raw data and labels never leave a site.
`make_probe` (Gaussian noise) needs no shared data but was worse than the
public probe in every CIFAR-10 configuration below.

**Measured (CIFAR-10, RTX 2080 Ti, plain CNN vs small ResNet on separate
shares, 2500 steps x 128 each, both through the real leader/worker over
TCP; `python experiments/codistill_cifar.py`, results in `results/codistill_e8/`).**
Test accuracy change vs each model trained solo on the same share:

| Split | KL setting | Probe | Site A (CNN) | Site B (ResNet) |
|---|---|---|---|---|
| IID halves | weight 1.0 from step 0, 3 seeds | public | -3.6 pp | -1.9 pp |
| IID halves | weight 1.0 from step 0, 3 seeds | synthetic | -6.8 pp | -11.8 pp |
| label skew 90/10 | weight 1.0 from step 0, 3 seeds | public | -8.1 pp | -3.0 pp |
| label skew 90/10 | weight 1.0 from step 0, 3 seeds | synthetic | -19.7 pp | -9.8 pp |
| label skew 90/10 | weight 0.25 after 30% burn-in, 2 seeds | public | **+2.6 pp** | **+1.9 pp** |
| IID halves | weight 0.25 after 30% burn-in, 2 seeds | public | -0.6 pp | -0.1 pp |

So: it transfers knowledge only when the sites' data differ, only with a
light, delayed KL term, and the gain is about 2 pp (hyperparameters picked
after the first run, so an upper estimate). The KL step also roughly halves
per-site img/s on one GPU with a 128-row probe slice per step; the
reference-fleet throughput criterion is still open (#102), which is why
SAK-0012 stays Proposed.

## Compression: SparseLoCo (`topk2` + `outer_opt="ef"`)

`topk2` (bucket-local top-2% + 2-bit values) with the `"ef"` outer step
follows SparseLoCo (arXiv 2508.15706): the leader keeps no outer-momentum
buffer; its broadcast error-feedback residual is decayed by μ and is the
momentum. CPU sim (`python sim_fed.py --sparseloco`, the real codec /
outer-step / BN-merge code on the synthetic teacher-MLP task, 4 workers with
1:2:4:8 shards, 60 block-mode rounds × 10 local steps, 5 seeds, target =
95% of a solo control's accuracy = 0.688; the GPU host CPU). Bytes are up + down
per worker per round, measured on the payloads; raw numbers in
`results/e7e8_sim/sparseloco.json`:

| codec | outer | lr / μ | B/round/worker | vs int8 | rounds to target (mean ± sd, hits) | final acc (mean ± sd) | CV |
|---|---|---|---|---|---|---|---|
| int8 (shipped) | plain | 1.0 / – | 57 400 | 1.0× | 8.4 ± 2.2 (5/5) | 0.7186 ± 0.007 | 0.009 |
| int8 | nesterov | 0.5 / 0.9 | 57 400 | 1.0× | 10.2 ± 1.5 (5/5) | 0.7148 ± 0.010 | 0.013 |
| int2 | plain | 1.0 / – | 14 392 | 4.0× | 22.8 ± 3.1 (5/5) | 0.7134 ± 0.013 | 0.018 |
| int2 | ef | 1.0 / 0.9 | 14 392 | 4.0× | 17.4 ± 3.5 (5/5) | 0.7187 ± 0.007 | 0.009 |
| topk2 | plain | 1.0 / – | 2 648 | 21.7× | 15.6 ± 2.2 (5/5) | 0.7224 ± 0.005 | 0.007 |
| topk2 | nesterov | 0.5 / 0.9 | 2 648 | 21.7× | 33.6 ± 2.2 (5/5) | 0.7108 ± 0.009 | 0.013 |
| topk2 | ef | 1.0 / 0.9 | 2 648 | 21.7× | 15.6 ± 2.9 (5/5) | 0.7211 ± 0.004 | 0.006 |
| **topk2** | **ef** | **0.5 / 0.95** | **2 648** | **21.7×** | **13.2 ± 1.5 (5/5)** | **0.7246 ± 0.011** | 0.015 |

Reading it honestly:

- `topk2`/`ef` sends 21.7× fewer bytes per round than int8 and ends at the
  same accuracy (within one sd), but needs **more rounds** to reach the
  target: 13.2 vs 8.4 (about 1.6×). In bytes to target it is about 14×
  cheaper (35 KB vs 482 KB per worker), so it wins wherever the round is
  bandwidth-bound (the WAN regime of this page), and loses where rounds are
  compute-bound.
- `ef` is the right outer step for aggressive sparsity: Nesterov's
  persistent buffer plus a 2%-dense codec overshoots (33.6 rounds), while
  `ef` at μ = 0.95 is the fastest `topk2` row with no extra full-length
  buffer on the leader.
- This is a small synthetic task on CPU; it has not been run on the WAN
  fleet. `int8`/`plain` stays the default.

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

## Scope advisory

Federation does not check whether it is the right tool, but it tells you
(SAK-0017). At the first committed round the leader measures each worker's
throughput (`n_samples / window_s`, the wall-clock window the worker reports
with its upload) and logs one line; every worker logs the same advisory at
its first completed round:

```
federation scope advisory: WARN (fleet 9253.6 samples/s = 2.073x fastest worker '2080ti'; solo epoch 11.2s; assumes IID shards, same architecture on every worker): '2080ti' alone would sweep all 50000 samples every 11.2s (< 60s): ...
```

The rule, with `r_max` the fastest worker's rate, `S` the fleet's rate over
`r_max` and `N` the dataset size:

| Check | Warns when | Why |
|---|---|---|
| Fleet speedup | `S < 1.2` | The fleet adds under 20% over its fastest member: staleness and drift for almost no extra throughput. |
| Solo epoch time | `N / r_max < 60 s` | One GPU sweeps the whole dataset in under a minute, so it can finish the job alone and the fleet's staleness cost likely exceeds its gain. |

The verdict is `warn` if either check fires, `ok` only if both ran and
passed, and `unchecked` when `N` is unknown (set `Leader(dataset_size=...)`,
or let each worker report `shard_size`). The two measured runs are the test
fixtures: E5 ResNet-50@224 (`S` 1.46, solo epoch ~790 s on an
ImageNet-1k-sized set: **ok**, and federation won by 43%) and E6
CIFAR-10/ResNet-18 (`S` 2.07, solo epoch ~11 s: **warn**, and the solo 4090
won by ~2 pp). E6 is why speedup alone is not enough: that fleet doubled the
throughput and still lost.

It is an advisory, never a refusal: two data points, one run each, separate
the thresholds with margin (5x and 13x) but do not make a fitted model. The
advisory is in `Leader.scope_advisory`, `FederationService.scope_advisory`
and the first `LEADER_ROUND` record (`scope_advisory`) for telemetry.

**Non-IID data is out of scope.** The advisory, like every number on this
page, assumes IID shards and the same architecture on every worker. Shards
with different label or feature distributions add client drift that local-SGD
averaging does not correct and that nothing here detects; an `ok` verdict
says nothing about them. Measure convergence against a single-GPU control
before trusting federation on non-IID data.

## Bandwidth reduction vs data-parallel (the INTELLECT-1 comparison)

INTELLECT-1 reports **400× less bandwidth than data-parallel** with 83–96%
compute utilisation. That number is a product of two choices: DiLoCo syncs
once every H = 100 inner steps (100×), and its all-reduce ships int8 instead
of fp32 (4×). sakura's honest equivalent counts the same thing — bytes per
optimizer step against a DDP baseline that all-reduces fp32 gradients every
step:

```
factor_w = Σ_rounds steps_w × 4·numel / Σ_rounds payload_bytes_w
         = H_eff,w × (fp32 bytes / int8 payload incl. scales) [× K when streaming]
```

Two differences from INTELLECT-1 matter. sakura's rounds are **time-based**,
so H_eff is not configured: each GPU's H_eff is its throughput × round period
/ batch, and under the fleet's 56:1 skew it varies about 14× between the
P620 and the 4090. The factor is therefore **per worker**; the *slowest
worker's* factor is the conservative headline and the step-weighted *fleet*
factor the generous one. Second, the int8 payload carries one fp32 scale per
4096-element bucket plus padding, so the codec contributes **3.996×**, not 4×.

Reproduce with `python sim_fed.py --report` (CPU, no Rust build; logic in
`sakura/services/federation/report.py`). It writes
`results/bandwidth_report/report.{json,md}`, and labels every number:
**measured** (parsed from the E3–E6 fleet logs under `results/`), **sim**
(bytes produced by the shipping codec on CPU) or **model** (a wall-time model
over the measured 248 ms / 11.5 MB/s link).

**Measured on the fleet** (ResNet-50, 3 GPUs, 2 regions, 56:1 skew; factor
vs fp32 DDP; compute util = aggregate img/s / sum of concurrent solos):

| Run | Config | H_eff P620 / 2080 Ti / 4090 | Bytes per sync vs fp32 | Factor, slowest | Factor, fleet | Compute util |
|---|---|---|---|---|---|---|
| E3 | fp32 block, T=30 s | 53 / 232 / 775 | 1× | 53× | 353× | 66.3% |
| E4 | int8 block, T=30 s | 53 / 228 / 777 | 4.0× | 212× | 1 410× | 81.9% |
| **E5** | **int8 overlapped, T=30 s** | **66 / 285 / 970** | **4.0×** | **265×** | **1 760×** | **97.5%** |
| E5s | + streaming K=4, T=8 s | 26 / 109 / 377 | 16.0× | 415× | 2 730× | 96.6% |
| E5r2 | int8 overlapped, self-paced | 30 / 128 / 432 | 4.0× | 120× | 769× | 95.8% |
| E6 | CIFAR-10 ResNet-18, int8, T=15 s | 62 / 199 / 196 | 4.0× | 249× | 609× | n/a |

Synchronous DDP over the same mesh measured 1.03 img/s = **0.04%** compute
utilisation. The model puts DDP at 0.5% even with an ideal ring all-reduce on
this link, and at **13.5% on an infinitely fast network**: lockstep DDP waits
for the P620 every step, so the skew alone caps it. (E6 has no ResNet-50
solo ceiling, so its compute utilisation is not reported.)

**The honest sakura equivalent of INTELLECT-1's 400×:** with the shipped
int8 codec and the shipped overlapped rounds (E5), sakura moves **265× less
than fp32 DDP per optimizer step for the slowest GPU and 1 760× for the
fleet as a whole, at 97.5% compute utilisation**, on measured bytes. With
streaming shards (K=4) it is 415× / 2 730× at 96.6%. Self-pacing to 50% of the
link (E5r2) trades factor for freshness: rounds every ~6.5 s give
120× / 769×. The factor is not a constant of the design: it scales linearly
with round period and with each GPU's step rate, and the codec contributes
only 4×.

Caveats: a local step is not worth a DDP step for convergence (DDP's global
batch is the sum of the workers' batches; local-SGD drifts between syncs), so
the factor compares wire bytes, not training progress; E6 is the convergence
check. INTELLECT-1's 400× is per sync at a fixed H on a homogeneous fleet;
sakura's slowest-worker number is the closer analogue. The "model" rows
assume all workers share the WAN link to the leader, while 2 of 3 workers sit
on the leader's LAN, so they are pessimistic for sakura (the model's 68% for
block rounds vs E4's measured 81.9%).

## Dynamic membership under churn (the INTELLECT-1 validation)

INTELLECT-1 trained with ~30 providers joining and leaving mid-run. The same
stress runs here against the **real** `Leader` and `FederationService` over
loopback (one thread per worker, leader bound to port 0), not a sim-level
model: the point is to exercise the shipping SAK-0014 join path, the
FULL_SYNC bootstrap and the consensus hash, and a full run of 30 rounds at a
1 s period takes about 40 s, so it fits in a test. A controller applies a
churn schedule at round boundaries: token joins of workers outside the
roster, graceful leaves, operator `retire()`, `admit()` re-authorisation of a
retired roster member, and reconnects into a kept slot. Every (re)start is a
fresh `FederationService`, i.e. a restarted process.

A **hash discrepancy** is a committed round with `consensus_ok = false`, or a
worker (at the end, or snapshotted when it left) whose anchor+BN hash is not
the leader's hash for the round it last committed.

```bash
python sim_fed.py --dynamic          # 2 seeds x (churn, static control)
pytest tests/services/test_dynamic_membership.py
```

Logic in `sakura/services/federation/churn.py`; results in
`results/dynamic_membership/results.json`. Measured on CPU (8-class teacher
task from `sim_fed.py`, 4 initial workers + 4 that join later, int8, 30
rounds, event every 2 rounds):

| | Churn (2 seeds) | Static control (2 seeds) |
|---|---|---|
| Join / leave events | 14 / 12 (11 token joins, 2 slot reconnects, 1 admit-rejoin; 7 leaves, 5 retires) | 0 / 0 |
| Rounds committed | 60 / 60 | 60 / 60 |
| Participants per round | 2–8 | 4 |
| **Hash discrepancies** | **0** | **0** |
| Final test accuracy (mean) | 0.711 | 0.693 |
| Final test loss (mean) | 1.42 | 1.92 |

Every joiner took part in the round right after its join was applied. The
accuracy difference is not a churn penalty or bonus in isolation: the 8
workers each own one eighth of the training set, so the churn run sees
shards the 4-worker control never trains on, and round cadence is
wall-clock based. What the run shows is that churn costs no correctness and
no measurable accuracy. The test replays a fixed 9-event schedule covering
every path above over 16 rounds and requires zero discrepancies.

## Experimental: ES round mode (EGGROLL, low-rank perturbations)

`sakura/services/federation/es.py` holds seeded low-rank evolution-strategies
primitives after EGGROLL (arXiv 2511.16652): from a 63-bit seed each weight
matrix gets a perturbation `E = A Bᵀ / sqrt(r)` (`A`, `B` standard normal,
rank `r`); vectors get dense noise. Members come in antithetic pairs
(`+E`, `-E`) that share one seed, so a pair costs **16 B** on the wire
(u64 seed + two f32 fitness values) whatever the model size. Any node holding
the anchor rebuilds the ES gradient from `(seed, f⁺, f⁻)` alone
(`es_gradient`, centered-rank fitness shaping). This is *not* wired into
`FederationService`/`Leader`; it exists only in the CPU simulator so far
(`python sim_fed.py --es`, results in `results/es_sim/`).

Synthetic task from `sim_fed.py` (64→128→128→8 MLP with BN, ~27k params,
4 workers on 1:2:4:8 shards, 3 seeds; ES: 16 pairs/worker, σ=0.02, lr=0.01,
batch 256; backprop: H=10 local SGD steps). Solo control 0.718; t85 = 0.610,
t95 = 0.682 (majority class = 0.46). WAN time is modeled on the E5 link
(11.5 MB/s, 248 ms RTT) as CPU s/round + one worker's bytes / bandwidth + RTT.

| Arm | Final acc | Rounds → t85 | Rounds → t95 | CPU s → t85 | WAN s → t85 | B/round up (all workers) | B/round down |
|---|---|---|---|---|---|---|---|
| backprop int8 + sgdm (60 rnd) | 0.724 ± 0.004 | 3 | 7 | 0.15 | 0.9 | 114 800 | 114 800 |
| backprop fp32 + plain (60 rnd) | 0.721 ± 0.002 | 3 | 6 | 0.18 | 1.0 | 422 016 | 422 016 |
| ES rank 1 (300 rnd) | 0.668 ± 0.006 | 42 | never | 4.3 | 14.7 | 1 056 | 4 224 |
| ES rank 4 (300 rnd) | 0.662 ± 0.007 | 38 | never | 4.5 | 13.9 | 1 056 | 4 224 |

Reading it honestly: ES moves ~109× fewer bytes per round than int8 deltas
(~400× fewer than fp32), and that gap grows linearly with model size (a
ResNet-50 int8 delta is ~25 MB per worker per round; the ES message stays at
264 B). But on this task ES needs **~14× more rounds to t85 (~16× modeled WAN
time, ~29× CPU time), never reaches t95 in 300 rounds, and plateaus ~5 pp below backprop**. The
link here is not the bottleneck (a worker's int8 delta takes ~3 ms of a
0.3 s round), so the byte savings buy nothing. ES only becomes interesting where
the per-round transfer dominates (very large models on thin links) or where
workers cannot run backprop (inference-only hardware). Rank 1 and rank 4
are indistinguishable here, consistent with EGGROLL's claim that low rank
does not cost quality. BN statistics are not shipped in ES mode; each node
recalibrates them for the anchor from local data (compute, not bytes).

Scope: low-rank ES for pretraining (not only fine-tuning) is a research
track with explicit go/no-go criteria, not a supported training mode; see
SAK-0018 in the ADR index.
