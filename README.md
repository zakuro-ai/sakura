<h1 align="center">Sakura</h1>

<h3 align="center">Faster PyTorch training, without rewriting your training loop.</h3>

<p align="center">
  <a href="https://pypi.org/project/sakura-ml/"><img alt="PyPI" src="https://img.shields.io/pypi/v/sakura-ml?color=blue"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License" src="https://img.shields.io/github/license/zakuro-ai/sakura.svg?color=blue">
</p>

<p align="center">
  <a href="https://docs.zakuro-ai.com/sakura/overview"><b>Documentation</b></a> •
  <a href="https://docs.zakuro-ai.com/sakura/quickstart">Quickstart</a> •
  <a href="https://docs.zakuro-ai.com/sakura/services">Services</a> •
  <a href="https://docs.zakuro-ai.com/sakura/benchmarks">Benchmarks</a> •
  <a href="CHANGELOG.md">Changelog</a>
</p>

---

Sakura plugs into PyTorch DDP, Lightning and the Hugging Face Trainer, and adds training services you switch on one line at a time: evaluation and checkpointing that run off the training thread, mixed precision, `torch.compile` and ZeRO-1 optimizer sharding. Your model, data and loop stay as they are.

## Try it in one command

```bash
git clone https://github.com/zakuro-ai/sakura && cd sakura
uv run demo.py
```

No GPU, no dataset download, no build step: [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installs the published `sakura-ml` wheel (and CPU-only PyTorch on Linux) into a throwaway environment, then trains the same small classifier twice, plain PyTorch first and then with Sakura's `AsyncEval`, and prints both timings. Each setup is first given its own fastest CPU thread count, so plain PyTorch isn't handicapped. On an Apple-silicon Mac it takes about 20 s from `git clone` to this, install included:

```
  setup           threads  median time   final val acc
  plain PyTorch         1       0.43 s           0.779
  with Sakura           1       0.32 s           0.779

  1.36x faster on this machine (27% less wall-clock), same model, same accuracy:
  evaluation now overlaps the next epoch instead of blocking it.
```

Your numbers will differ (five runs on that Mac: 1.31x to 1.36x). On a 2017 4-core Intel laptop the same demo measures about 1.03x, because training there already keeps every core busy and there is nothing for evaluation to overlap with; the demo says so instead of hiding it. The whole demo is one file, [`demo.py`](demo.py); the Sakura part is about fifteen lines of `run_sakura()`.

## Use it

```bash
pip install sakura-ml                              # wheels for macOS, Linux and Windows, Python 3.10+
pip install 'sakura-ml[lightning,huggingface]'     # with the framework integrations
```

On Linux, add `--extra-index-url https://download.pytorch.org/whl/cpu` to get the CPU-only PyTorch build instead of the multi-GB CUDA one.

```python
from sakura import SakuraRuntime
from sakura.adapters import DDPAdapter            # or LightningAdapter, HFAdapter
from sakura.dispatch import ThreadDispatcher
from sakura.services import AsyncEval

with SakuraRuntime() as rt:
    rt.install(AsyncEval(eval_fn=eval_epoch,       # eval_fn(epoch, payload) -> dict, run off the training thread
                         eval_payload=snapshots, dispatcher=ThreadDispatcher()))
    loop = DDPAdapter(rt, rank=0, world_size=1)
    loop.on_train_begin(model, optimizer, train_loader)
    for epoch in range(epochs):
        loop.on_epoch_begin(epoch)
        train_one_epoch(model, optimizer, train_loader)
        snapshots[epoch] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        loop.on_epoch_end(epoch, model, optimizer, metrics={})  # evaluation starts in the background
    loop.on_train_end(model)                                      # waits for the last evaluation
```

[`demo.py`](demo.py) is the complete, runnable version. Lightning and the Hugging Face Trainer take a callback instead of hook calls: see the [quickstart](https://docs.zakuro-ai.com/sakura/quickstart).

Everything else is in the **[documentation](https://docs.zakuro-ai.com/sakura/overview)**: [install options](https://docs.zakuro-ai.com/install/sakura), [services](https://docs.zakuro-ai.com/sakura/services), [adapters and dispatchers](https://docs.zakuro-ai.com/sakura/adapters-dispatchers), [benchmarks](https://docs.zakuro-ai.com/sakura/benchmarks) and [migrating from 0.1](https://docs.zakuro-ai.com/sakura/migration). Sending work to a separate `sakura-worker` runs your callable there: read the [dispatch trust model](docs/dispatch.md) first. Sakura is part of [Zakuro](https://github.com/zakuro-ai/zakuro).

---

<p align="center">
  If Sakura saves you time, a ⭐ helps other people find it, and it's the easiest way to follow new releases.<br>
  <a href="CONTRIBUTING.md">Contributing</a> • <a href="SECURITY.md">Security</a> • <a href="LICENSE">BSD-3-Clause</a>
</p>
