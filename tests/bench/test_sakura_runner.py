"""SakuraRunner smoke test — adapter + Telemetry observed."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.bench.harness import SakuraRunner, Workload
from sakura.services.telemetry import Telemetry


def _make_synthetic_workload() -> Workload:
    def make_model():
        return torch.nn.Linear(4, 2)

    def make_loader():
        torch.manual_seed(0)
        ds = torch.utils.data.TensorDataset(
            torch.randn(8, 4), torch.randint(0, 2, (8,))
        )
        return torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

    def eval_fn(model, loader):
        return {"val_acc": 0.5}  # constant; suffices for smoke

    return Workload(
        name="sakura-runner-smoke",
        tier="smoke",
        make_model=make_model,
        make_train_loader=make_loader,
        make_val_loader=make_loader,
        eval_fn=eval_fn,
        epochs=1,
    )


def test_sakura_runner_telemetry_observes_events():
    sink: list[dict] = []
    runner = SakuraRunner(framework="pytorch-ddp", services=[Telemetry(output=sink.append)])
    report = runner.run(_make_synthetic_workload())

    assert report.workload == "sakura-runner-smoke"
    assert "telemetry" in report.sakura_services
    # Telemetry should have observed OnTrainBegin, OnEpochBegin, OnTrainStepBegin (×2),
    # OnOptimizerStep (×2), OnEpochEnd, OnTrainEnd — at least 7 events.
    assert len(sink) >= 7
    assert any(r["event"] == "OnTrainBegin" for r in sink)
    assert any(r["event"] == "OnEpochEnd" for r in sink)


def test_sakura_runner_lightning_telemetry_observes_events():
    """SakuraRunner with framework='lightning' installs LightningAdapter and observes events."""
    pytest.importorskip("lightning")
    sink: list[dict] = []
    runner = SakuraRunner(framework="lightning", services=[Telemetry(output=sink.append)])
    report = runner.run(_make_synthetic_workload())

    assert report.workload == "sakura-runner-smoke"
    assert report.framework == "lightning"
    assert "telemetry" in report.sakura_services
    # Lightning adapter emits OnTrainBegin, OnEpochBegin, OnTrainStepBegin per batch,
    # OnBeforeOptimizerStep per batch, OnEpochEnd, OnTrainEnd. At minimum we see the
    # epoch lifecycle.
    assert len(sink) >= 4
    assert any(r["event"] == "OnTrainBegin" for r in sink)
    assert any(r["event"] == "OnEpochEnd" for r in sink)
    assert any(r["event"] == "OnTrainEnd" for r in sink)


def _make_hf_synthetic_workload() -> Workload:
    """HF-shaped synthetic workload — see test_baseline_runner._make_hf_synthetic_workload."""
    class _TinyHFModel(torch.nn.Module):
        def __init__(self, vocab=100, dim=8, n_classes=2):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab, dim)
            self.head = torch.nn.Linear(dim, n_classes)

        def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
            x = self.embed(input_ids).mean(dim=1)
            logits = self.head(x)
            out = {"logits": logits}
            if labels is not None:
                out["loss"] = torch.nn.functional.cross_entropy(logits, labels)
            return out

    def make_model():
        return _TinyHFModel()

    def collate(rows):
        return {
            "input_ids": torch.stack([r["input_ids"] for r in rows]),
            "attention_mask": torch.stack([r["attention_mask"] for r in rows]),
            "labels": torch.stack([r["labels"] for r in rows]),
        }

    def make_loader():
        torch.manual_seed(0)
        n, seq_len = 8, 4

        class _DS(torch.utils.data.Dataset):
            def __len__(self_):
                return n

            def __getitem__(self_, i):
                return {
                    "input_ids": torch.randint(0, 100, (seq_len,)),
                    "attention_mask": torch.ones(seq_len, dtype=torch.long),
                    "labels": torch.randint(0, 2, ()),
                }

        return torch.utils.data.DataLoader(_DS(), batch_size=4, collate_fn=collate)

    def eval_fn(model, loader):
        return {"val_acc": 0.5}

    return Workload(
        name="sakura-hf-runner-smoke",
        tier="smoke",
        make_model=make_model,
        make_train_loader=make_loader,
        make_val_loader=make_loader,
        eval_fn=eval_fn,
        epochs=1,
    )


def test_sakura_runner_hf_telemetry_observes_events():
    """SakuraRunner with framework='hf-trainer' installs HFAdapter and observes events."""
    pytest.importorskip("transformers")
    sink: list[dict] = []
    runner = SakuraRunner(framework="hf-trainer", services=[Telemetry(output=sink.append)])
    report = runner.run(_make_hf_synthetic_workload())

    assert report.workload == "sakura-hf-runner-smoke"
    assert report.framework == "hf-trainer"
    assert "telemetry" in report.sakura_services
    # HF adapter emits OnTrainBegin + OnTrainEnd at minimum; per-step + per-epoch
    # depend on Trainer's hook firings (logging_strategy="no" doesn't suppress
    # the lifecycle hooks themselves).
    assert any(r["event"] == "OnTrainBegin" for r in sink)
    assert any(r["event"] == "OnTrainEnd" for r in sink)
