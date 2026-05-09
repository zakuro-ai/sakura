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
