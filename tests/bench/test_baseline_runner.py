"""BaselineRunner smoke test on a tiny synthetic workload."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sakura.bench.harness import BaselineRunner, Workload


def _make_synthetic_workload() -> Workload:
    """4-batch tiny MLP — runs in <1s."""
    def make_model():
        return torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
        )

    def make_loader():
        # 32 samples × 8 features → 4 classes; batch size 8 → 4 batches.
        # Use a real DataLoader because Lightning's Trainer.fit requires one
        # (or auto-wraps an iterable, which produces unexpected batch shapes).
        torch.manual_seed(0)
        ds = torch.utils.data.TensorDataset(
            torch.randn(32, 8), torch.randint(0, 4, (32,))
        )
        return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)

    def eval_fn(model, loader):
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                pred = model(x).argmax(dim=-1)
                correct += int((pred == y).sum())
                total += int(y.numel())
        return {"val_acc": correct / max(total, 1)}

    return Workload(
        name="synthetic-tiny-mlp",
        tier="smoke",
        make_model=make_model,
        make_train_loader=make_loader,
        make_val_loader=make_loader,
        eval_fn=eval_fn,
        epochs=1,
    )


def test_baseline_runner_pytorch_ddp_completes():
    wl = _make_synthetic_workload()
    runner = BaselineRunner(framework="pytorch-ddp")
    report = runner.run(wl)

    assert report.workload == "synthetic-tiny-mlp"
    assert report.framework == "pytorch-ddp"
    assert report.elapsed_secs > 0
    assert report.samples_per_sec > 0
    assert "val_acc" in report.final_metrics
    assert report.git_sha != "" or report.git_sha == ""  # either is fine


def test_baseline_runner_lightning_completes():
    """Lightning baseline auto-wraps the nn.Module in a LightningModule."""
    pytest.importorskip("lightning")
    wl = _make_synthetic_workload()
    runner = BaselineRunner(framework="lightning")
    report = runner.run(wl)

    assert report.workload == "synthetic-tiny-mlp"
    assert report.framework == "lightning"
    assert report.elapsed_secs > 0
    assert report.samples_per_sec > 0
    assert "val_acc" in report.final_metrics


def _make_hf_synthetic_workload() -> Workload:
    """HF-shaped synthetic workload: tiny model that returns ModelOutput-like dict.

    Avoids dragging in 66M DistilBERT params just for a smoke test.
    """
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
            def __len__(self):
                return n

            def __getitem__(self, i):
                return {
                    "input_ids": torch.randint(0, 100, (seq_len,)),
                    "attention_mask": torch.ones(seq_len, dtype=torch.long),
                    "labels": torch.randint(0, 2, ()),
                }

        return torch.utils.data.DataLoader(_DS(), batch_size=4, collate_fn=collate)

    def eval_fn(model, loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                out = model(**batch)
                correct += int((out["logits"].argmax(dim=-1) == batch["labels"]).sum())
                total += int(batch["labels"].numel())
        return {"val_acc": correct / max(total, 1)}

    return Workload(
        name="hf-synthetic-tiny",
        tier="smoke",
        make_model=make_model,
        make_train_loader=make_loader,
        make_val_loader=make_loader,
        eval_fn=eval_fn,
        epochs=1,
    )


def test_baseline_runner_hf_trainer_completes():
    """HF Trainer baseline runs end-to-end on an HF-shaped synthetic workload."""
    pytest.importorskip("transformers")
    wl = _make_hf_synthetic_workload()
    runner = BaselineRunner(framework="hf-trainer")
    report = runner.run(wl)

    assert report.workload == "hf-synthetic-tiny"
    assert report.framework == "hf-trainer"
    assert report.elapsed_secs > 0
    assert report.samples_per_sec > 0
    assert "val_acc" in report.final_metrics


def test_baseline_runner_hf_trainer_rejects_non_dataloader():
    """HF Trainer path requires loaders to be DataLoaders (we read .dataset off them)."""
    pytest.importorskip("transformers")
    wl = _make_synthetic_workload()  # generic (x,y) tuple workload
    # Strip the .dataset attribute by replacing make_train_loader with one that
    # returns a plain iterable — Trainer needs a real Dataset.
    bad = Workload(
        name="bad-hf", tier="smoke",
        make_model=wl.make_model,
        make_train_loader=lambda: iter([]),
        make_val_loader=wl.make_val_loader,
        eval_fn=wl.eval_fn,
        epochs=1,
    )
    runner = BaselineRunner(framework="hf-trainer")
    with pytest.raises(ValueError, match=r"DataLoader"):
        runner.run(bad)
