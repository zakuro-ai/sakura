import argparse
import os
import time

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST



class MNISTModel(L.LightningModule):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def validation_step(self, batch, batch_nb):
        with torch.no_grad():
            x, y = batch
            loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def make_loaders(path, batch_size):
    train_ds = MNIST(path, train=True, download=True, transform=transforms.ToTensor())
    val_ds = MNIST(path, train=False, download=True, transform=transforms.ToTensor())
    return (
        DataLoader(train_ds, batch_size=batch_size),
        DataLoader(val_ds, batch_size=batch_size),
    )


def run_baseline(train_loader, val_loader, epochs):
    model = MNISTModel()
    trainer = L.Trainer(
        accelerator="auto", max_epochs=epochs, default_root_dir="/tmp"
    )
    t0 = time.perf_counter()
    trainer.fit(model, train_loader, val_loader)
    return time.perf_counter() - t0


def run_sakura(train_loader, val_loader, epochs, val_compute=None):
    from sakura.lightning import SakuraTrainer

    model = MNISTModel()
    trainer = SakuraTrainer(
        max_epochs=epochs,
        accelerator="auto",
        default_root_dir="/tmp",
        val_compute=val_compute,            # None → Zakuro standalone fallback
        model_factory=MNISTModel,           # rebuilt on the worker
        val_loader_factory=lambda: val_loader,
        model_path="models/best_model.pth",
    )
    t0 = time.perf_counter()
    trainer.run(model, train_loader, val_loader)
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description="Benchmark Sakura vs Lightning")
    parser.add_argument(
        "--mode",
        choices=["baseline", "sakura", "both"],
        default="both",
        help="Which trainer(s) to benchmark (default: both)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-dir", default=os.environ.get("PATH_DATASETS", "/tmp/datasets"))
    parser.add_argument(
        "--val-worker",
        default=os.environ.get("SAKURA_VAL_WORKER"),
        help="Zakuro URI of the validation worker (e.g. quic://host:4433). "
             "If unset, Sakura uses Zakuro's standalone fallback.",
    )
    args = parser.parse_args()

    batch_size = args.batch_size or (2000 if torch.cuda.is_available() else 64)
    train_loader, val_loader = make_loaders(args.data_dir, batch_size)

    results = {}

    if args.mode in ("baseline", "both"):
        print(f"--- Baseline Lightning Trainer ({args.epochs} epochs) ---")
        results["baseline"] = run_baseline(train_loader, val_loader, args.epochs)
        print(f"Baseline: {results['baseline']:.2f}s")

    if args.mode in ("sakura", "both"):
        print(f"--- Sakura Trainer ({args.epochs} epochs) ---")
        val_compute = None
        if args.val_worker:
            import zakuro as zk
            val_compute = zk.Compute(uri=args.val_worker)
            print(f"    validation → {val_compute.uri}")
        else:
            print("    validation → zakuro standalone (in-process)")
        results["sakura"] = run_sakura(
            train_loader, val_loader, args.epochs, val_compute=val_compute
        )
        print(f"Sakura:   {results['sakura']:.2f}s")

    if "baseline" in results and "sakura" in results:
        diff = results["baseline"] - results["sakura"]
        pct = (diff / results["baseline"]) * 100
        faster = "sakura" if diff > 0 else "baseline"
        print(f"\n{'='*40}")
        print(f"Baseline: {results['baseline']:.2f}s")
        print(f"Sakura:   {results['sakura']:.2f}s")
        print(f"{faster} is {abs(pct):.1f}% faster ({abs(diff):.2f}s)")


if __name__ == "__main__":
    main()
