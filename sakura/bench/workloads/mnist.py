"""MNIST + tiny MLP workload (smoke tier).

Runs in <2 minutes on a 4090, <30 minutes on CPU. The simplest workload —
used for CI smoke testing of the runners themselves.
"""
from __future__ import annotations

import os
import tempfile

import torch

from sakura.bench.harness import Workload


def _make_model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28 * 28, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    )


def _make_loaders(batch_size: int = 64, n_train: int = 1024, n_val: int = 256):
    """Returns (train_loader, val_loader). Uses synthetic data if MNIST download
    is unavailable (offline CI). Real MNIST when torchvision can fetch it."""
    try:
        from torchvision import datasets, transforms
        # Cache in a tmpdir so concurrent tests don't race.
        cache_dir = os.path.join(tempfile.gettempdir(), "sakura-mnist-cache")
        os.makedirs(cache_dir, exist_ok=True)
        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(cache_dir, train=True, download=True, transform=transform)
        val = datasets.MNIST(cache_dir, train=False, download=True, transform=transform)
        # Subsample for speed (smoke = fast).
        train = torch.utils.data.Subset(train, list(range(min(n_train, len(train)))))
        val = torch.utils.data.Subset(val, list(range(min(n_val, len(val)))))
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size)
        return train_loader, val_loader
    except Exception:
        # Fallback: synthetic random tensors with the right shape.
        torch.manual_seed(0)
        train_imgs = torch.randn(n_train, 1, 28, 28)
        train_lbls = torch.randint(0, 10, (n_train,))
        val_imgs = torch.randn(n_val, 1, 28, 28)
        val_lbls = torch.randint(0, 10, (n_val,))
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_imgs, train_lbls),
            batch_size=batch_size, shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_imgs, val_lbls),
            batch_size=batch_size,
        )
        return train_loader, val_loader


def _eval_fn(model: torch.nn.Module, loader) -> dict:
    model.eval()
    device = next(model.parameters()).device
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss_sum += float(torch.nn.functional.cross_entropy(logits, y, reduction="sum"))
            correct += int((logits.argmax(dim=-1) == y).sum())
            total += int(y.numel())
    return {
        "val_loss": loss_sum / max(total, 1),
        "val_acc": correct / max(total, 1),
    }


def make_workload(*, batch_size: int = 64, epochs: int = 1) -> Workload:
    """Build the MNIST + tiny MLP workload. Defaults to a fast smoke shape."""
    train_loader, val_loader = _make_loaders(batch_size=batch_size)

    return Workload(
        name="mnist-mlp",
        tier="smoke",
        make_model=_make_model,
        make_train_loader=lambda: train_loader,
        make_val_loader=lambda: val_loader,
        eval_fn=_eval_fn,
        epochs=epochs,
    )


def _make_model_multi() -> torch.nn.Module:
    """Slightly bigger MLP for the AsyncEval overlap demo.

    Forward pass cost must be dominated by tensor matmul (GIL-released by
    torch's C++ kernels) rather than Python overhead, otherwise the
    background eval thread can't actually run in parallel with the main
    training thread. 256-hidden gets us there on CPU at modest size.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    )


def _make_synthetic_tensor_loaders(
    batch_size: int, n_train: int, n_val: int, n_features: int,
):
    """Plain torch.TensorDataset loaders — no PIL.

    The standard MNIST workload uses torchvision/PIL transforms which hold
    the GIL during decode; that defeats thread-based eval/train overlap.
    The multi-epoch overlap workload uses pure tensor data so the train
    and eval threads share only torch C++ kernels (which release the GIL).
    """
    torch.manual_seed(0)
    train_x = torch.randn(n_train, n_features)
    train_y = torch.randint(0, 10, (n_train,))
    val_x = torch.randn(n_val, n_features)
    val_y = torch.randint(0, 10, (n_val,))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=batch_size, shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_x, val_y),
        batch_size=batch_size,
    )
    return train_loader, val_loader


def _eval_fn_multi(model: torch.nn.Module, loader) -> dict:
    """Eval over (x, y) tuples on a tensor loader. Same semantics as
    `_eval_fn` but skips the .to(device) calls already handled by
    `_DeviceLoader`."""
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss_sum += float(torch.nn.functional.cross_entropy(logits, y, reduction="sum"))
            correct += int((logits.argmax(dim=-1) == y).sum())
            total += int(y.numel())
    return {
        "val_loss": loss_sum / max(total, 1),
        "val_acc": correct / max(total, 1),
    }


def make_workload_multi(
    *, batch_size: int = 64, epochs: int = 5, n_train: int = 2048, n_val: int = 16384,
    n_features: int = 128,
) -> Workload:
    """Multi-epoch synthetic workload where per-epoch eval cost ≈ train cost.

    Designed to expose the AsyncEval overlap win: with 5 epochs and an eval
    set 8× larger than train, per-epoch eval is the dominant blocking
    operation in a baseline loop. Sakura+AsyncEval+ThreadDispatcher overlaps
    eval(N) with train(N+1), saving most of the per-epoch eval time.

    Pure tensor data (no PIL) is intentional — PIL holds the GIL during
    decode and would prevent the eval thread from actually running in
    parallel with training. With tensor data, both threads share only
    torch's C++ kernels which release the GIL during compute.

    Default shape (2k train / 16k val × 5 epochs, 256-hidden MLP) reproduces
    the README's measured 1.26x AsyncEval speedup at OMP_NUM_THREADS=2.
    """
    train_loader, val_loader = _make_synthetic_tensor_loaders(
        batch_size=batch_size, n_train=n_train, n_val=n_val, n_features=n_features,
    )
    return Workload(
        name="mnist-mlp-multi",
        tier="smoke",
        make_model=_make_model_multi,
        make_train_loader=lambda: train_loader,
        make_val_loader=lambda: val_loader,
        eval_fn=_eval_fn_multi,
        epochs=epochs,
    )


__all__ = ["make_workload", "make_workload_multi"]
