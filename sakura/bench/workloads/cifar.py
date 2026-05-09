"""CIFAR-10 + ResNet-50 workload (CI tier).

Smoke variant: 1 epoch with batch size 64 on a small subset (256 train,
64 val), runs in ~30s on CPU. Real CIFAR-10 via torchvision when network
is available; synthetic 3×32×32 random tensors otherwise.
"""
from __future__ import annotations

import os
import tempfile

import torch

from sakura.bench.harness import Workload


def _make_model(num_classes: int = 10) -> torch.nn.Module:
    """ResNet-50 from torchvision (no pretrained weights for benchmarking)."""
    from torchvision import models
    m = models.resnet50(weights=None)
    # Replace final classification head for CIFAR-10's 10 classes.
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m


def _make_loaders(batch_size: int = 64, n_train: int = 256, n_val: int = 64):
    try:
        from torchvision import datasets, transforms
        cache_dir = os.path.join(tempfile.gettempdir(), "sakura-cifar-cache")
        os.makedirs(cache_dir, exist_ok=True)
        # ResNet-50 expects roughly ImageNet-statistics normalization; for a
        # smoke benchmark the exact normalization doesn't matter, but we
        # apply a sensible default so the model doesn't see raw [0,1] floats.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train = datasets.CIFAR10(cache_dir, train=True, download=True, transform=transform)
        val = datasets.CIFAR10(cache_dir, train=False, download=True, transform=transform)
        train = torch.utils.data.Subset(train, list(range(min(n_train, len(train)))))
        val = torch.utils.data.Subset(val, list(range(min(n_val, len(val)))))
        return (
            torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
            torch.utils.data.DataLoader(val, batch_size=batch_size),
        )
    except Exception:
        torch.manual_seed(0)
        train_imgs = torch.randn(n_train, 3, 32, 32)
        train_lbls = torch.randint(0, 10, (n_train,))
        val_imgs = torch.randn(n_val, 3, 32, 32)
        val_lbls = torch.randint(0, 10, (n_val,))
        return (
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_imgs, train_lbls),
                batch_size=batch_size, shuffle=True,
            ),
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(val_imgs, val_lbls),
                batch_size=batch_size,
            ),
        )


def _eval_fn(model: torch.nn.Module, loader) -> dict:
    model.eval()
    device = next(model.parameters()).device
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            if hasattr(x, "to"):
                x = x.to(device)
            if hasattr(y, "to"):
                y = y.to(device)
            logits = model(x)
            loss_sum += float(torch.nn.functional.cross_entropy(logits, y, reduction="sum"))
            correct += int((logits.argmax(dim=-1) == y).sum())
            total += int(y.numel())
    return {
        "val_loss": loss_sum / max(total, 1),
        "val_acc": correct / max(total, 1),
    }


def make_workload(*, batch_size: int = 64, epochs: int = 1, n_train: int = 256,
                   n_val: int = 64) -> Workload:
    """Build the CIFAR-10 + ResNet-50 workload. Default is fast smoke shape."""
    train_loader, val_loader = _make_loaders(
        batch_size=batch_size, n_train=n_train, n_val=n_val,
    )
    return Workload(
        name="cifar10-resnet50",
        tier="ci",
        make_model=_make_model,
        make_train_loader=lambda: train_loader,
        make_val_loader=lambda: val_loader,
        eval_fn=_eval_fn,
        epochs=epochs,
    )


__all__ = ["make_workload"]
