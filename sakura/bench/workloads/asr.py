"""ASR (DeepSpeech2-lite, CTC) bench workload.

A self-contained, character-level speech recognizer — Conv2d feature stack →
bidirectional GRU → linear → log-softmax — trained with ``CTCLoss`` on synthetic
spectrograms.

Why this workload exists
------------------------
The other bench workloads are classification (``model(x) -> logits`` +
cross-entropy). ASR grounds Sakura's async-eval in a *different* training shape:

  * the objective is **CTC**, not classification, and the batch is not an
    ``(x, y)`` tuple — it exercises the harness ``loss_fn`` hook;
  * the eval phase is **decode-heavy** (greedy CTC decode + character-error-rate
    over the whole val set), i.e. a non-trivial, largely Python/CPU cost — which
    is exactly the kind of per-epoch eval that overlapping it with the next
    epoch's training is meant to hide.

Data is synthetic (no dataset download) so the workload is hermetic and CI-safe;
sizes are env-tunable (``SAKURA_ASR_*``) so the perf run can be scaled to balance
train (GPU) against eval (CPU) on the target box.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sakura.bench.harness import Workload

N_CLASSES = 29  # 26 letters + space + apostrophe + CTC blank (index 0)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


N_FEATS = _env_int("SAKURA_ASR_FEATS", 64)
SEQ_T = _env_int("SAKURA_ASR_T", 256)
RNN_DIM = _env_int("SAKURA_ASR_RNN_DIM", 256)
N_RNN = _env_int("SAKURA_ASR_RNN_LAYERS", 2)
BATCH = _env_int("SAKURA_ASR_BATCH", 16)
N_TRAIN_BATCHES = _env_int("SAKURA_ASR_TRAIN_BATCHES", 20)
N_VAL_BATCHES = _env_int("SAKURA_ASR_VAL_BATCHES", 8)
EPOCHS = _env_int("SAKURA_ASR_EPOCHS", 8)


class DeepSpeech2Lite(nn.Module):
    """Conv2d feature stack → BiGRU → linear classifier over characters."""

    def __init__(self, n_feats: int = N_FEATS, n_classes: int = N_CLASSES,
                 rnn_dim: int = RNN_DIM, n_rnn: int = N_RNN) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(11, 11), stride=(2, 2), padding=(5, 5)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(11, 11), stride=(1, 1), padding=(5, 5)),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        # one stride-2 conv halves the frequency axis: feat = 32 channels * n_feats/2
        feat = 32 * (n_feats // 2)
        self.rnn = nn.GRU(
            feat, rnn_dim, num_layers=n_rnn, batch_first=True, bidirectional=True
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * rnn_dim),
            nn.Linear(2 * rnn_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, n_feats, T)
        x = self.conv(x)  # (B, 32, n_feats/2, T/2)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)  # (B, T', C*F)
        x, _ = self.rnn(x)  # (B, T', 2*rnn_dim)
        x = self.head(x)  # (B, T', n_classes)
        return x.log_softmax(-1)


def make_model() -> DeepSpeech2Lite:
    return DeepSpeech2Lite()


def _make_batches(n_batches: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Synthetic batches: (spectrograms, concatenated targets, target_lengths).

    Returns a list so loader iteration is GIL-cheap (matters for the async-eval
    thread-overlap). The 3-tuple (not (x, y)) routes the eval bridge through the
    DataLoader fallback, i.e. through this workload's own ``eval_fn``.
    """
    g = torch.Generator().manual_seed(seed)
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for _ in range(n_batches):
        specs = torch.randn(BATCH, 1, N_FEATS, SEQ_T, generator=g)
        lens = torch.randint(8, 20, (BATCH,), generator=g)
        targets = torch.cat([
            torch.randint(1, N_CLASSES, (int(length),), generator=g) for length in lens
        ])
        batches.append((specs, targets, lens))
    return batches


def make_train_loader() -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return _make_batches(N_TRAIN_BATCHES, seed=1234)


def make_val_loader() -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return _make_batches(N_VAL_BATCHES, seed=4321)


def loss_fn(model: Any, batch: Any, device: Any) -> tuple[torch.Tensor, int]:
    """CTC training step. Returns (loss, batch_size)."""
    specs, targets, target_lengths = batch
    specs = specs.to(device)
    log_probs = model(specs).permute(1, 0, 2)  # (T', B, C) for CTC
    input_lengths = torch.full((specs.size(0),), log_probs.size(0), dtype=torch.long)
    # CTC wants targets/lengths on CPU even when log_probs is on CUDA.
    loss = F.ctc_loss(
        log_probs,
        targets.cpu(),
        input_lengths,
        target_lengths.cpu(),
        blank=0,
        zero_infinity=True,
    )
    return loss, specs.size(0)


def _greedy_decode(log_probs_row: torch.Tensor) -> list[int]:
    """Collapse repeats and drop the blank (0) from an argmax path."""
    ids = log_probs_row.argmax(-1).tolist()
    out: list[int] = []
    prev = -1
    for i in ids:
        if i != prev and i != 0:
            out.append(i)
        prev = i
    return out


def _cer(ref: list[int], hyp: list[int]) -> float:
    """Character error rate = edit_distance(ref, hyp) / len(ref)."""
    if not ref:
        return 1.0 if hyp else 0.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def eval_fn(model: Any, val_loader: Any) -> dict[str, float]:
    """Greedy-decode the val set and report mean CER + CTC val loss.

    Decode + edit-distance is deliberately Python/CPU-bound: this is the
    per-epoch cost that async-eval overlaps with the next epoch's training.
    """
    device = next(model.parameters()).device
    model.eval()
    total_cer, n_utts, loss_sum, n_batches = 0.0, 0, 0.0, 0
    with torch.no_grad():
        for specs, targets, target_lengths in val_loader:
            specs = specs.to(device)
            log_probs = model(specs)  # (B, T', C)
            lp = log_probs.permute(1, 0, 2)
            input_lengths = torch.full((specs.size(0),), lp.size(0), dtype=torch.long)
            loss_sum += float(
                F.ctc_loss(lp, targets.cpu(), input_lengths, target_lengths.cpu(),
                           blank=0, zero_infinity=True)
            )
            n_batches += 1
            # split the concatenated targets back into per-utterance references
            refs: list[list[int]] = []
            offset = 0
            tcpu = targets.cpu()
            for length in target_lengths.cpu().tolist():
                refs.append(tcpu[offset:offset + length].tolist())
                offset += length
            log_probs_cpu = log_probs.cpu()
            for b in range(specs.size(0)):
                total_cer += _cer(refs[b], _greedy_decode(log_probs_cpu[b]))
                n_utts += 1
    return {
        "val_cer": total_cer / max(n_utts, 1),
        "val_loss": loss_sum / max(n_batches, 1),
    }


def make_workload() -> Workload:
    return Workload(
        name="asr-deepspeech",
        tier="perf",
        make_model=make_model,
        make_train_loader=make_train_loader,
        make_val_loader=make_val_loader,
        eval_fn=eval_fn,
        loss_fn=loss_fn,
        epochs=EPOCHS,
    )
