"""The decision task on a causal LM: option-letter readout, loss, training loop, shared-prefix scoring.

Imported by the peft backend inside its methods, never at import time: this module needs torch (every
backend's stack is installed only in its own venv), but it asks nothing of ``transformers`` or ``peft``,
only of the model and tokenizer it is handed, so the unit tests drive it with a tiny fake.

The answer is the logit of an option letter at the answer slot, restricted to the K letters of that
question (not the whole vocabulary); training is the cross-entropy of that restricted softmax against
the target (one-hot for a hard label, the given distribution for a soft one). The loss sits on the
answer slot alone and no EOS is appended: the sequence simply ends at ``Answer:``.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, replace
from typing import Any, Callable

import torch

from sakura.atelier.decision import (
    LETTERS,
    DecisionDataError,
    Instance,
    Scored,
    build_prompt,
    identity_order,
    render_state,
    shuffled_order,
    split_metrics,
    to_letter_order,
)

ANSWER_SLOT = "Answer:"


@dataclass(frozen=True)
class Readout:
    """What the model needs to read an answer."""

    #: Token id of ``" A"`` ... ``" Z"``: the one token after ``Answer:`` that names an option.
    letters: list[int]
    #: Ids every sequence starts with (a BOS, for tokenizers that add one; none for Qwen).
    start: list[int]
    pad_id: int


@dataclass
class Encoded:
    prefix: list[int]  # start + the state section
    suffix: list[int]  # the question section, ending at the answer slot
    k: int
    truncated: bool = False


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    max_len: int
    seed: int


def make_readout(tokenizer: Any) -> Readout:
    """Check that the tokenizer gives each option letter its own single token, and that a letter after
    ``Answer:`` is that token in running text too, then return the ids.

    A base whose tokenizer fails this cannot be read by the letter logits; that is a platform error
    (a bad base choice), not a customer-data one."""
    ids: list[int] = []
    for letter in LETTERS:
        enc = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(enc) != 1:
            raise RuntimeError(f"option letter {' ' + letter!r} is not a single token for this tokenizer "
                               f"(got {list(enc)}); the decision readout needs one token per letter")
        ids.append(int(enc[0]))
    if len(set(ids)) != len(ids):
        raise RuntimeError("two option letters share a token id for this tokenizer; the decision readout needs "
                           "a distinct token per letter")
    slot = list(tokenizer.encode(ANSWER_SLOT, add_special_tokens=False))
    if list(tokenizer.encode(f"{ANSWER_SLOT} A", add_special_tokens=False)) != slot + [ids[0]]:
        raise RuntimeError(f"{ANSWER_SLOT!r} followed by ' A' does not tokenise as {ANSWER_SLOT!r} plus the letter "
                           "token for this tokenizer, so the answer slot is not where the model reads it")
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    return Readout(letters=ids, start=[int(i) for i in tokenizer.encode("", add_special_tokens=True)],
                   pad_id=int(pad) if pad is not None else 0)


def encode_instance(tokenizer: Any, readout: Readout, instance: Instance, order: list[int], max_len: int,
                    suffix_floor: int = 0) -> Encoded:
    """Token ids of one (state, question) in the given option order, within ``max_len``.

    A state too long for the budget is cut (as much of it as fits); the question and its options never are.
    ``suffix_floor`` reserves room for a longer question than this one, so every question about one state
    cuts it at the same place and can share its prefix."""
    prefix_text, suffix_text = build_prompt(instance, order)
    suffix = [int(i) for i in tokenizer.encode(suffix_text, add_special_tokens=False)]
    room = max_len - max(len(suffix), suffix_floor)

    def prefix_ids(state_text: str) -> list[int]:
        text = build_prompt(replace(instance, state=state_text), order)[0]
        return readout.start + [int(i) for i in tokenizer.encode(text, add_special_tokens=False)]

    prefix = readout.start + [int(i) for i in tokenizer.encode(prefix_text, add_special_tokens=False)]
    if len(prefix) <= room:
        return Encoded(prefix, suffix, len(order))
    state_text = render_state(instance.state)
    if len(prefix_ids("")) > room:
        raise DecisionDataError(f"question {instance.qid!r} with its options takes {len(suffix)} tokens; "
                                f"max_seq_length is {max_len}")
    lo, hi = 0, len(state_text)  # the most characters of the state that still fit
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(prefix_ids(state_text[:mid])) <= room:
            lo = mid
        else:
            hi = mid - 1
    return Encoded(prefix_ids(state_text[:lo]), suffix, len(order), truncated=True)


def _letter_logits(model: Any, readout: Readout, seqs: list[list[int]], ks: list[int], device: Any) -> list[Any]:
    """The K option-letter logits at the last position of each sequence (left-padded into one batch)."""
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), readout.pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for row, s in enumerate(seqs):
        ids[row, width - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[row, width - len(s):] = 1
    kwargs = {"input_ids": ids.to(device), "attention_mask": mask.to(device)}
    try:
        out = model(**kwargs, logits_to_keep=1)
    except TypeError:
        out = model(**kwargs)
    last = out.logits[:, -1, :].float()
    letters = torch.tensor(readout.letters, dtype=torch.long, device=last.device)
    return [last[row, letters[:k]] for row, k in enumerate(ks)]


def batch_loss(model: Any, tokenizer: Any, readout: Readout, batch: list[tuple[Instance, list[int]]],
               max_len: int, device: Any) -> tuple[Any, int]:
    """``(mean cross-entropy of the restricted K-letter softmax, number of states cut to fit)`` over
    ``batch`` of ``(instance, option order)``. The target is re-indexed to the order shown."""
    encoded = [encode_instance(tokenizer, readout, inst, order, max_len) for inst, order in batch]
    vectors = _letter_logits(model, readout, [e.prefix + e.suffix for e in encoded], [e.k for e in encoded], device)
    losses = []
    for vec, (inst, order) in zip(vectors, batch):
        logp = torch.log_softmax(vec, dim=-1)
        target = torch.tensor(to_letter_order(inst.target, order), dtype=logp.dtype, device=logp.device)
        losses.append(-(target * logp).sum())
    return torch.stack(losses).mean(), sum(e.truncated for e in encoded)


def score_instances(model: Any, tokenizer: Any, readout: Readout, instances: list[Instance], max_len: int,
                    device: Any, batch_size: int = 8) -> list[Scored]:
    """Each instance's restricted logits in the option order GIVEN (never shuffled), in the input order."""
    if not instances:
        return []
    encoded = [encode_instance(tokenizer, readout, i, identity_order(len(i.labels)), max_len) for i in instances]
    by_length = sorted(range(len(instances)), key=lambda j: len(encoded[j].prefix) + len(encoded[j].suffix))
    logits: dict[int, list[float]] = {}
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(by_length), batch_size):
                chunk = by_length[start:start + batch_size]
                vectors = _letter_logits(model, readout, [encoded[j].prefix + encoded[j].suffix for j in chunk],
                                         [encoded[j].k for j in chunk], device)
                for j, vec in zip(chunk, vectors):
                    logits[j] = [float(x) for x in vec.cpu().tolist()]
    finally:
        model.train(was_training)
    return [Scored(qid=i.qid, kind=i.kind, logits=logits[j], target=list(i.target)) for j, i in enumerate(instances)]


def score_state(model: Any, tokenizer: Any, readout: Readout, state: Any, instances: list[Instance],
                max_len: int, device: Any) -> list[list[float]]:
    """The restricted logits of every question about ONE ``state``, the state prefix run once.

    With a single question this is one forward pass. With several, the state section goes through the model
    once (``use_cache``) and each question then runs from a copy of that cache, so a second question costs
    the length of the question, not of the state."""
    suffix_floor = max(len(tokenizer.encode(build_prompt(i, identity_order(len(i.labels)))[1], add_special_tokens=False))
                       for i in instances)
    encoded = [encode_instance(tokenizer, readout, i, identity_order(len(i.labels)), max_len, suffix_floor)
               for i in instances]
    was_training = bool(getattr(model, "training", False))
    model.eval()
    out: list[list[float]] = []
    try:
        with torch.no_grad():
            letters = torch.tensor(readout.letters, dtype=torch.long)
            if len(encoded) == 1 or any(e.prefix != encoded[0].prefix for e in encoded):
                for e in encoded:
                    vec = _letter_logits(model, readout, [e.prefix + e.suffix], [e.k], device)[0]
                    out.append([float(x) for x in vec.cpu().tolist()])
                return out
            first = model(input_ids=torch.tensor([encoded[0].prefix], dtype=torch.long, device=device), use_cache=True)
            for e in encoded:
                cache = copy.deepcopy(first.past_key_values)
                step = model(input_ids=torch.tensor([e.suffix], dtype=torch.long, device=device),
                             past_key_values=cache, use_cache=True)
                last = step.logits[0, -1, :].float()
                out.append([float(x) for x in last[letters[:e.k].to(last.device)].cpu().tolist()])
    finally:
        model.train(was_training)
    return out


def count_truncated(tokenizer: Any, readout: Readout, instances: list[Instance], max_len: int) -> int:
    """How many of ``instances`` have a state longer than ``max_len`` allows, cut at inference as in training."""
    return sum(encode_instance(tokenizer, readout, i, identity_order(len(i.labels)), max_len).truncated
               for i in instances)


def train_loop(model: Any, tokenizer: Any, readout: Readout, train: list[Instance], calibration: list[Instance], *,
               cfg: TrainConfig, device: Any, optimizer: Any, metrics: Any,
               order_fn: Callable[[random.Random, int], list[int]] = shuffled_order,
               watch: Any = None) -> dict[str, Any]:
    """Train on ``train`` for ``cfg.epochs``, reshuffling both the example order and each example's option
    order every epoch. After each epoch the log loss and accuracy on ``calibration`` (uncalibrated, options
    in the given order) go to ``metrics`` as validation points; the held-out EVAL split is not touched here.

    Returns the last epoch's ``loss`` (train), ``log_loss`` and ``accuracy`` (calibration) and how many
    training examples had their state cut to fit ``cfg.max_len``. ``watch`` (a ``WrapUpWatch``) is asked before
    every batch: once it says stop, learning ends there, the half-accumulated step is dropped, and the result
    carries ``stopped_early``."""
    rng = random.Random(cfg.seed)
    model.train()
    step = 0
    last_loss = 0.0
    last_val: dict[str, Any] = {}
    truncated = 0
    for epoch in range(cfg.epochs):
        order = list(range(len(train)))
        rng.shuffle(order)
        batches = [order[i:i + cfg.batch_size] for i in range(0, len(order), cfg.batch_size)]
        truncated = 0
        for bi, idx in enumerate(batches):
            if watch is not None and watch.stop(epoch + bi * cfg.batch_size / len(order), step):
                optimizer.zero_grad()
                break
            batch = [(train[j], order_fn(rng, len(train[j].labels))) for j in idx]
            loss, cut = batch_loss(model, tokenizer, readout, batch, cfg.max_len, device)
            truncated += cut
            (loss / cfg.grad_accum).backward()
            boundary = (bi + 1) % cfg.grad_accum == 0 or bi == len(batches) - 1
            if boundary:
                optimizer.step()
                optimizer.zero_grad()
            last_loss = float(loss.detach())
            if metrics is not None:
                metrics.log("loss", last_loss, step=step, split="train", epoch=epoch + bi * cfg.batch_size / len(order))
            if boundary:
                step += 1
        if watch is not None and watch.triggered:
            break
        if calibration:
            last_val = split_metrics(score_instances(model, tokenizer, readout, calibration, cfg.max_len, device,
                                                     cfg.batch_size), {})
            if metrics is not None:
                metrics.log("log_loss", last_val["log_loss"], step=step, split="validation", epoch=float(epoch + 1))
                metrics.log("accuracy", last_val["accuracy"], step=step, split="validation", epoch=float(epoch + 1))
    return {"loss": last_loss, "log_loss": last_val.get("log_loss"), "accuracy": last_val.get("accuracy"),
            "truncated": truncated, "stopped_early": None if watch is None else watch.stopped_early}
