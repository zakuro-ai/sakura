"""Direct Preference Optimization (Rafailov et al., 2023) for the peft backend (SAK-0019), in plain torch.

A row is ``{prompt, chosen, rejected}``. With ``pi`` the policy being trained and ``ref`` the frozen
reference (the starting weights), and ``log p(y | x)`` the sum of the token log-probabilities of a
completion ``y`` (its end-of-sequence token included) after the prompt ``x``::

    loss = -log sigmoid(beta * ((log pi(chosen|x) - log pi(rejected|x)) - (log ref(chosen|x) - log ref(rejected|x))))

The reference never changes, so its two log-probabilities per row are computed once, before the LoRA is
attached, and the reference model is not kept in memory beside the policy.

This module is the torch half and is imported only inside ``PEFTBackend.train``.
"""

from __future__ import annotations

import math
from typing import Any

#: The DPO loss of a policy that equals its reference: ``-log sigmoid(0)``.
LOSS_AT_REFERENCE = math.log(2.0)


def encode_pair(tokenizer: Any, prompt: str, completion: str, max_len: int) -> tuple[list[int], int]:
    """``(token ids of prompt + completion + eos, number of prompt tokens)``, cut to ``max_len``."""
    ids = tokenizer(prompt + completion + tokenizer.eos_token, truncation=True, max_length=max_len)["input_ids"]
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"])
    return list(ids), min(prompt_len, len(ids))


def completion_logprob(model: Any, tokenizer: Any, encoded: list[tuple[list[int], int]], device: str) -> Any:
    """``log p(completion | prompt)`` for each encoded pair: the sum of the log-probabilities of the tokens
    after the prompt. One tensor of ``len(encoded)`` values; gradients flow when the model is training."""
    import torch

    width = max(len(ids) for ids, _ in encoded)
    pad = int(tokenizer.pad_token_id)
    input_ids = torch.full((len(encoded), width), pad, dtype=torch.long)
    attention = torch.zeros((len(encoded), width), dtype=torch.long)
    scored = torch.zeros((len(encoded), width), dtype=torch.bool)
    for row, (ids, prompt_len) in enumerate(encoded):
        input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[row, :len(ids)] = 1
        scored[row, prompt_len:len(ids)] = True  # the completion's tokens, eos included
    input_ids, attention, scored = input_ids.to(device), attention.to(device), scored.to(device)
    logits = model(input_ids=input_ids, attention_mask=attention).logits
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)  # position i predicts token i + 1
    token = logp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (token * scored[:, 1:]).sum(dim=-1)


def dpo_loss(policy_chosen: Any, policy_rejected: Any, reference_chosen: Any, reference_rejected: Any,
             beta: float) -> Any:
    """The mean DPO loss of a batch (the four arguments are log-probabilities, one value per row)."""
    import torch.nn.functional as F

    margin = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    return -F.logsigmoid(float(beta) * margin).mean()


def preference_stats(chosen: list[float], rejected: list[float], reference_chosen: list[float],
                     reference_rejected: list[float], beta: float) -> dict[str, float]:
    """What is reported for a model on a set of pairs: its DPO loss against the reference, the fraction of
    pairs where it gives ``chosen`` a higher log-probability than ``rejected`` (margin > 0), and the mean
    margin."""
    n = len(chosen)
    if n == 0:
        return {"loss": 0.0, "preference_accuracy": 0.0, "mean_margin": 0.0}
    margins = [c - r for c, r in zip(chosen, rejected)]
    shifted = [m - (rc - rr) for m, rc, rr in zip(margins, reference_chosen, reference_rejected)]
    # -log sigmoid(z) = log(1 + exp(-z)), computed without overflow
    loss = sum(max(-beta * z, 0.0) + math.log1p(math.exp(-abs(beta * z))) for z in shifted) / n
    return {"loss": loss, "preference_accuracy": sum(m > 0 for m in margins) / n, "mean_margin": sum(margins) / n}
