"""Logit distillation, shared by the peft backend (causal LMs) and the Hugging Face classifiers (SAK-0019).

The student's loss is ``alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * hard``, where the two
distributions are softmaxes of the logits divided by the temperature ``T`` and ``hard`` is the usual loss
on the labels (cross-entropy on the class, or on the next token). ``T^2`` keeps the size of the soft term's
gradient independent of the temperature (Hinton et al., 2015).

Where the teacher comes from and the checks on the three keys are ``sakura.atelier.recipe`` (no torch);
this module is the torch half and is imported only inside a backend's ``train``.
"""

from __future__ import annotations

from typing import Any


def kd_loss(student_logits: Any, teacher_logits: Any, temperature: float) -> Any:
    """``T^2 * KL(softmax(teacher / T) || softmax(student / T))``, averaged over the rows (the last
    dimension is the class or the vocabulary)."""
    import torch.nn.functional as F

    t = float(temperature)
    student = F.log_softmax(student_logits.float() / t, dim=-1)
    teacher = F.softmax(teacher_logits.float() / t, dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean") * (t * t)


def blend(hard: Any, soft: Any, alpha: float) -> Any:
    """The distillation loss: ``alpha`` of the soft (teacher) term, the rest of the hard (label) term."""
    return float(alpha) * soft + (1.0 - float(alpha)) * hard


def freeze(model: Any) -> Any:
    """A teacher never learns: evaluation mode, no gradients."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def label_permutation(teacher_labels: list[str], labels: list[str], ref: str) -> list[int]:
    """The teacher's logit column for each of the student's ``labels``, in the student's order.

    The two label spaces must be the same set of names (the order may differ); anything else is an error
    that names what is missing on each side, because a teacher's probability for a class the student does
    not have cannot be distilled."""
    if set(teacher_labels) != set(labels) or len(teacher_labels) != len(labels):
        only_teacher = sorted(set(teacher_labels) - set(labels))
        only_student = sorted(set(labels) - set(teacher_labels))
        raise ValueError(
            f"teacher {ref!r} does not have the student's label space: the teacher classifies "
            f"{sorted(teacher_labels)} and the data has {sorted(labels)} (only in the teacher: {only_teacher}; "
            f"only in the data: {only_student}). Distillation needs the same classes on both sides.")
    return [teacher_labels.index(label) for label in labels]


def check_same_vocabulary(student_tokenizer: Any, teacher_tokenizer: Any, ref: str) -> None:
    """Token-level distillation compares two distributions over the same tokens: the two tokenizers must
    map the same strings to the same ids."""
    student, teacher = student_tokenizer.get_vocab(), teacher_tokenizer.get_vocab()
    if student == teacher:
        return
    differing = sum(1 for tok, i in student.items() if teacher.get(tok) != i)
    raise ValueError(
        f"teacher {ref!r} does not share the student's vocabulary: the student's tokenizer has {len(student)} "
        f"tokens, the teacher's {len(teacher)}, and {differing} of the student's tokens are missing from the "
        "teacher or have another id. Logit distillation needs the same tokenizer on both sides.")
