import math

import torch
import torch.nn.functional as F


def _validate_temperature(temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")


def _validate_logits(student_logits, teacher_logits, labels):
    if student_logits.ndim != 2 or teacher_logits.ndim != 2:
        raise ValueError("Student and teacher logits must be rank-two tensors")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have the same shape")
    if labels.ndim != 1 or labels.shape[0] != student_logits.shape[0]:
        raise ValueError("Labels must contain one class index per sample")
    if student_logits.shape[1] < 2:
        raise ValueError("At least two classes are required")
    if labels.numel() and (
        labels.min().item() < 0 or labels.max().item() >= student_logits.shape[1]
    ):
        raise ValueError("Labels contain an out-of-range class index")


def target_mask(logits, labels):
    """Return a Boolean mask selecting exactly one target class per sample."""
    if logits.ndim != 2:
        raise ValueError("Logits must be a rank-two tensor")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("Labels must contain one class index per sample")
    mask = torch.zeros_like(logits, dtype=torch.bool)
    return mask.scatter_(1, labels.reshape(-1, 1), True)


def _two_bin_log_probs(logits, mask, temperature):
    """Log probabilities for the target and aggregate non-target bins."""
    scaled = logits / temperature
    normalizer = torch.logsumexp(scaled, dim=1)
    target = scaled.masked_fill(~mask, -torch.inf).logsumexp(dim=1)
    non_target = scaled.masked_fill(mask, -torch.inf).logsumexp(dim=1)
    return torch.stack((target - normalizer, non_target - normalizer), dim=1)


def _non_target_log_probs(logits, mask, temperature):
    scaled = (logits / temperature).masked_fill(mask, -torch.inf)
    return F.log_softmax(scaled, dim=1)

def kd_loss(
        student_logits, 
        teacher_logits, 
        labels, 
        T=20.0,
        alpha=0.9):
    _validate_temperature(T)
    _validate_logits(student_logits, teacher_logits, labels)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Alpha must be between 0 and 1")
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.detach().float()
    ce_loss = F.cross_entropy(student_logits, labels)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    soft_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    ) * (T * T)
    total_loss = (1 - alpha) * ce_loss + alpha * soft_loss
    return total_loss, ce_loss, soft_loss


def tckd_loss(student_logits, teacher_logits, labels, temperature=4.0):
    """Target-class KD from the two-bin target/non-target distributions."""
    _validate_temperature(temperature)
    _validate_logits(student_logits, teacher_logits, labels)
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.detach().float()
    mask = target_mask(student_logits, labels)
    student_log_probs = _two_bin_log_probs(student_logits, mask, temperature)
    teacher_probs = _two_bin_log_probs(
        teacher_logits, mask, temperature
    ).exp()
    return F.kl_div(
        student_log_probs, teacher_probs, reduction="batchmean"
    ) * (temperature * temperature)


def nckd_loss(student_logits, teacher_logits, labels, temperature=4.0):
    """Non-target-class KD after removing the ground-truth class."""
    _validate_temperature(temperature)
    _validate_logits(student_logits, teacher_logits, labels)
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.detach().float()
    mask = target_mask(student_logits, labels)
    student_log_probs = _non_target_log_probs(
        student_logits, mask, temperature
    )
    teacher_probs = _non_target_log_probs(
        teacher_logits, mask, temperature
    ).exp()
    elementwise_kl = F.kl_div(
        student_log_probs, teacher_probs, reduction="none"
    ).masked_fill(mask, 0.0)
    return elementwise_kl.sum() * (temperature * temperature) / labels.shape[0]


def dkd_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature=4.0,
    alpha=1.0,
    beta=8.0,
):
    """Return weighted DKD loss and its unweighted TCKD/NCKD components."""
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("DKD alpha must be finite and non-negative")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("DKD beta must be finite and non-negative")
    tckd = tckd_loss(student_logits, teacher_logits, labels, temperature)
    nckd = nckd_loss(student_logits, teacher_logits, labels, temperature)
    return alpha * tckd + beta * nckd, tckd, nckd
