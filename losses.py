import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def kd_loss(
        student_logits, 
        teacher_logits, 
        labels, 
        T=20.0,
        alpha=0.9):
    if  math.isinf(T):
        raise ValueError("Temperature must not be infinite")
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
    return total_loss,ce_loss, soft_loss