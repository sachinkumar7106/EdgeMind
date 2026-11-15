
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()

        self.temperature = temperature
        self.alpha = alpha

        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        # Loss from the true dataset labels
        loss_ce = self.ce_loss(student_logits, labels)

        # Softened teacher and student predictions
        teacher_probs = F.softmax(
            teacher_logits / self.temperature, dim=1
        )

        student_log_probs = F.log_softmax(
            student_logits / self.temperature, dim=1
        )

        # Knowledge-distillation loss
        loss_kd = self.kl_loss(
            student_log_probs, teacher_probs
        ) * (self.temperature ** 2)

        # Combined loss
        total_loss = (
            (1 - self.alpha) * loss_ce
            + self.alpha * loss_kd
        )

        return total_loss
