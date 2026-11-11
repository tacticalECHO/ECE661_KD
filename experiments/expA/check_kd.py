
import argparse

import os
import sys
from pathlib import Path

# Shared code (models/, losses.py, utils.py) lives at the repo root; relative paths
# (results/, checkpoints/, logs/) resolve inside experiments/expA from any working directory.
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))
os.chdir(EXP_DIR)

import torch

from models.mnist import TeacherCNN, StudentCNN
from prepare_data import build_loaders
from losses import kd_loss


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, default="checkpoints/teacher_final.pt")
    args = parser.parse_args()

    device = torch.device("cuda")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # 1. Load real MNIST data
    train_loader, _, _ = build_loaders(
        batch_size=128,
        seed=42,
        augment=False,
        exclude_classes=None
    )

    images, labels = next(iter(train_loader))

    images = images.to(device)
    labels = labels.to(device)

    # 2. Load trained teacher
    teacher = TeacherCNN().to(device)

    checkpoint = torch.load(
        args.teacher,
        map_location=device,
        weights_only=True
    )

    teacher.load_state_dict(
        checkpoint["model_state_dict"]
    )

    teacher.eval()
    teacher.requires_grad_(False)

    # 3. Generate teacher logits once
    with torch.no_grad():
        teacher_logits = teacher(images)

    assert torch.isfinite(teacher_logits).all().item()

    print("Teacher loaded successfully!")
    print("Teacher logits shape:", teacher_logits.shape)

    print(
        "Teacher logit range:",
        teacher_logits.min().item(),
        teacher_logits.max().item()
    )

    # 4. Test different temperatures
    temperatures = [1, 2, 4, 8, 20]

    for T in temperatures:

        # Same initial student for every temperature
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        student = StudentCNN().to(device)
        student.train()

        optimizer = torch.optim.Adam(
            student.parameters(),
            lr=0.005
        )

        optimizer.zero_grad()

        student_logits = student(images)

        # KD loss
        loss, ce, soft = kd_loss(
            student_logits,
            teacher_logits,
            labels,
            T=T,
            alpha=0.9
        )

        assert torch.isfinite(loss).item(), \
            f"Non-finite loss at T={T}"

        # Backward
        loss.backward()

        # Check all gradients
        for name, p in student.named_parameters():

            if p.grad is not None:

                assert torch.isfinite(p.grad).all().item(), \
                    f"Invalid gradient: {name}, T={T}"

        optimizer.step()
        torch.cuda.synchronize()

        print(
            f"T={T:2d} | "
            f"CE={ce.item():.4f} | "
            f"Soft={soft.item():.4f} | "
            f"Total={loss.item():.4f} | "
            f"PASS",
            flush=True
        )

    print("\nAll KD temperature checks passed!")


if __name__ == "__main__":
    main()
