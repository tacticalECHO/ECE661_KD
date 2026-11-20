# Adapted from experiments/expA/train_student_kd.py.
SEED = 456
SPLIT_SEED = 42
EPOCHS = 200
SCHEDULE_EPOCHS = 200
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 10
MIN_LR = 0.0
DEVICE = "cuda"
AUGMENT = True
STUDENT = "resnet50"
TEACHER = "checkpoints/teacher_resnet34_s42_aa89a5a55c45d775.pt"
TEMPERATURE = 4.0
ALPHA = 0.9
RUN_NAME = f"student_kd_{STUDENT}_T{TEMPERATURE:g}_alpha{ALPHA:g}_seed{SEED}".replace(".", "p")

import csv
import math
import random
from pathlib import Path

import os
import sys

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))
os.chdir(EXP_DIR)

import numpy as np
import torch
import torch.nn as nn

from models.cifar10_resnet import build_model
from prepare_data import build_loaders
from evaluate import evaluate
from losses import kd_loss


def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(DEVICE)

    train_loader, val_loader, _ = build_loaders(
        batch_size=BATCH_SIZE,
        seed=SEED,
        augment=AUGMENT,
        split_seed=SPLIT_SEED,
        download=False
    )
    student = build_model(STUDENT).to(device)
    print("Student model:", STUDENT)
    optimizer = torch.optim.SGD(student.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    total_steps = SCHEDULE_EPOCHS * steps_per_epoch

    # Keep the same per-update warmup and cosine schedule used by the completed runs.
    def learning_rate_multiplier(update):
        if update < warmup_steps:
            return (update + 1) / warmup_steps
        progress = min(1.0, (update - warmup_steps) / (total_steps - warmup_steps))
        minimum = MIN_LR / LR
        return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)
    criterion = nn.CrossEntropyLoss()
    teacher = build_model("resnet34").to(device)
    teacher_checkpoint = torch.load(
        TEACHER,
        map_location="cpu",
        weights_only=True
    )

    teacher.load_state_dict(
        teacher_checkpoint["model_state_dict"]
    )
    teacher.eval()
    teacher.requires_grad_(False)

    print("Teacher loaded and frozen:", TEACHER)
    run_name = RUN_NAME
    checkpoint_path = Path("checkpoints") / f"{run_name}.pt"
    log_path = Path("results") / f"{run_name}.csv"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0

    print("Run:", run_name, flush=True)
    print("Epoch log:", log_path, flush=True)
    print("\nTraining CIFAR-10 Student Reverse KD")
    print("Device:", device)
    print("Epochs:", EPOCHS)
    print("Base learning rate:", LR)
    print("Initial optimizer learning rate:", optimizer.param_groups[0]["lr"])
    print("Batch size:", BATCH_SIZE)
    print("Seed:", SEED)
    print("Temperature:", TEMPERATURE)
    print("Alpha:", ALPHA)
    print("Data augmentation:", AUGMENT)
    print("Train samples:", len(train_loader.dataset))
    print("Val samples:", len(val_loader.dataset))

    with open(log_path, "w", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(EPOCHS):

            student.train()

            total_loss = 0.0
            total_ce = 0.0
            total_kd = 0.0
            correct = 0
            total = 0
            for batch_idx, (images, labels) in enumerate(
                train_loader
            ):

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.no_grad():
                    teacher_logits = teacher(images)

                student_logits = student(images)

                loss, ce, soft = kd_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    T=TEMPERATURE,
                    alpha=ALPHA
                )

                if not torch.isfinite(loss).item():
                    raise RuntimeError(
                        f"Non-finite loss at "
                        f"epoch {epoch + 1}, "
                        f"batch {batch_idx + 1}, "
                        f"T={TEMPERATURE}"
                    )

                loss.backward()

                optimizer.step()
                scheduler.step()

                batch_size = labels.size(0)

                total_loss += loss.item() * batch_size
                total_ce += ce.item() * batch_size
                total_kd += soft.item() * batch_size

                correct += (
                    student_logits.argmax(dim=1) == labels
                ).sum().item()

                total += batch_size

                if epoch == 0 and (batch_idx + 1) % 100 == 0:

                    print(
                        f"  Batch {batch_idx + 1}/"
                        f"{len(train_loader)} | "
                        f"CE: {ce.item():.4f} | "
                        f"KD: {soft.item():.4f} | "
                        f"Total: {loss.item():.4f}",
                        flush=True
                    )

            train_loss = total_loss / total
            train_ce = total_ce / total
            train_kd = total_kd / total
            train_acc = correct / total

            val_loss, val_acc = evaluate(
                student, val_loader, criterion, device
            )

            print(
                f"Epoch {epoch + 1:02d}/{EPOCHS} | "
                f"Loss: {train_loss:.4f} | "
                f"CE: {train_ce:.4f} | "
                f"KD: {train_kd:.4f} | "
                f"Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc * 100:.2f}%",
                flush=True
            )

            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])
            f.flush()
            if val_acc > best_val_acc:

                best_val_acc = val_acc

                torch.save(
                    {
                        "model_state_dict": student.state_dict(),
                        "run_name": run_name,
                        "epoch": epoch + 1,
                        "val_acc": val_acc,
                        "seed": SEED,
                        "split_seed": SPLIT_SEED,
                        "batch_size": BATCH_SIZE,
                        "model_name": STUDENT,
                    },
                    checkpoint_path
                )

                print(
                    f"  Saved best checkpoint: "
                    f"{best_val_acc * 100:.2f}%",
                    flush=True
                )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    student.load_state_dict(checkpoint["model_state_dict"])

    print("Run:", run_name)
    print("Checkpoint:", checkpoint_path)
    print("Best epoch:", checkpoint["epoch"])
    print(f"Best validation accuracy: {100 * checkpoint['val_acc']:.2f}%")
    print("Training completed!.", flush=True)

if __name__ == "__main__":
    main()
