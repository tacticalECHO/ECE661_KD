
import argparse
from pathlib import Path

import os
import sys

# Shared code (models/, losses.py, utils.py) lives at the repo root; relative paths
# (results/, checkpoints/, logs/) resolve inside experiments/expA from any working directory.
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))
os.chdir(EXP_DIR)

import torch
import torch.nn as nn

from models.mnist import TeacherCNN
from prepare_data import build_loaders


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    
    parser.add_argument(
        "--output",
        type=str,
        default=f"checkpoints/teacher_best.pt"
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader, test_loader = build_loaders(
        batch_size=args.batch_size,
        seed=args.seed,
        augment=True,
        exclude_classes=None,
        split_seed=42
    )

    device = torch.device("cuda")

    model = TeacherCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    best_val_acc = -1.0

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Training Teacher CNN")
    print("Device:", device)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.lr)
    print("Batch size:", args.batch_size)
    print("Seed:", args.seed)
    print("Data augmentation: True")

    print("Train samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    for epoch in range(args.epochs):

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(
            train_loader
        ):

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch + 1}, "
                    f"batch {batch_idx + 1}"
                )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

            correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            total += labels.size(0)
        train_loss = total_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%",
            flush=True
        )
        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                    "seed": args.seed,
                    "split_seed": 42,
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "augment": True
                },
                output_path
            )

            print(
                f"  Saved best checkpoint: "
                f"{best_val_acc * 100:.2f}%",
                flush=True
            )

    print("\nTraining completed!")

    print(
        f"Best validation accuracy: "
        f"{best_val_acc * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
