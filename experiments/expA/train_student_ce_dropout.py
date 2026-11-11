
import argparse
import csv
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

from models.mnist import StudentCNNDropout, StudentCNNExsDropout, StudentCNNExs2Dropout
from prepare_data import build_loaders
from train_teacher import evaluate
from utils import evaluate_per_class, append_summary

SUMMARY_PATH = "results/summary_dropout.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str, default="StudentCNNDropout", choices=["StudentCNNDropout", "StudentCNNExsDropout", "StudentCNNExs2Dropout"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--exclude-classes", type=int, nargs="+", default=None)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")

    train_loader, val_loader, test_loader = build_loaders(
        batch_size=args.batch_size,
        seed=args.seed,
        augment=False,
        exclude_classes=args.exclude_classes,
        split_seed=42
    )

    if args.student == "StudentCNNDropout":
        model = StudentCNNDropout(p=args.dropout).to(device)
    elif args.student == "StudentCNNExsDropout":
        model = StudentCNNExsDropout(p=args.dropout).to(device)
    elif args.student == "StudentCNNExs2Dropout":
        model = StudentCNNExs2Dropout(p=args.dropout).to(device)
    print("Student model:", args.student)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )
    if args.exclude_classes is not None:
        exclude_classes_str = "_".join(map(str, args.exclude_classes))
        checkpoint_path = Path(
            f"checkpoints/student_Dropout{args.dropout}_ce_seed{args.seed}_exclude_{exclude_classes_str}_{args.student}.pt"
        )
        log_path = Path(
            f"results/student_Dropout{args.dropout}_ce_seed{args.seed}_exclude_{exclude_classes_str}_{args.student}.csv"
        )
    else:
        checkpoint_path = Path(
            f"checkpoints/student_Dropout{args.dropout}_ce_seed{args.seed}_{args.student}.pt"
        )
        log_path = Path(
            f"results/student_Dropout{args.dropout}_ce_seed{args.seed}_{args.student}.csv"
        )
    run_name = checkpoint_path.stem
    checkpoint_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    log_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    best_val_acc = -1.0

    print("Training Student CE Baseline")
    print("Device:", device)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.lr)
    print("Batch size:", args.batch_size)
    print("Seed:", args.seed)
    print("Data augmentation: False")
    print("Loss: CrossEntropyLoss")
    with open(log_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc"
        ])

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
                        f"Non-finite loss at "
                        f"epoch {epoch + 1}, "
                        f"batch {batch_idx + 1}"
                    )
                loss.backward()

                optimizer.step()
                total_loss += (
                    loss.item() * labels.size(0)
                )

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

            writer.writerow([
                epoch + 1,
                train_loss,
                train_acc,
                val_loss,
                val_acc
            ])

            f.flush()
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
                        "method": "CE",
                        "augment": False,
                        "exclude_classes": args.exclude_classes if args.exclude_classes is not None else "",
                        "run_name": run_name
                    },
                    checkpoint_path
                )

                print(
                    f"  Saved best checkpoint: "
                    f"{best_val_acc * 100:.2f}%",
                    flush=True
                )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    if args.exclude_classes is not None:

        _, full_val_loader, _ = build_loaders(
            batch_size=args.batch_size,
            seed=args.seed,
            augment=False,
            split_seed=42,
            exclude_classes=None
        )

        full_val = evaluate_per_class(
            model,
            full_val_loader,
            criterion,
            device
        )

        print("\nFull Validation Evaluation")
        print("Samples:", len(full_val_loader.dataset))
        print(f"Full Val Loss: {full_val['loss']:.4f}")
        print(f"Full Val Accuracy: {full_val['acc'] * 100:.2f}%")
        print("Full Val Errors:", full_val["errors"])

        for c in args.exclude_classes:
            print(
                f"Missing Digit {c} Val Accuracy: "
                f"{full_val['per_class_acc'][c] * 100:.2f}%"
            )

        # Save full validation results
        full_val_path = Path(
            f"results/{run_name}_full_val.csv"
        )

        full_val_row = {
            "run_name": run_name,
            "method": "ce",
            "seed": args.seed,
            "exclude_classes": "_".join(
                map(str, args.exclude_classes)
            ),
            "best_epoch": checkpoint["epoch"],
            "selection_val_acc": best_val_acc,
            "full_val_acc": full_val["acc"],
            "full_val_loss": full_val["loss"],
            "full_val_errors": full_val["errors"]
        }

        # Per-class validation accuracy
        for c, acc in enumerate(full_val["per_class_acc"]):
            full_val_row[f"full_val_acc_{c}"] = acc

        # Per-class validation errors
        for c, errors in enumerate(full_val["per_class_errors"]):
            full_val_row[f"full_val_errors_{c}"] = errors

        with open(full_val_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(full_val_row.keys())
            )

            writer.writeheader()
            writer.writerow(full_val_row)

        print("Full validation results:", full_val_path)

    test = evaluate_per_class(
        model,
        test_loader,
        criterion,
        device
    )

    num_params = sum(p.numel() for p in model.parameters())

    print("\nTraining completed!")
    print("------------------")
    print("Best epoch:", checkpoint["epoch"])
    print(f"Best Val Acc: {best_val_acc * 100:.2f}%")
    print(f"Test Loss: {test['loss']:.4f}")
    print(f"Test Accuracy: {test['acc'] * 100:.2f}%")
    print("Test Errors:", test["errors"])
    print(
        "Per-class errors:",
        test["per_class_errors"]
    )
    print("Parameters:", num_params)

    row = {
        "run_name": run_name,
        "method": "ce",
        "T": "",
        "alpha": "",
        "optimizer": "adam",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "exclude_classes": (
            "_".join(map(str, args.exclude_classes))
            if args.exclude_classes is not None
            else ""
        ),
        "best_epoch": checkpoint["epoch"],
        "best_val_acc": best_val_acc,
        "test_acc": test["acc"],
        "test_errors": test["errors"],
        "test_loss": test["loss"],
        "params": num_params,
        "student": args.student,
    }

    for c, acc in enumerate(test["per_class_acc"]):
        row[f"acc_{c}"] = acc

    row["teacher"] = "" 

    row["dropout"] = args.dropout

    append_summary(SUMMARY_PATH, row)

    print("Checkpoint:", checkpoint_path)
    print("Training log:", log_path)
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
