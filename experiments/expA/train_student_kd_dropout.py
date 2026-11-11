
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

from models.mnist import TeacherCNNDropout, StudentCNN, StudentCNNExs, StudentCNNExs2
from prepare_data import build_loaders
from train_teacher import evaluate
from losses import kd_loss
from utils import evaluate_per_class, append_summary

SUMMARY_PATH = "results/summary_dropout.csv"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str, default="StudentCNN", choices=["StudentCNN", "StudentCNNExs", "StudentCNNExs2"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--exclude-classes", type=int, nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true")
    # e.g. checkpoints/teacher_dropout0.5_best.pt; its dropout rate is read from the checkpoint
    parser.add_argument("--teacher", type=str, required=True)

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

    if args.student == "StudentCNN":
        student = StudentCNN().to(device)
    elif args.student == "StudentCNNExs":
        student = StudentCNNExs().to(device)
    elif args.student == "StudentCNNExs2":
        student = StudentCNNExs2().to(device)
    print("Student model:", args.student)
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=args.lr
    )

    criterion = nn.CrossEntropyLoss()

    teacher_checkpoint = torch.load(
        args.teacher,
        map_location=device,
        weights_only=True
    )

    # Take the teacher's dropout rate from its own checkpoint so the run name
    # can never disagree with the teacher that was actually loaded.
    if "dropout" not in teacher_checkpoint:
        raise ValueError(f"{args.teacher} has no 'dropout' entry; is it a dropout teacher?")
    teacher_dropout = teacher_checkpoint["dropout"]

    teacher = TeacherCNNDropout(p=teacher_dropout).to(device)

    teacher.load_state_dict(
        teacher_checkpoint["model_state_dict"]
    )

    teacher.eval()
    teacher.requires_grad_(False)

    print("Teacher loaded and frozen:", args.teacher)
    print("Teacher dropout:", teacher_dropout)

    run_name = (
        f"student_kd_tdropout{teacher_dropout}"
        f"_T{args.temperature:g}"
        f"_alpha{args.alpha:g}"
        f"_seed{args.seed}"
        f"_{args.student}"
    )

    exclude_classes_str = ""

    if args.exclude_classes is not None:

        exclude_classes_str = "_".join(
            map(str, sorted(args.exclude_classes))
        )

        run_name += f"_exclude_{exclude_classes_str}"

    if args.smoke:
        run_name += "_smoke"

    checkpoint_path = Path(
        f"checkpoints/{run_name}.pt"
    )

    log_path = Path(
        f"results/{run_name}.csv"
    )

    checkpoint_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    log_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    best_val_acc = -1.0

    print("\nTraining Student KD")
    print("Device:", device)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.lr)
    print("Batch size:", args.batch_size)
    print("Seed:", args.seed)
    print("Temperature:", args.temperature)
    print("Alpha:", args.alpha)
    print("Data augmentation: False")
    print("Excluded classes:", args.exclude_classes)
    print("Train samples:", len(train_loader.dataset))
    print("Val samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    with open(log_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_ce_loss",
            "train_kd_loss",
            "train_acc",
            "val_loss",
            "val_acc"
        ])

        for epoch in range(args.epochs):

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
                    T=args.temperature,
                    alpha=args.alpha
                )

                if not torch.isfinite(loss).item():
                    raise RuntimeError(
                        f"Non-finite loss at "
                        f"epoch {epoch + 1}, "
                        f"batch {batch_idx + 1}, "
                        f"T={args.temperature}"
                    )

                loss.backward()

                optimizer.step()

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
                student,
                val_loader,
                criterion,
                device
            )

            print(
                f"Epoch {epoch + 1:02d}/{args.epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"CE: {train_ce:.4f} | "
                f"KD: {train_kd:.4f} | "
                f"Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc * 100:.2f}%",
                flush=True
            )

            writer.writerow([
                epoch + 1,
                train_loss,
                train_ce,
                train_kd,
                train_acc,
                val_loss,
                val_acc
            ])

            f.flush()

            if val_acc > best_val_acc:

                best_val_acc = val_acc

                torch.save(
                    {
                        "model_state_dict": student.state_dict(),
                        "epoch": epoch + 1,
                        "val_acc": val_acc,
                        "seed": args.seed,
                        "split_seed": 42,
                        "lr": args.lr,
                        "batch_size": args.batch_size,
                        "temperature": args.temperature,
                        "alpha": args.alpha,
                        "method": "KD",
                        "augment": False,
                        "exclude_classes": args.exclude_classes,
                        "run_name": run_name,
                        "teacher": args.teacher,
                        "teacher_dropout": teacher_dropout
                    },
                    checkpoint_path
                )

                print(
                    f"  Saved best checkpoint: "
                    f"{best_val_acc * 100:.2f}%",
                    flush=True
                )

    if args.smoke:

        print("\nKD training smoke test passed!")
        return

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    student.load_state_dict(
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
            student,
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

        full_val_path = Path(
            f"results/{run_name}_full_val.csv"
        )

        full_val_row = {
            "run_name": run_name,
            "method": "kd",
            "seed": args.seed,
            "exclude_classes": exclude_classes_str,
            "best_epoch": checkpoint["epoch"],
            "selection_val_acc": best_val_acc,
            "full_val_acc": full_val["acc"],
            "full_val_loss": full_val["loss"],
            "full_val_errors": full_val["errors"]
        }

        for c, acc in enumerate(full_val["per_class_acc"]):

            full_val_row[f"full_val_acc_{c}"] = acc

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
        student,
        test_loader,
        criterion,
        device
    )

    num_params = sum(
        p.numel() for p in student.parameters()
    )

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

    if args.exclude_classes is not None:

        for c in args.exclude_classes:

            print(
                f"Missing Digit {c} Test Accuracy: "
                f"{test['per_class_acc'][c] * 100:.2f}%"
            )

    print("Parameters:", num_params)

    row = {
        "run_name": run_name,
        "method": "kd",
        "T": args.temperature,
        "alpha": args.alpha,
        "optimizer": "adam",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "exclude_classes": exclude_classes_str,
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

    row["teacher"] = args.teacher

    row["dropout"] = 0.0

    append_summary(SUMMARY_PATH, row)

    print("Checkpoint:", checkpoint_path)
    print("Training log:", log_path)
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
