"""Train the Experiment C ResNet8x4 student with CE, KD, or DKD variants."""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(EXP_DIR)

from losses import dkd_loss, kd_loss
from models.cifar_resnet import count_parameters, resnet8x4, resnet32x4
from prepare_data import CIFAR100_MEAN, CIFAR100_STD, build_loaders
from train_teacher import SUMMARY_FIELDS, evaluate, resolve_device, set_seed
from utils import append_summary


SUMMARY_PATH = "results/summary.csv"
DEFAULT_TEACHER = "checkpoints/teacher_resnet32x4_best.pt"


DISTILLATION_METHODS = {"kd", "tckd", "nckd", "dkd"}
DKD_METHODS = {"tckd", "nckd", "dkd"}


def default_run_name(
    method,
    seed,
    temperature,
    alpha,
    dkd_alpha,
    dkd_beta,
    warmup,
):
    if method == "ce":
        return f"student_resnet8x4_ce_seed{seed}_best"
    if method == "kd":
        settings = f"T{temperature:g}_alpha{alpha:g}"
    elif method == "tckd":
        settings = f"T{temperature:g}_alpha{dkd_alpha:g}_warmup{warmup}"
    elif method == "nckd":
        settings = f"T{temperature:g}_beta{dkd_beta:g}_warmup{warmup}"
    else:
        settings = (
            f"T{temperature:g}_alpha{dkd_alpha:g}_beta{dkd_beta:g}_"
            f"warmup{warmup}"
        )
    return f"student_resnet8x4_{method}_{settings}_seed{seed}_best"


def distillation_weight(epoch, warmup):
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if warmup == 0:
        return 1.0
    return min(float(epoch) / warmup, 1.0)


def load_frozen_teacher(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != "ResNet32x4":
        raise ValueError(
            f"Expected a ResNet32x4 teacher, got {checkpoint.get('architecture')!r}"
        )
    teacher = resnet32x4(num_classes=checkpoint.get("num_classes", 100)).to(device)
    teacher.load_state_dict(checkpoint["model_state_dict"])
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher, checkpoint


def train_one_epoch(
    student,
    teacher,
    loader,
    optimizer,
    device,
    method,
    temperature,
    alpha,
    dkd_alpha,
    dkd_beta,
    warmup,
    epoch,
    max_batches=None,
):
    student.train()
    if teacher is not None:
        teacher.eval()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_kd_loss = 0.0
    total_tckd_loss = 0.0
    total_nckd_loss = 0.0
    correct = 0
    total = 0
    non_blocking = device.type == "cuda"

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        student_logits = student(images)
        if method == "ce":
            ce_loss = nn.functional.cross_entropy(student_logits, labels)
            soft_loss = student_logits.new_zeros(())
            loss = ce_loss
        else:
            with torch.no_grad():
                teacher_logits = teacher(images)
            ce_loss = nn.functional.cross_entropy(student_logits, labels)
            if method == "kd":
                loss, ce_loss, soft_loss = kd_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    T=temperature,
                    alpha=alpha,
                )
                tckd = student_logits.new_zeros(())
                nckd = student_logits.new_zeros(())
            else:
                method_alpha = dkd_alpha if method in {"tckd", "dkd"} else 0.0
                method_beta = dkd_beta if method in {"nckd", "dkd"} else 0.0
                distill_loss, tckd, nckd = dkd_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    temperature=temperature,
                    alpha=method_alpha,
                    beta=method_beta,
                )
                soft_loss = student_logits.new_zeros(())
                loss = ce_loss + distillation_weight(epoch, warmup) * distill_loss

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}, batch {batch_index + 1}"
            )
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_ce_loss += ce_loss.item() * batch_size
        total_kd_loss += soft_loss.item() * batch_size
        total_tckd_loss += tckd.item() * batch_size if method != "ce" else 0.0
        total_nckd_loss += nckd.item() * batch_size if method != "ce" else 0.0
        correct += (student_logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    if total == 0:
        raise RuntimeError("Training loader produced no samples")
    return {
        "loss": total_loss / total,
        "ce_loss": total_ce_loss / total,
        "kd_loss": total_kd_loss / total,
        "tckd_loss": total_tckd_loss / total,
        "nckd_loss": total_nckd_loss / total,
        "distill_weight": (
            distillation_weight(epoch, warmup) if method in DKD_METHODS else 0.0
        ),
        "acc": correct / total,
    }


def write_log_header(path, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass --overwrite or choose another output"
        )
    with open(path, "w", newline="") as stream:
        csv.writer(stream).writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_ce_loss",
                "train_kd_loss",
                "train_tckd_loss",
                "train_nckd_loss",
                "distill_weight",
                "train_acc",
                "val_loss",
                "val_acc",
            ]
        )


def append_epoch_log(path, row):
    with open(path, "a", newline="") as stream:
        csv.writer(stream).writerow(row)


def checkpoint_payload(
    student,
    optimizer,
    scheduler,
    args,
    epoch,
    val_acc,
    teacher_path,
):
    return {
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "architecture": "ResNet8x4",
        "num_classes": 100,
        "method": args.method,
        "temperature": (
            args.temperature if args.method in DISTILLATION_METHODS else None
        ),
        "alpha": args.alpha if args.method == "kd" else None,
        "dkd_alpha": args.dkd_alpha if args.method in {"tckd", "dkd"} else None,
        "dkd_beta": args.dkd_beta if args.method in {"nckd", "dkd"} else None,
        "warmup": args.warmup if args.method in DKD_METHODS else None,
        "teacher": teacher_path if args.method in DISTILLATION_METHODS else None,
        "epoch": epoch,
        "val_acc": val_acc,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "val_size": args.val_size,
        "optimizer": "sgd",
        "lr": args.lr,
        "lr_milestones": tuple(args.lr_milestones),
        "lr_decay": args.lr_decay,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "normalization_mean": CIFAR100_MEAN,
        "normalization_std": CIFAR100_STD,
        "augmentation": "RandomCrop(32, padding=4) + RandomHorizontalFlip",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=("ce", "kd", "tckd", "nckd", "dkd"), required=True
    )
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-milestones", type=int, nargs="+", default=[150, 180, 210])
    parser.add_argument("--lr-decay", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--dkd-alpha", type=float, default=1.0)
    parser.add_argument("--dkd-beta", type=float, default=8.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--teacher", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one epoch with two train/validation/test batches",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.dkd_alpha < 0 or args.dkd_beta < 0:
        raise ValueError("DKD alpha and beta must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.smoke:
        args.epochs = 1
        args.overwrite = True

    device = resolve_device(args.device)
    set_seed(args.seed)
    train_loader, val_loader, test_loader = build_loaders(
        batch_size=args.batch_size,
        seed=args.seed,
        split_seed=args.split_seed,
        val_size=args.val_size,
        num_workers=args.num_workers,
        download=args.download,
    )

    # Reset immediately before student construction so CE and KD have identical
    # initial weights for the same seed, independent of loader/teacher setup.
    set_seed(args.seed)
    student = resnet8x4().to(device)
    teacher = None
    teacher_path = ""
    if args.method in DISTILLATION_METHODS:
        teacher_path = args.teacher
        teacher, _ = load_frozen_teacher(teacher_path, device)

    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_milestones,
        gamma=args.lr_decay,
    )
    criterion = nn.CrossEntropyLoss()

    run_name = default_run_name(
        args.method,
        args.seed,
        args.temperature,
        args.alpha,
        args.dkd_alpha,
        args.dkd_beta,
        args.warmup,
    )
    if args.smoke:
        run_name = f"student_{args.method}_smoke"
    output_path = Path(args.output or f"checkpoints/{run_name}.pt")
    log_path = Path(args.log or f"results/{run_name}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite or choose another output"
        )
    write_log_header(log_path, args.overwrite)

    max_batches = 2 if args.smoke else None
    best_val_acc = -1.0

    print("Training CIFAR-100 Student")
    print("Method / architecture:", args.method, "/ ResNet8x4")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
    print(
        "Train/validation/test:",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )
    print("Epochs / seed:", args.epochs, "/", args.seed)
    if args.method in DISTILLATION_METHODS:
        print("Teacher:", teacher_path)
        if args.method == "kd":
            print("Temperature / KD alpha:", args.temperature, "/", args.alpha)
        else:
            print(
                "Temperature / DKD alpha / beta / warmup:",
                args.temperature,
                "/",
                args.dkd_alpha,
                "/",
                args.dkd_beta,
                "/",
                args.warmup,
            )

    for epoch_index in range(args.epochs):
        epoch = epoch_index + 1
        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            student,
            teacher,
            train_loader,
            optimizer,
            device,
            args.method,
            args.temperature,
            args.alpha,
            args.dkd_alpha,
            args.dkd_beta,
            args.warmup,
            epoch,
            max_batches=max_batches,
        )
        val_loss, val_acc, _ = evaluate(
            student,
            val_loader,
            criterion,
            device,
            max_batches=max_batches,
        )
        append_epoch_log(
            log_path,
            [
                epoch,
                current_lr,
                train_metrics["loss"],
                train_metrics["ce_loss"],
                train_metrics["kd_loss"] if args.method == "kd" else "",
                train_metrics["tckd_loss"] if args.method in DKD_METHODS else "",
                train_metrics["nckd_loss"] if args.method in DKD_METHODS else "",
                train_metrics["distill_weight"] if args.method in DKD_METHODS else "",
                train_metrics["acc"],
                val_loss,
                val_acc,
            ],
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | LR {current_lr:.5f} | "
            f"Train {train_metrics['loss']:.4f}/{100 * train_metrics['acc']:.2f}% | "
            f"Val {val_loss:.4f}/{100 * val_acc:.2f}%",
            flush=True,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                checkpoint_payload(
                    student,
                    optimizer,
                    scheduler,
                    args,
                    epoch,
                    val_acc,
                    teacher_path,
                ),
                output_path,
            )
            print(f"  Saved best checkpoint: {100 * val_acc:.2f}%", flush=True)
        scheduler.step()

    checkpoint = torch.load(output_path, map_location=device, weights_only=True)
    student.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_n = evaluate(
        student,
        test_loader,
        criterion,
        device,
        max_batches=max_batches,
    )
    test_errors = round((1 - test_acc) * test_n)

    if teacher is not None:
        if any(parameter.grad is not None for parameter in teacher.parameters()):
            raise RuntimeError("Frozen teacher unexpectedly received gradients")

    print("Best epoch:", checkpoint["epoch"])
    print(f"Best validation accuracy: {100 * checkpoint['val_acc']:.2f}%")
    print(f"Test accuracy: {100 * test_acc:.2f}% ({test_errors}/{test_n} errors)")
    print("Checkpoint:", output_path)
    print("Epoch log:", log_path)

    if not args.smoke:
        row = dict.fromkeys(SUMMARY_FIELDS, "")
        row.update(
            {
                "run_name": run_name,
                "method": args.method,
                "teacher_arch": (
                    "ResNet32x4" if args.method in DISTILLATION_METHODS else ""
                ),
                "student_arch": "ResNet8x4",
                "T": (
                    args.temperature if args.method in DISTILLATION_METHODS else ""
                ),
                "kd_alpha": args.alpha if args.method == "kd" else "",
                "dkd_alpha": (
                    args.dkd_alpha if args.method in {"tckd", "dkd"} else ""
                ),
                "dkd_beta": (
                    args.dkd_beta if args.method in {"nckd", "dkd"} else ""
                ),
                "warmup": args.warmup if args.method in DKD_METHODS else "",
                "optimizer": "sgd",
                "lr": args.lr,
                "lr_milestones": "_".join(map(str, args.lr_milestones)),
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "best_epoch": checkpoint["epoch"],
                "best_val_acc": checkpoint["val_acc"],
                "test_acc": test_acc,
                "test_errors": test_errors,
                "test_loss": test_loss,
                "params": count_parameters(student),
                "teacher": (
                    teacher_path if args.method in DISTILLATION_METHODS else ""
                ),
            }
        )
        append_summary(SUMMARY_PATH, row)
        print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
