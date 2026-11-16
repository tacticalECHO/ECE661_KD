"""Train the Experiment C ResNet32x4 teacher on CIFAR-100."""

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

from models.cifar_resnet import count_parameters, resnet32x4
from prepare_data import CIFAR100_MEAN, CIFAR100_STD, build_loaders
from utils import append_summary


DEFAULT_OUTPUT = "checkpoints/teacher_resnet32x4_best.pt"
SUMMARY_PATH = "results/summary.csv"
SUMMARY_FIELDS = (
    "run_name",
    "method",
    "teacher_arch",
    "student_arch",
    "T",
    "kd_alpha",
    "dkd_alpha",
    "dkd_beta",
    "warmup",
    "optimizer",
    "lr",
    "lr_milestones",
    "momentum",
    "weight_decay",
    "batch_size",
    "epochs",
    "seed",
    "split_seed",
    "best_epoch",
    "best_val_acc",
    "test_acc",
    "test_errors",
    "test_loss",
    "params",
    "teacher",
)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    non_blocking = device.type == "cuda"

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        raise RuntimeError("Evaluation loader produced no samples")
    return total_loss / total, correct / total, total


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
    max_batches=None,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    non_blocking = device.type == "cuda"

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}, batch {batch_index + 1}"
            )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise RuntimeError("Training loader produced no samples")
    return total_loss / total, correct / total


def check_small_batch_fit(model, loader, criterion, optimizer, device, steps=20):
    """Verify that the teacher can reduce loss on one fixed mini-batch."""
    model.train()
    images, labels = next(iter(loader))
    images = images[:16].to(device)
    labels = labels[:16].to(device)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite loss in small-batch fit check")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if losses[-1] >= losses[0]:
        raise RuntimeError(
            f"Small-batch loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )
    print(f"Small-batch fit: {losses[0]:.4f} -> {losses[-1]:.4f} PASS")


def write_log_header(path, overwrite):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass --overwrite or choose another output"
        )
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"]
        )


def append_epoch_log(path, row):
    with open(path, "a", newline="") as stream:
        csv.writer(stream).writerow(row)


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    args,
    epoch,
    val_acc,
):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "architecture": "ResNet32x4",
        "num_classes": 100,
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
    parser.add_argument(
        "--download",
        action="store_true",
        help="download CIFAR-100 if it is not already present",
    )
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=str, default=None)
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
    if args.smoke:
        args.epochs = 1
        if args.output == DEFAULT_OUTPUT:
            args.output = "checkpoints/teacher_smoke.pt"
        args.overwrite = True

    set_seed(args.seed)
    device = resolve_device(args.device)
    train_loader, val_loader, test_loader = build_loaders(
        batch_size=args.batch_size,
        seed=args.seed,
        split_seed=args.split_seed,
        val_size=args.val_size,
        num_workers=args.num_workers,
        download=args.download,
    )

    model = resnet32x4().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_milestones,
        gamma=args.lr_decay,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite or choose another output"
        )
    run_name = output_path.stem
    log_path = Path(args.log) if args.log else Path("results") / f"{run_name}.csv"
    write_log_header(log_path, args.overwrite)

    max_batches = 2 if args.smoke else None
    best_val_acc = -1.0

    print("Training CIFAR-100 Teacher")
    print("Architecture: ResNet32x4")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
    print("Train/validation/test:", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    print("Epochs:", args.epochs)
    print("Seed / split seed:", args.seed, "/", args.split_seed)

    if args.smoke:
        check_small_batch_fit(model, train_loader, criterion, optimizer, device)

    for epoch_index in range(args.epochs):
        epoch = epoch_index + 1
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            max_batches=max_batches,
        )
        val_loss, val_acc, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_batches=max_batches,
        )
        append_epoch_log(
            log_path,
            [epoch, current_lr, train_loss, train_acc, val_loss, val_acc],
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | LR {current_lr:.5f} | "
            f"Train {train_loss:.4f}/{100 * train_acc:.2f}% | "
            f"Val {val_loss:.4f}/{100 * val_acc:.2f}%",
            flush=True,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    args,
                    epoch,
                    val_acc,
                ),
                output_path,
            )
            print(f"  Saved best checkpoint: {100 * val_acc:.2f}%", flush=True)
        scheduler.step()

    checkpoint = torch.load(output_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_n = evaluate(
        model,
        test_loader,
        criterion,
        device,
        max_batches=max_batches,
    )
    test_errors = round((1 - test_acc) * test_n)

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
                "method": "teacher",
                "teacher_arch": "ResNet32x4",
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
                "params": count_parameters(model),
            }
        )
        append_summary(SUMMARY_PATH, row)
        print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
