"""Evaluate a frozen Experiment C teacher checkpoint."""

import argparse
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
from prepare_data import build_loaders
from train_teacher import evaluate, resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher",
        type=str,
        default="checkpoints/teacher_resnet32x4_best.pt",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="evaluation batch size; defaults to the checkpoint training batch size",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.teacher, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != "ResNet32x4":
        raise ValueError(
            f"Expected a ResNet32x4 checkpoint, got {checkpoint.get('architecture')!r}"
        )

    batch_size = args.batch_size or checkpoint["batch_size"]
    _, val_loader, test_loader = build_loaders(
        batch_size=batch_size,
        seed=checkpoint["seed"],
        split_seed=checkpoint["split_seed"],
        val_size=checkpoint["val_size"],
        num_workers=args.num_workers,
        download=False,
    )
    model = resnet32x4(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.requires_grad_(False)

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, val_n = evaluate(
        model, val_loader, criterion, device
    )
    test_loss, test_acc, test_n = evaluate(
        model, test_loader, criterion, device
    )

    print("CIFAR-100 Teacher Evaluation")
    print("Checkpoint:", args.teacher)
    print("Device:", device)
    print("Batch size:", batch_size)
    print("Best epoch:", checkpoint["epoch"])
    print(f"Validation: loss={val_loss:.4f}, accuracy={100 * val_acc:.2f}% ({val_n} images)")
    print(
        f"Test: loss={test_loss:.4f}, accuracy={100 * test_acc:.2f}%, "
        f"errors={round((1 - test_acc) * test_n)}/{test_n}"
    )
    print("Parameters:", count_parameters(model))


if __name__ == "__main__":
    main()
