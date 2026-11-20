# Adapted from experiments/expA/evaluate_teacher.py
CHECKPOINT = "checkpoints/teacher_resnet34_s42_aa89a5a55c45d775.pt"
DEVICE = "cuda"
SPLIT = "test"

import os
import sys
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))
os.chdir(EXP_DIR)

import torch
import torch.nn as nn

from models.cifar10_resnet import build_model
from prepare_data import build_loaders
from utils import evaluate_per_class


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


def evaluate_checkpoint(checkpoint_path, device, split):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_name = checkpoint["model_name"]
    settings = checkpoint.get("config", {}).get("training", checkpoint)
    device = torch.device(device)
    _, val_loader, test_loader = build_loaders(
        batch_size=settings["batch_size"], seed=settings["seed"], augment=False,
        split_seed=settings["split_seed"], download=False
    )
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = val_loader if split == "validation" else test_loader
    metrics = evaluate_per_class(model, loader, nn.CrossEntropyLoss(), device, num_classes=10)
    return {"run_name": checkpoint["run_name"], "model_name": model_name, "seed": settings["seed"],
            "best_epoch": checkpoint["epoch"], "best_val_acc": checkpoint["val_acc"], "metrics": metrics}


def main():
    result = evaluate_checkpoint(CHECKPOINT, DEVICE, SPLIT)
    print("Model:", result["model_name"])
    print("Checkpoint:", CHECKPOINT)
    print("Split:", SPLIT)
    print("Best epoch:", result["best_epoch"])
    print("Metrics:", result["metrics"])


if __name__ == "__main__":
    main()
