
import argparse

import os
import sys
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))
os.chdir(EXP_DIR)

import torch
import torch.nn as nn

from models.mnist import TeacherCNN
from prepare_data import build_loaders
from train_teacher import evaluate


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=str, default="checkpoints/teacher_final.pt")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Full validation and test datasets
    _, val_loader, test_loader = build_loaders(
        batch_size=128,
        seed=42,
        augment=False,
        exclude_classes=None
    )

    # Load the best teacher
    model = TeacherCNN().to(device)

    checkpoint = torch.load(
        args.teacher,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    criterion = nn.CrossEntropyLoss()

    # Validation
    val_loss, val_acc = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    # Test
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    test_errors = round(
        (1 - test_acc) * len(test_loader.dataset)
    )

    print("Teacher Evaluation")
    print("------------------")

    print("Best epoch:", checkpoint["epoch"])

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    print("Test Errors:", test_errors)

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )


if __name__ == "__main__":
    main()
