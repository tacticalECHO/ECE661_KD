
import csv
from pathlib import Path

import torch


def evaluate_per_class(model, loader, criterion, device, num_classes=10):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            hit = preds == labels

            total_loss += loss.item() * labels.size(0)
            correct += hit.sum().item()
            total += labels.size(0)

            class_total += torch.bincount(
                labels, minlength=num_classes
            ).cpu()

            class_correct += torch.bincount(
                labels[hit], minlength=num_classes
            ).cpu()

    per_class_acc = (
        class_correct.float() / class_total.clamp(min=1).float()
    ).tolist()

    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "errors": total - correct,
        "n": total,
        "per_class_acc": per_class_acc,
        "per_class_errors": (class_total - class_correct).tolist(),
    }


def append_summary(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_file = not path.exists()

    if not new_file:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != list(row.keys()):
            raise ValueError(
                f"{path} has columns {header}, but this row has "
                f"{list(row.keys())}. Update the file's header first "
                f"(or write to a new file) instead of appending misaligned rows."
            )

    with open(path, "a", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(row.keys())
        )

        if new_file:
            writer.writeheader()

        writer.writerow(row)
