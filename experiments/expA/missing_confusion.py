"""Confusion matrices of the missing-digit students, summed over seeds.

For each setting (CE, KD T=4, KD T=20) trained without the omitted digit, the runs are looked up
in results/summary.csv, each seed's best checkpoint is evaluated on the full MNIST test set, and
the per-seed confusion matrices are summed.

Outputs:
  results/confusion_missing{d}_{setting}.csv   summed counts (rows: true digit, cols: predicted)
  figures/confusion_missing{d}.pdf / .png      row-normalized matrices, one panel per setting
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))
os.chdir(EXP_DIR)

from models.mnist import StudentCNN
from prepare_data import build_loaders

SUMMARY_PATH = Path("results/summary.csv")
SEEDS = ("42", "123", "456")
NUM_CLASSES = 10

# (file tag, panel title, summary.csv filter)
SETTINGS = [
    ("ce", "CE", {"method": "ce", "T": ""}),
    ("kd_T4", "KD, T = 4", {"method": "kd", "T": "4.0"}),
    ("kd_T20", "KD, T = 20", {"method": "kd", "T": "20.0"}),
]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
HIGHLIGHT = "#eb6834"


def find_runs(digit, criteria):
    """Run names of the StudentCNN runs that omitted `digit`, one per seed."""
    with open(SUMMARY_PATH, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["student"] == "StudentCNN" and r["exclude_classes"] == str(digit)
                and all(r[k] == v for k, v in criteria.items())]
    seeds = sorted(r["seed"] for r in rows)
    if seeds != sorted(SEEDS):
        raise ValueError(f"{criteria}: expected seeds {SEEDS}, found {seeds}")
    return [r["run_name"] for r in rows]


def confusion_matrix(path, loader, device):
    model = StudentCNN().to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    counts = torch.zeros(NUM_CLASSES * NUM_CLASSES, dtype=torch.long, device=device)
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            counts += torch.bincount(labels * NUM_CLASSES + preds,
                                     minlength=NUM_CLASSES * NUM_CLASSES)
    return counts.view(NUM_CLASSES, NUM_CLASSES).cpu().numpy()


def save_csv(conf_matrix, class_names, path):
    df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    df.index.name = "true"
    df.to_csv(path)
    print(f"Confusion matrix saved to {path}")


def plot_confusion_matrices(matrices, titles, class_names, digit, n_seeds):
    """Row-normalized matrices side by side; the omitted digit's row is outlined."""
    fig, axes = plt.subplots(1, len(matrices), figsize=(3.3 * len(matrices) + 0.6, 3.6),
                             constrained_layout=True)
    for ax, counts, title in zip(axes, matrices, titles):
        percent = 100 * counts / counts.sum(axis=1, keepdims=True)
        im = ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(NUM_CLASSES), class_names)
        ax.set_yticks(range(NUM_CLASSES), class_names)
        ax.tick_params(length=0, labelsize=8, colors=INK_SECONDARY)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Predicted digit", fontsize=8, color=INK)
        ax.set_title(title, loc="left", fontsize=10, color=INK)

        # label only the cells that hold at least 1% of their row
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                value = percent[i, j]
                if value >= 1:
                    ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if value >= 50 else INK)

        ax.add_patch(Rectangle((-0.5, digit - 0.5), NUM_CLASSES, 1, fill=False,
                               edgecolor=HIGHLIGHT, linewidth=1.5))
    axes[0].set_ylabel("True digit", fontsize=8, color=INK)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("% of the true digit's test images", fontsize=8, color=INK)
    cbar.ax.tick_params(labelsize=7, colors=INK_SECONDARY)
    cbar.outline.set_visible(False)
    fig.suptitle(f"Test-set confusion matrices, digit {digit} omitted from the transfer set "
                 f"(summed over {n_seeds} seeds; cells < 1% unlabeled)",
                 x=0.0, ha="left", fontsize=9, color=INK_SECONDARY)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digit", type=int, default=1, help="Digit omitted from the transfer set")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, loader = build_loaders(batch_size=1000, augment=False, split_seed=42)
    class_names = [str(i) for i in range(NUM_CLASSES)]

    matrices, titles = [], []
    for tag, title, criteria in SETTINGS:
        total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for run_name in find_runs(args.digit, criteria):
            total += confusion_matrix(Path(f"checkpoints/{run_name}.pt"), loader, device)
        save_csv(total, class_names, Path(f"results/confusion_missing{args.digit}_{tag}.csv"))

        row = total[args.digit]
        print(f"{title}: digit {args.digit} correct {row[args.digit]}/{row.sum()} "
              f"({100 * row[args.digit] / row.sum():.1f}%), "
              f"other digits predicted as {args.digit}: "
              f"{total[:, args.digit].sum() - row[args.digit]}")
        matrices.append(total)
        titles.append(title)

    fig = plot_confusion_matrices(matrices, titles, class_names, args.digit, len(SEEDS))
    Path("figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        path = Path(f"figures/confusion_missing{args.digit}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print("Figure saved to", path)


if __name__ == "__main__":
    main()
