import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
os.chdir(EXP_DIR)

RESULTS = Path("results")
SUMMARY_PATH = RESULTS / "final_evaluation" / "summary.csv"
FIGURES = Path("figures")

SEEDS = ("42", "123", "456")

# Fixed colors per method, used identically in every figure.
COLORS = {
    "kd": "#2a78d6",
    "ce": "#eb6834",
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": INK_SECONDARY,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


# ---------------------------------------------------------------- data access

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_summary():
    return read_csv(SUMMARY_PATH)


def select(rows, **criteria):
    """Rows whose columns equal every given value (compared as strings)."""
    out = [r for r in rows if all(r[k] == str(v) for k, v in criteria.items())]
    return sorted(out, key=lambda r: SEEDS.index(r["seed"]) if r["seed"] in SEEDS else 0)


def require_seeds(rows, label):
    seeds = [r["seed"] for r in rows]
    if sorted(seeds) != sorted(SEEDS):
        raise ValueError(f"{label}: expected seeds {SEEDS}, found {seeds}")
    return rows


def errors(rows):
    return np.array([int(r["test_errors"]) for r in rows], dtype=float)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std(ddof=1)


def epoch_log(run_name, column):
    rows = read_csv(RESULTS / f"{run_name}.csv")
    return np.array([float(r[column]) for r in rows])


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print("saved", FIGURES / f"{name}.pdf", "and .png")


def seed_dots(ax, x, values, color, jitter=0.0):
    offsets = np.linspace(-jitter, jitter, len(values)) if jitter else np.zeros(len(values))
    ax.scatter(np.asarray(x) + offsets, values, s=14, color=color, alpha=0.45,
               linewidths=0, zorder=3)

# ---------------------------------------------------------------- figures


def draw_validation_curves(rows, first_epoch):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for method, label in (("ce", "Student CE"), ("kd", "Reverse KD")):
        group = require_seeds(select(rows, method=method), label)
        for column, ax in (("val_loss", axes[0]), ("val_acc", axes[1])):
            curves = np.stack([epoch_log(row["run_name"], column) for row in group])
            epochs = np.arange(1, curves.shape[1] + 1)[first_epoch - 1:]
            curves = curves[:, first_epoch - 1:]
            if column == "val_acc":
                curves *= 100
            for curve in curves:
                ax.plot(epochs, curve, color=COLORS[method], linewidth=0.65, alpha=0.25)
            ax.plot(epochs, curves.mean(axis=0), color=COLORS[method], linewidth=1.8, label=label)
    axes[1].set_ylabel("Validation accuracy (%)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.margins(x=0)
        ax.legend()
    return fig, axes


def fig_training_curves(rows):
    fig, axes = draw_validation_curves(rows, first_epoch=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Validation cross-entropy (log scale)")
    axes[0].set_title("Validation loss; thin: seeds, thick: mean")
    axes[1].set_title("Validation accuracy")
    fig.tight_layout()
    save(fig, "training_curves")


def fig_late_training_curves(rows):
    fig, axes = draw_validation_curves(rows, first_epoch=151)
    for ax in axes:
        ax.margins(y=0.06)
    axes[0].set_ylabel("Validation cross-entropy")
    axes[0].set_title("Validation CE, epochs 151-200 (zoom)")
    axes[1].set_title("Validation accuracy, epochs 151-200 (zoom)")
    fig.tight_layout()
    save(fig, "late_training_curves")


def fig_test_comparison(rows):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
    teacher = select(rows, method="teacher")[0]
    teacher_acc = 100 * float(teacher["test_acc"])
    groups = {}
    for x, method, label in ((0, "ce", "CE"), (1, "kd", "Reverse KD")):
        group = require_seeds(select(rows, method=method), label)
        values = np.array([100 * float(row["test_acc"]) for row in group])
        groups[method] = values
        mean, sd = mean_sd(values)
        axes[0].scatter(x + np.linspace(-0.07, 0.07, 3), values,
                        color=COLORS[method], alpha=0.55, s=22)
        axes[0].errorbar(x, mean, yerr=sd, fmt="D", color=COLORS[method], capsize=5, markersize=5)
        axes[0].annotate(f"{mean:.2f}%", (x, mean + sd + 0.035), ha="center", fontsize=8)
    axes[0].axhline(teacher_acc, color=INK_SECONDARY, linestyle="--", linewidth=1,
                    label=f"Teacher: {teacher_acc:.2f}% (one seed)")
    axes[0].set_xticks([0, 1], ["CE", "Reverse KD"])
    axes[0].set_xlim(-0.5, 1.5)
    axes[0].set_ylim(95.05, 95.85)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("Dots: seeds; diamonds: mean +/- sample SD")
    axes[0].legend(loc="lower center", fontsize=7)
    differences = groups["kd"] - groups["ce"]
    axes[1].axhline(0, color=INK_SECONDARY, linestyle="--", linewidth=1)
    axes[1].scatter(range(3), differences, color=COLORS["kd"], s=35)
    axes[1].set_xticks(range(3), SEEDS)
    axes[1].set_xlabel("Paired student seed")
    axes[1].set_ylabel("KD - CE test accuracy (percentage points)")
    axes[1].set_ylim(-0.3, 0.08)
    for x, difference in enumerate(differences):
        axes[1].annotate(f"{difference:+.2f}", (x, difference), xytext=(0, 8),
                         textcoords="offset points", ha="center", fontsize=8)
    axes[1].set_title(f"Paired mean difference: {differences.mean():+.3f} pp")
    fig.tight_layout()
    save(fig, "test_comparison")


def fig_temperature():
    temperatures = (2, 4, 8)
    accuracies = []
    for temperature in temperatures:
        run_name = f"student_kd_resnet50_T{temperature}_alpha0p9_seed42"
        accuracies.append(100 * epoch_log(run_name, "val_acc").max())
    baseline = 100 * epoch_log("student_ce_resnet50_seed42", "val_acc").max()
    fig, ax = plt.subplots(figsize=(5.1, 2.8))
    ax.plot(temperatures, accuracies, "o-", color=COLORS["kd"], label="KD; alpha = 0.9")
    ax.scatter([4], [accuracies[1]], s=90, facecolors="none", edgecolors=INK, label="Selected: T = 4")
    ax.axhline(baseline, color=COLORS["ce"], linestyle="--", label=f"CE: {baseline:.2f}%")
    for temperature, accuracy in zip(temperatures, accuracies):
        ax.annotate(f"{accuracy:.2f}%", (temperature, accuracy), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(temperatures)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_ylim(95.75, 96.24)
    ax.set_title("Temperature selection: validation only, seed 42")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "temperature_validation")


def fig_model_size():
    rows = read_csv(RESULTS / "model_stats.csv")
    models = {row["model"]: row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, field, scale, label in ((axes[0], "params", 1e6, "Parameters (millions)"),
                                    (axes[1], "flops", 1e9, "FLOPs per image (billions)")):
        values = [float(models[name][field]) / scale for name in ("resnet34", "resnet50")]
        ax.bar([0, 1], values, color=[INK_SECONDARY, COLORS["ce"]], width=0.55)
        ax.set_xticks([0, 1], ["Teacher\nResNet-34", "Student\nResNet-50"])
        ax.set_ylabel(label)
        ax.set_ylim(0, max(values) * 1.22)
        for x, value in enumerate(values):
            ax.text(x, value + max(values) * 0.025, f"{value:.3f}", ha="center", fontsize=8)
    axes[0].set_title("Model capacity")
    axes[1].set_title("Conv/Linear only; FLOPs = 2 x MACs")
    fig.tight_layout()
    save(fig, "model_size")


def main():
    rows = load_summary()
    fig_training_curves(rows)
    fig_late_training_curves(rows)
    fig_test_comparison(rows)
    fig_temperature()
    fig_model_size()


if __name__ == "__main__":
    main()
