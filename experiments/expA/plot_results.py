"""Draw the Experiment A figures from the raw result CSVs.

Inputs (all under results/):
  summary.csv           one row per run: temperature sweep, missing digit, capacity
  summary_dropout.csv   one row per run: dropout teachers and dropout students
  <run_name>.csv        per-epoch training log of each run (used for the curves)

Outputs: figures/<name>.pdf (for the report) and figures/<name>.png (for the poster).
"""

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = Path(__file__).resolve().parent
os.chdir(EXP_DIR)

RESULTS = Path("results")
FIGURES = Path("figures")

SEEDS = ("42", "123", "456")
MAIN_TEACHER = "checkpoints/teacher_final.pt"
OMITTED_DIGIT = 1

# Fixed colors per method, used identically in every figure.
COLORS = {
    "kd": "#2a78d6",       # blue
    "ce": "#eb6834",       # orange
    "kd_t20": "#1baf7a",   # aqua (second KD setting in the missing-digit figure)
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
    rows = read_csv(RESULTS / "summary.csv") + read_csv(RESULTS / "summary_dropout.csv")
    for r in rows:
        r.setdefault("dropout", "")
    return rows


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

def fig_temperature(rows):
    """Test errors of StudentCNN vs. temperature, with the CE baseline as a band."""
    temps = [1, 2, 4, 8, 20]
    kd = [require_seeds(select(rows, method="kd", student="StudentCNN", T=float(t),
                               exclude_classes="", teacher=MAIN_TEACHER), f"KD T={t}")
          for t in temps]
    ce = require_seeds(select(rows, method="ce", student="StudentCNN", exclude_classes="",
                              dropout=""), "CE")

    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    ce_mean, ce_sd = mean_sd(errors(ce))
    ax.axhspan(ce_mean - ce_sd, ce_mean + ce_sd, color=COLORS["ce"], alpha=0.12, linewidth=0)
    ax.axhline(ce_mean, color=COLORS["ce"], linewidth=1.5, linestyle="--",
               label="CE baseline (mean ± sd band)")
    ax.text(1.1, ce_mean + 1.5, f"CE {ce_mean:.0f}", color=INK_SECONDARY,
            ha="left", va="bottom", fontsize=8)

    stats = [mean_sd(errors(g)) for g in kd]
    means = [m for m, _ in stats]
    sds = [s for _, s in stats]
    for t, g in zip(temps, kd):
        seed_dots(ax, [t] * len(g), errors(g), COLORS["kd"])
    ax.errorbar(temps, means, yerr=sds, color=COLORS["kd"], linewidth=2, marker="o",
                markersize=5, capsize=3, elinewidth=1, label="KD (mean ± sd)", zorder=4)

    ax.set_xscale("log")
    ax.set_xticks(temps)
    ax.set_xticklabels([str(t) for t in temps])
    ax.minorticks_off()
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Test errors (of 10,000)")
    ax.set_title("StudentCNN: KD vs. temperature (3 seeds)", loc="left", color=INK)
    ax.set_ylim(top=ce_mean + ce_sd + 22)
    ax.legend(loc="upper right", ncol=1)
    save(fig, "temperature_sweep")


def fig_training_curves(rows):
    """Validation loss and accuracy per epoch: CE, CE + dropout 0.5, and KD (T=4)."""
    groups = [
        ("CE", COLORS["ce"], "-",
         select(rows, method="ce", student="StudentCNN", exclude_classes="", dropout="")),
        ("CE + dropout 0.5", COLORS["ce"], ":",
         select(rows, method="ce", student="StudentCNNDropout", dropout="0.5")),
        ("KD, T = 4", COLORS["kd"], "-",
         select(rows, method="kd", student="StudentCNN", T=4.0, exclude_classes="",
                teacher=MAIN_TEACHER)),
    ]
    for label, _, _, g in groups:
        require_seeds(g, label)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for column, ax, ylabel in [("val_loss", axes[0], "Validation loss (CE)"),
                               ("val_acc", axes[1], "Validation accuracy (%)")]:
        for label, color, style, g in groups:
            curves = np.stack([epoch_log(r["run_name"], column) for r in g])
            if column == "val_acc":
                curves = curves * 100
            epochs = np.arange(1, curves.shape[1] + 1)
            for c in curves:
                ax.plot(epochs, c, color=color, linestyle=style, linewidth=0.6, alpha=0.3)
            ax.plot(epochs, curves.mean(axis=0), color=color, linestyle=style,
                    linewidth=2, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 60)
    axes[1].set_ylim(98.0, 99.6)
    axes[0].set_title("Validation loss (thin: seeds, thick: mean)", loc="left", color=INK)
    axes[1].set_title("Validation accuracy", loc="left", color=INK)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "training_curves")


def fig_missing_digit(rows):
    """Omitted digit 1: its test accuracy, and the errors on the other nine digits."""
    exc = str(OMITTED_DIGIT)
    groups = [
        ("CE", COLORS["ce"],
         select(rows, method="ce", student="StudentCNN", exclude_classes=exc)),
        ("KD, T = 4", COLORS["kd"],
         select(rows, method="kd", student="StudentCNN", T=4.0, exclude_classes=exc)),
        ("KD, T = 20", COLORS["kd_t20"],
         select(rows, method="kd", student="StudentCNN", T=20.0, exclude_classes=exc)),
    ]
    for label, _, g in groups:
        require_seeds(g, label)

    n_digit = 1135  # MNIST test set: 1,135 images of digit 1

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    x = np.arange(len(groups))
    for i, (label, color, g) in enumerate(groups):
        acc = np.array([float(r[f"acc_{OMITTED_DIGIT}"]) * 100 for r in g])
        other_err = np.array([int(r["test_errors"])
                              - round((1 - float(r[f"acc_{OMITTED_DIGIT}"])) * n_digit)
                              for r in g], dtype=float)
        for ax, vals in [(axes[0], acc), (axes[1], other_err)]:
            m, s = mean_sd(vals)
            ax.bar(i, m, width=0.6, color=color, edgecolor="white", linewidth=2)
            ax.errorbar(i, m, yerr=s, color=INK, capsize=3, elinewidth=1, linewidth=0)
            seed_dots(ax, [i] * len(vals), vals, INK, jitter=0.12)
            ax.text(i, m + s + (2 if ax is axes[0] else 3), f"{m:.1f}" if ax is axes[0]
                    else f"{m:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([label for label, _, _ in groups])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(0, 135)
    axes[0].set_ylabel(f"Test accuracy on digit {OMITTED_DIGIT} (%)")
    axes[0].set_title(f"Never-seen digit {OMITTED_DIGIT} ({n_digit} test images)",
                      loc="left", color=INK)
    axes[1].set_ylabel("Test errors on the other 9 digits")
    axes[1].set_title("Errors on the 9 seen digits", loc="left", color=INK)
    fig.tight_layout()
    save(fig, "missing_digit")


def fig_capacity(rows):
    """Test errors vs. student size for CE and KD (T = 4)."""
    students = ["StudentCNNExs2", "StudentCNNExs", "StudentCNN"]
    names = {"StudentCNNExs2": "XXS", "StudentCNNExs": "XS", "StudentCNN": "S"}
    series = {
        "ce": [require_seeds(select(rows, method="ce", student=s, exclude_classes="",
                                    dropout=""), f"CE {s}") for s in students],
        "kd": [require_seeds(select(rows, method="kd", student=s, T=4.0, exclude_classes="",
                                    teacher=MAIN_TEACHER), f"KD {s}") for s in students],
    }
    params = [int(series["ce"][i][0]["params"]) for i in range(len(students))]

    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for key, label in [("ce", "CE"), ("kd", "KD, T = 4")]:
        stats = [mean_sd(errors(g)) for g in series[key]]
        for p, g in zip(params, series[key]):
            seed_dots(ax, [p] * len(g), errors(g), COLORS[key])
        ax.errorbar(params, [m for m, _ in stats], yerr=[s for _, s in stats],
                    color=COLORS[key], linewidth=2, marker="o", markersize=5, capsize=3,
                    elinewidth=1, label=label, zorder=4)
        ax.text(params[-1] * 1.12, stats[-1][0], label, color=INK, va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(params)
    ax.set_xticklabels([f"{names[s]}\n{p:,}" for s, p in zip(students, params)])
    ax.minorticks_off()
    ax.set_xlim(params[0] / 1.5, params[-1] * 2.0)
    ax.set_xlabel("Student size (parameters)")
    ax.set_ylabel("Test errors (of 10,000)")
    ax.set_title("Student capacity: CE vs. KD (3 seeds)", loc="left", color=INK)
    ax.legend(loc="upper right")
    save(fig, "capacity")


def fig_dropout(rows):
    """Test errors of StudentCNN with and without dropout, for CE and KD (T = 4)."""
    groups = [
        ("CE", "ce", select(rows, method="ce", student="StudentCNN", exclude_classes="",
                            dropout="")),
        ("CE, student dropout 0.3", "ce",
         select(rows, method="ce", student="StudentCNNDropout", dropout="0.3")),
        ("CE, student dropout 0.5", "ce",
         select(rows, method="ce", student="StudentCNNDropout", dropout="0.5")),
        ("KD, plain teacher", "kd",
         select(rows, method="kd", student="StudentCNN", T=4.0, exclude_classes="",
                teacher=MAIN_TEACHER)),
        ("KD, teacher dropout 0.3", "kd",
         select(rows, method="kd", teacher="checkpoints/teacher_dropout0.3_final.pt")),
        ("KD, teacher dropout 0.5", "kd",
         select(rows, method="kd", teacher="checkpoints/teacher_dropout0.5_final.pt")),
    ]
    for label, _, g in groups:
        require_seeds(g, label)

    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    y = np.arange(len(groups))[::-1]
    for yi, (label, key, g) in zip(y, groups):
        m, s = mean_sd(errors(g))
        ax.barh(yi, m, height=0.6, color=COLORS[key], edgecolor="white", linewidth=2)
        ax.errorbar(m, yi, xerr=s, color=INK, capsize=3, elinewidth=1, linewidth=0)
        ax.scatter(errors(g), [yi] * len(g), s=14, color=INK, alpha=0.45, linewidths=0,
                   zorder=3)
        ax.text(m + s + 3, yi, f"{m:.0f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in groups])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 160)
    ax.set_xlabel("Test errors (of 10,000; bars: mean ± sd, dots: seeds)")
    ax.set_title("Dropout vs. distillation (StudentCNN)", loc="left", color=INK)
    save(fig, "dropout")


def main():
    rows = load_summary()
    fig_temperature(rows)
    fig_training_curves(rows)
    fig_missing_digit(rows)
    fig_capacity(rows)
    fig_dropout(rows)


if __name__ == "__main__":
    main()
