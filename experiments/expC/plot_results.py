"""Validate Experiment C results and generate Stage 5 tables and figures.

Inputs are ``results/summary.csv`` and the 15 student per-epoch CSV logs.  The
script refuses to generate final artifacts unless every CE, KD, TCKD, NCKD,
and DKD run is present for seeds 42, 123, and 456 with 240 complete epochs.

Outputs:

* ``results/method_summary.csv``: three-seed aggregate statistics;
* ``results/paper_comparison.csv``: project-versus-paper comparison;
* ``figures/test_error_comparison.{pdf,png}``;
* ``figures/validation_curves.{pdf,png}``;
* ``figures/dkd_components.{pdf,png}``.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"

METHODS = ("ce", "kd", "tckd", "nckd", "dkd")
SEEDS = (42, 123, 456)
EPOCHS = 240
DISPLAY = {
    "ce": "CE",
    "kd": "KD",
    "tckd": "TCKD",
    "nckd": "NCKD",
    "dkd": "DKD",
}
COLORS = {
    "ce": "#E76F51",
    "kd": "#457B9D",
    "tckd": "#9C6ADE",
    "nckd": "#2A9D8F",
    "dkd": "#1D3557",
}
INK = "#111111"
INK_SECONDARY = "#555555"
GRID = "#E2E2E2"

# CVPR 2022, Table 6, ResNet32x4 -> ResNet8x4.  The paper reports averages
# over five trials.  The student row is the no-distillation baseline.
PAPER_RESULTS = {
    "teacher": 0.7942,
    "ce": 0.7250,
    "kd": 0.7333,
    "dkd": 0.7632,
}

plt.rcParams.update(
    {
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
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path.relative_to(EXP_DIR)}")


def as_float(row: dict[str, str], column: str, label: str) -> float:
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label}: invalid {column!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label}: non-finite {column!r}")
    return value


def mean_sd(values: list[float] | np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def load_and_validate() -> tuple[dict[tuple[str, int], dict[str, str]], dict[tuple[str, int], list[dict[str, str]]], dict[str, str]]:
    summary_path = RESULTS_DIR / "summary.csv"
    summary = read_csv(summary_path)
    teacher_rows = [row for row in summary if row["method"] == "teacher"]
    if len(teacher_rows) != 1:
        raise ValueError(f"expected one teacher row, found {len(teacher_rows)}")

    student_rows: dict[tuple[str, int], dict[str, str]] = {}
    for row in summary:
        if row["method"] not in METHODS:
            continue
        key = (row["method"], int(row["seed"]))
        if key in student_rows:
            raise ValueError(f"duplicate summary row for method/seed {key}")
        student_rows[key] = row

    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    missing = sorted(expected - set(student_rows))
    extra = sorted(set(student_rows) - expected)
    if missing or extra:
        raise ValueError(f"student matrix mismatch: missing={missing}, extra={extra}")

    logs: dict[tuple[str, int], list[dict[str, str]]] = {}
    for key in sorted(expected):
        row = student_rows[key]
        label = f"{key[0]} seed {key[1]}"
        log_path = RESULTS_DIR / f"{row['run_name']}.csv"
        log = read_csv(log_path)
        if len(log) != EPOCHS:
            raise ValueError(f"{label}: expected {EPOCHS} epochs, found {len(log)}")
        epochs = [int(item["epoch"]) for item in log]
        if epochs != list(range(1, EPOCHS + 1)):
            raise ValueError(f"{label}: epoch column is incomplete or out of order")
        for column in ("train_loss", "train_ce_loss", "train_acc", "val_loss", "val_acc"):
            for item in log:
                as_float(item, column, label)

        validation = np.array([float(item["val_acc"]) for item in log])
        best_index = int(np.argmax(validation))
        recorded_epoch = int(row["best_epoch"])
        recorded_value = as_float(row, "best_val_acc", label)
        if recorded_epoch != best_index + 1 or not math.isclose(
            recorded_value, float(validation[best_index]), abs_tol=1e-12
        ):
            raise ValueError(
                f"{label}: summary best checkpoint does not match its epoch log"
            )
        logs[key] = log

    return student_rows, logs, teacher_rows[0]


def build_method_summary(student_rows: dict[tuple[str, int], dict[str, str]]) -> list[dict[str, object]]:
    means: dict[str, float] = {}
    for method in METHODS:
        means[method] = np.mean(
            [float(student_rows[(method, seed)]["test_acc"]) for seed in SEEDS]
        )

    output: list[dict[str, object]] = []
    for method in METHODS:
        rows = [student_rows[(method, seed)] for seed in SEEDS]
        accuracies = [float(row["test_acc"]) for row in rows]
        errors = [float(row["test_errors"]) for row in rows]
        validations = [float(row["best_val_acc"]) for row in rows]
        best_epochs = [float(row["best_epoch"]) for row in rows]
        acc_mean, acc_sd = mean_sd(accuracies)
        error_mean, error_sd = mean_sd(errors)
        output.append(
            {
                "method": method,
                "seed_42_test_acc": round(accuracies[0], 6),
                "seed_123_test_acc": round(accuracies[1], 6),
                "seed_456_test_acc": round(accuracies[2], 6),
                "mean_test_acc": round(acc_mean, 6),
                "sample_sd_test_acc": round(acc_sd, 6),
                "mean_test_errors": round(error_mean, 2),
                "sample_sd_test_errors": round(error_sd, 2),
                "mean_best_val_acc": round(float(np.mean(validations)), 6),
                "mean_best_epoch": round(float(np.mean(best_epochs)), 2),
                "delta_vs_ce_pp": round(100.0 * (acc_mean - means["ce"]), 4),
                "delta_vs_kd_pp": round(100.0 * (acc_mean - means["kd"]), 4),
            }
        )
    return output


def build_paper_comparison(
    method_summary: list[dict[str, object]], teacher_row: dict[str, str]
) -> list[dict[str, object]]:
    aggregates = {str(row["method"]): row for row in method_summary}
    project_values = {
        "teacher": float(teacher_row["test_acc"]),
        "ce": float(aggregates["ce"]["mean_test_acc"]),
        "kd": float(aggregates["kd"]["mean_test_acc"]),
        "dkd": float(aggregates["dkd"]["mean_test_acc"]),
    }
    output = []
    for method in ("teacher", "ce", "kd", "dkd"):
        project = project_values[method]
        paper = PAPER_RESULTS[method]
        project_sd: object = ""
        project_trials = 1
        if method != "teacher":
            project_sd = aggregates[method]["sample_sd_test_acc"]
            project_trials = len(SEEDS)
        output.append(
            {
                "method": method,
                "project_test_acc": round(project, 6),
                "project_sample_sd": project_sd,
                "project_trials": project_trials,
                "paper_test_acc": paper,
                "paper_trials": 5,
                "project_minus_paper_pp": round(100.0 * (project - paper), 4),
                "protocol_note": (
                    "Project uses a fixed 45k/5k train/validation split and selects the "
                    "best validation checkpoint; paper values use the authors' protocol."
                ),
            }
        )
    return output


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = FIGURES_DIR / f"{name}.{extension}"
        fig.savefig(path)
    plt.close(fig)
    print(f"saved figures/{name}.pdf and .png")


def plot_test_errors(student_rows: dict[tuple[str, int], dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    positions = np.arange(len(METHODS))
    for position, method in zip(positions, METHODS):
        values = np.array(
            [int(student_rows[(method, seed)]["test_errors"]) for seed in SEEDS],
            dtype=float,
        )
        mean, sd = mean_sd(values)
        ax.bar(position, mean, width=0.68, color=COLORS[method], edgecolor="white")
        ax.errorbar(position, mean, yerr=sd, color=INK, capsize=3, linewidth=0, elinewidth=1)
        offsets = np.linspace(-0.13, 0.13, len(SEEDS))
        ax.scatter(position + offsets, values, color=INK, alpha=0.5, s=16, linewidths=0)
        ax.text(position, mean + sd + 28, f"{mean:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY[method] for method in METHODS])
    ax.set_ylabel("Test errors (of 10,000)")
    ax.set_title("CIFAR-100 test errors (bars: mean, whiskers: sample SD)", loc="left")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 3500)
    fig.tight_layout()
    save_figure(fig, "test_error_comparison")


def log_curve(
    logs: dict[tuple[str, int], list[dict[str, str]]], method: str, column: str
) -> np.ndarray:
    return np.stack(
        [
            np.array([float(row[column]) for row in logs[(method, seed)]], dtype=float)
            for seed in SEEDS
        ]
    )


def plot_validation_curves(logs: dict[tuple[str, int], list[dict[str, str]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    epochs = np.arange(1, EPOCHS + 1)
    for method in METHODS:
        for column, ax, scale in (
            ("val_acc", axes[0], 100.0),
            ("val_loss", axes[1], 1.0),
        ):
            curves = log_curve(logs, method, column) * scale
            for curve in curves:
                ax.plot(epochs, curve, color=COLORS[method], linewidth=0.45, alpha=0.12)
            ax.plot(
                epochs,
                curves.mean(axis=0),
                color=COLORS[method],
                linewidth=1.55,
                label=DISPLAY[method],
            )
    for ax in axes:
        for milestone in (150, 180, 210):
            ax.axvline(milestone, color="#AAAAAA", linewidth=0.55, linestyle=":")
        ax.set_xlim(1, EPOCHS)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[1].set_ylabel("Validation cross-entropy loss")
    axes[0].set_title("Validation accuracy", loc="left")
    axes[1].set_title("Validation loss", loc="left")
    axes[0].legend(loc="lower right", ncol=2)
    fig.tight_layout()
    save_figure(fig, "validation_curves")


def plot_dkd_components(logs: dict[tuple[str, int], list[dict[str, str]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    epochs = np.arange(1, EPOCHS + 1)
    component_methods = ("tckd", "nckd", "dkd")
    panels = (
        ("train_tckd_loss", axes[0], "Unweighted TCKD loss"),
        ("train_nckd_loss", axes[1], "Unweighted NCKD loss"),
    )
    for column, ax, ylabel in panels:
        for method in component_methods:
            curves = log_curve(logs, method, column)
            for curve in curves:
                ax.plot(epochs, curve, color=COLORS[method], linewidth=0.45, alpha=0.12)
            ax.plot(
                epochs,
                curves.mean(axis=0),
                color=COLORS[method],
                linewidth=1.55,
                label=DISPLAY[method],
            )
        ax.axvspan(1, 20, color="#BBBBBB", alpha=0.12, linewidth=0)
        for milestone in (150, 180, 210):
            ax.axvline(milestone, color="#AAAAAA", linewidth=0.55, linestyle=":")
        ax.set_xlim(1, EPOCHS)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
    axes[0].set_title("TCKD component (warmup shaded)", loc="left")
    axes[1].set_title("NCKD component (warmup shaded)", loc="left")
    axes[1].legend(loc="best")
    fig.tight_layout()
    save_figure(fig, "dkd_components")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the 15-run matrix and epoch logs without writing artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    student_rows, logs, teacher_row = load_and_validate()
    print("validated 15 student runs and 3,600 student epochs")
    if args.check_only:
        return

    method_summary = build_method_summary(student_rows)
    write_csv(
        RESULTS_DIR / "method_summary.csv",
        [
            "method",
            "seed_42_test_acc",
            "seed_123_test_acc",
            "seed_456_test_acc",
            "mean_test_acc",
            "sample_sd_test_acc",
            "mean_test_errors",
            "sample_sd_test_errors",
            "mean_best_val_acc",
            "mean_best_epoch",
            "delta_vs_ce_pp",
            "delta_vs_kd_pp",
        ],
        method_summary,
    )
    write_csv(
        RESULTS_DIR / "paper_comparison.csv",
        [
            "method",
            "project_test_acc",
            "project_sample_sd",
            "project_trials",
            "paper_test_acc",
            "paper_trials",
            "project_minus_paper_pp",
            "protocol_note",
        ],
        build_paper_comparison(method_summary, teacher_row),
    )
    plot_test_errors(student_rows)
    plot_validation_curves(logs)
    plot_dkd_components(logs)


if __name__ == "__main__":
    main()
