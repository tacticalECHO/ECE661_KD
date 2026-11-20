SEEDS = (42, 123, 456)
TEMPERATURE = 4.0
ALPHA = 0.9
DEVICE = "cuda"
TEACHER_CHECKPOINT = "checkpoints/teacher_resnet34_s42_aa89a5a55c45d775.pt"
OUTPUT_DIR = "results/evaluation"

import csv
import json
import statistics
from pathlib import Path

from evaluate import evaluate_checkpoint

EXP_DIR = Path(__file__).resolve().parent


def main():
    jobs = [("teacher", TEACHER_CHECKPOINT)]
    for seed in SEEDS:
        ce_name = f"student_ce_resnet50_seed{seed}"
        kd_name = f"student_kd_resnet50_T{TEMPERATURE:g}_alpha{ALPHA:g}_seed{seed}".replace(".", "p")
        jobs.extend([("ce", f"checkpoints/{ce_name}.pt"), ("kd", f"checkpoints/{kd_name}.pt")])

    output = EXP_DIR / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (method, checkpoint) in enumerate(jobs, 1):
        checkpoint = EXP_DIR / checkpoint
        print(f"Evaluating {index}/{len(jobs)}: {checkpoint.stem}", flush=True)
        result = evaluate_checkpoint(checkpoint, DEVICE, "test")
        metrics = result["metrics"]
        row = {"run_name": result["run_name"], "method": method,
               "model_name": result["model_name"], "seed": result["seed"],
               "checkpoint": str(checkpoint), "best_epoch": result["best_epoch"],
               "best_val_acc": result["best_val_acc"],
               "temperature": TEMPERATURE if method == "kd" else "",
               "alpha": ALPHA if method == "kd" else "",
               "test_acc": metrics["acc"], "test_errors": metrics["errors"],
               "test_loss": metrics["loss"], "n": metrics["n"]}
        row.update({f"acc_{i}": value for i, value in enumerate(metrics["per_class_acc"])})
        rows.append(row)

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    accuracy_by_seed = {"ce": {}, "kd": {}}
    for row in rows:
        method = row["method"]
        if method in accuracy_by_seed:
            accuracy_by_seed[method][row["seed"]] = row["test_acc"]

    paired = []
    for seed in SEEDS:
        ce = accuracy_by_seed["ce"][seed]
        kd = accuracy_by_seed["kd"][seed]
        paired.append({"seed": seed, "kd_minus_ce_pp": 100 * (kd - ce)})

    aggregate = {}
    for method, accuracies in accuracy_by_seed.items():
        percentages = [100 * accuracies[seed] for seed in SEEDS]
        aggregate[method] = {
            "mean_test_acc_percent": statistics.mean(percentages),
            "sample_std_test_acc_pp": statistics.stdev(percentages),
        }
    differences = [pair["kd_minus_ce_pp"] for pair in paired]
    aggregate.update(paired=paired, mean_kd_minus_ce_pp=statistics.mean(differences),
                     sample_std_kd_minus_ce_pp=statistics.stdev(differences),
                     teacher_test_acc=rows[0]["test_acc"], seeds=list(SEEDS), temperature=TEMPERATURE, alpha=ALPHA)
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print("Final test evaluation completed successfully.", flush=True)
    print("Summary:", output / "summary.csv", flush=True)
    print("Aggregate:", output / "aggregate.json", flush=True)


if __name__ == "__main__":
    main()
