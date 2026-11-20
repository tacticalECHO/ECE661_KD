# Adapted from experiments/expA/FLOPs.py
import csv
from pathlib import Path

import os
import sys

# Shared code (models/, losses.py, utils.py) lives at the repo root; relative paths
# (results/, checkpoints/, logs/) resolve inside experiments/expA from any working directory.
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))
os.chdir(EXP_DIR)

import torch
import torch.nn as nn

from models.cifar10_resnet import build_model

STATS_PATH = Path("results/model_stats.csv")


def profile_model(model):
    macs = 0
    hooks = []
    def count_layer(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            kh, kw = module.kernel_size
            macs += (
                output.numel()
                * (module.in_channels // module.groups)
                * kh
                * kw
            )
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features
        else:
            return

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(count_layer))
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 3, 32, 32))
    finally:
        for hook in hooks:
            hook.remove()

    params = sum(p.numel() for p in model.parameters())
    flops = 2 * macs
    return params, macs, flops


def main():
    models = {
        "resnet34": build_model("resnet34"),
        "resnet50": build_model("resnet50"),
    }

    print(f"{'Model':<10} {'Params':>12} {'MACs':>15} {'FLOPs':>15} {'MFLOPs':>10}")
    print("-" * 70)

    rows = []
    for name, model in models.items():
        params, macs, flops = profile_model(model)
        print(f"{name:<10} {params:>12,} {macs:>15,} {flops:>15,} {flops / 1e6:>10.4f}")

        rows.append({
            "model": name,
            "student": name,
            "params": params,
            "macs": macs,
            "flops": flops,
        })

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(STATS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {STATS_PATH}")

    print("\nFLOPs = 2 × MACs; counts Conv2d and Linear only.")


if __name__ == "__main__":
    main()
