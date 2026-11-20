"""Parameter, MAC, and FLOP counts for the Experiment C model pair."""

import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(EXP_DIR)

from models.cifar_resnet import resnet8x4, resnet32x4


OUTPUT_PATH = Path("results/model_stats.csv")


def profile_model(model):
    macs_by_layer = []
    hooks = []

    def count_layer(name):
        def hook(module, inputs, output):
            if isinstance(module, nn.Conv2d):
                kernel_height, kernel_width = module.kernel_size
                macs = (
                    output.numel()
                    * (module.in_channels // module.groups)
                    * kernel_height
                    * kernel_width
                )
            elif isinstance(module, nn.Linear):
                macs = output.numel() * module.in_features
            else:
                return
            macs_by_layer.append((name, macs))

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(count_layer(name)))

    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, 3, 32, 32))
    finally:
        for hook in hooks:
            hook.remove()

    parameters = sum(parameter.numel() for parameter in model.parameters())
    macs = sum(value for _, value in macs_by_layer)
    return parameters, macs, 2 * macs


def main():
    models = {
        "Teacher": ("ResNet32x4", resnet32x4()),
        "Student": ("ResNet8x4", resnet8x4()),
    }
    rows = []
    for role, (architecture, model) in models.items():
        parameters, macs, flops = profile_model(model)
        rows.append(
            {
                "role": role,
                "architecture": architecture,
                "params": parameters,
                "macs": macs,
                "flops": flops,
            }
        )
        print(
            f"{role:<8} {architecture:<12} params={parameters:,} "
            f"MACs={macs:,} FLOPs={flops:,}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
