
import os
import sys
from pathlib import Path

# Shared code (models/, losses.py, utils.py) lives at the repo root; relative paths
# (results/, checkpoints/, logs/) resolve inside experiments/expA from any working directory.
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))
os.chdir(EXP_DIR)

import torch
import torch.nn as nn
from models.mnist import TeacherCNN, StudentCNN


def test_model(model, name, device):

    print(f"\nTesting {name}...", flush=True)

    model = model.to(device)
    model.train()

    # Small batch for GPU smoke test
    images = torch.randn(4, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (4,)).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.CrossEntropyLoss()

    # Forward
    outputs = model(images)
    torch.cuda.synchronize()

    print("Forward passed:", outputs.shape, flush=True)

    # Loss
    loss = criterion(outputs, labels)

    assert torch.isfinite(loss).item()

    print("Loss:", loss.item(), flush=True)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()

    print("Backward passed!", flush=True)

    # Check gradients
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all().item()

    # Update parameters
    optimizer.step()
    torch.cuda.synchronize()

    print("Optimizer step passed!", flush=True)

    print(
        "Peak GPU memory:",
        round(
            torch.cuda.max_memory_allocated()
            / 1024**2, 2
        ),
        "MB",
        flush=True
    )


if __name__ == "__main__":

    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))

    print(
        "Total GPU memory:",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory / 1024**3, 2
        ),
        "GB"
    )

    # Test teacher
    torch.cuda.reset_peak_memory_stats()
    test_model(TeacherCNN(), "Teacher CNN", device)

    torch.cuda.empty_cache()

    # Test student
    torch.cuda.reset_peak_memory_stats()
    test_model(StudentCNN(), "Student CNN", device)

    print("\nAll GPU tests passed!")
