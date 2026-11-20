# Adapted from experiments/expA/prepare_data.py for CIFAR-10.
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SPLIT_DIR = Path(__file__).resolve().parent / "splits"
TRAIN_SIZE = 45000


class SeededTransform:
    def __init__(self, transform, seed):
        self.transform = transform
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, image):
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.generator.get_state())
            result = self.transform(image)
            self.generator.set_state(torch.get_rng_state())
        return result


def _normalization_stats(images, train_indices):
    # Compute channel statistics from the training subset
    pixels = images[train_indices].astype(np.float64) / 255.0
    return {"mean": pixels.mean(axis=(0, 1, 2)).tolist(),
            "std": pixels.std(axis=(0, 1, 2)).tolist()}


def _load_or_create_split(dataset, split_seed):
    split_path = SPLIT_DIR / f"cifar10_seed{split_seed}.json"
    if split_path.exists():
        metadata = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        generator = torch.Generator().manual_seed(split_seed)
        indices = torch.randperm(len(dataset), generator=generator).tolist()
        train_indices, val_indices = indices[:TRAIN_SIZE], indices[TRAIN_SIZE:]
        metadata = {
            "normalization": _normalization_stats(dataset.data, train_indices),
            "train_indices": train_indices,
            "val_indices": val_indices,
        }
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        with split_path.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream)
    return metadata


def build_loaders(
    batch_size=128,
    seed=42,
    augment=True,
    split_seed=42,
    download=True
):
    full_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=download)
    metadata = _load_or_create_split(full_train, split_seed)
    mean = metadata["normalization"]["mean"]
    std = metadata["normalization"]["std"]
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if augment:
        train_transform = SeededTransform(transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]), seed)
    else:
        train_transform = eval_transform

    full_train.transform = train_transform

    full_val = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=eval_transform
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=eval_transform
    )

    train_indices = metadata["train_indices"]
    val_indices = metadata["val_indices"]

    train_dataset = Subset(
        full_train,
        train_indices
    )

    val_dataset = Subset(
        full_val,
        val_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed)
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_loaders()

    print("Train samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Image dtype:", images.dtype)

    print("Data preparation passed!")
