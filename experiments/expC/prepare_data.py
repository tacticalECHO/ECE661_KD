"""CIFAR-100 loaders for Experiment C.

The original 50,000-image training set is deterministically divided into a
45,000-image training split and a 5,000-image model-selection split. Separate
dataset objects ensure that augmentation is used only for training.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

try:
    from torchvision import datasets, transforms
except ModuleNotFoundError:
    datasets = None
    transforms = None


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _require_torchvision():
    if datasets is None or transforms is None:
        raise RuntimeError(
            "torchvision is required for CIFAR-100. Install the torchvision "
            "version compatible with the installed PyTorch build."
        )


def make_split_indices(dataset_size=50_000, val_size=5_000, split_seed=42):
    if dataset_size <= 1:
        raise ValueError("dataset_size must be greater than one")
    if not 0 < val_size < dataset_size:
        raise ValueError("val_size must be between zero and dataset_size")

    generator = torch.Generator().manual_seed(split_seed)
    shuffled = torch.randperm(dataset_size, generator=generator).tolist()
    split_at = dataset_size - val_size
    return shuffled[:split_at], shuffled[split_at:]


def build_transforms():
    _require_torchvision()
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return train_transform, eval_transform


def build_loaders(
    batch_size=64,
    seed=42,
    split_seed=42,
    val_size=5_000,
    num_workers=0,
    download=True,
    pin_memory=None,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    train_transform, eval_transform = build_transforms()
    augmented_train = datasets.CIFAR100(
        root=DATA_DIR,
        train=True,
        download=download,
        transform=train_transform,
    )
    evaluation_train = datasets.CIFAR100(
        root=DATA_DIR,
        train=True,
        download=download,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR100(
        root=DATA_DIR,
        train=False,
        download=download,
        transform=eval_transform,
    )

    train_indices, val_indices = make_split_indices(
        dataset_size=len(augmented_train),
        val_size=val_size,
        split_seed=split_seed,
    )
    train_dataset = Subset(augmented_train, train_indices)
    val_dataset = Subset(evaluation_train, val_indices)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train, val, test = build_loaders()
    images, labels = next(iter(train))
    print(
        "Train/validation/test:",
        len(train.dataset),
        len(val.dataset),
        len(test.dataset),
    )
    print("Batch:", tuple(images.shape), tuple(labels.shape))
    print("Label range:", labels.min().item(), labels.max().item())
