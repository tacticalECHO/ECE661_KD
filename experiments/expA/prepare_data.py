
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# MNIST is shared across experiments and stored once at the repo root.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def build_loaders(
    batch_size=128,
    seed=42,
    augment=False,
    exclude_classes=None,
    split_seed=42
):

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=0,
                translate=(2 / 28, 2 / 28)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = eval_transform

    full_train = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )

    full_val = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=eval_transform
    )

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=eval_transform
    )

    generator = torch.Generator().manual_seed(split_seed)

    indices = torch.randperm(
        len(full_train),
        generator=generator
    ).tolist()

    train_indices = indices[:55000]
    val_indices = indices[55000:]

    if exclude_classes is not None:

        exclude_classes = set(exclude_classes)

        train_indices = [
            idx for idx in train_indices
            if int(full_train.targets[idx])
            not in exclude_classes
        ]
        val_indices = [
            idx for idx in val_indices
            if int(full_val.targets[idx])
            not in exclude_classes
        ]

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
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
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
