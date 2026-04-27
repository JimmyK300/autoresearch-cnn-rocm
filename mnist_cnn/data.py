"""MNIST data loading."""

from __future__ import annotations

from pathlib import Path

import torch
from mnist import MNIST
from torch.utils.data import DataLoader, Dataset


class MnistTensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def _to_image_tensor(images: list[list[int]]) -> torch.Tensor:
    return torch.tensor(images, dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0


def load_mnist(
    batch_size: int,
    samples_dir: str | Path = "samples",
) -> tuple[DataLoader, DataLoader, int, int]:
    samples_path = Path(samples_dir)
    required_files = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    missing = [name for name in required_files if not (samples_path / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing MNIST sample files in {samples_path.as_posix()}: {joined}"
        )

    mnist = MNIST(samples_path.as_posix())
    train_images, train_labels = mnist.load_training()
    test_images, test_labels = mnist.load_testing()

    train_dataset = MnistTensorDataset(
        _to_image_tensor(train_images), torch.tensor(train_labels, dtype=torch.long)
    )
    test_dataset = MnistTensorDataset(
        _to_image_tensor(test_images), torch.tensor(test_labels, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader, len(train_dataset), len(test_dataset)


def prepare_data(batch_size: int = 64, samples_dir: str | Path = "samples"):
    """Compatibility wrapper for loading prepared MNIST files."""
    return load_mnist(batch_size=batch_size, samples_dir=samples_dir)
