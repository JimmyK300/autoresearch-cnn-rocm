#!/usr/bin/env python3
"""Data preparation for autoresearch-cnn-rocm."""
import argparse
from pathlib import Path

# Exclusive data prepare for MNIST
from mnist import MNIST
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform  # transform modify the image
        self.target_transform = target_transform  # target_transform modify the label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def PrepareMnist(batch_size):
    mndata = MNIST("samples")

    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    train_size = len(train_images)
    test_size = len(test_images)

    train_images = (
        torch.tensor(train_images, dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0
    )
    train_labels = torch.tensor(train_labels, dtype=torch.long)

    test_images = (
        torch.tensor(test_images, dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0
    )
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    training_dataset = CustomImageDataset(train_images, train_labels)
    testing_dataset = CustomImageDataset(test_images, test_labels)

    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_dataset, batch_size=batch_size)

    return (
        train_dataloader,
        test_dataloader,
        train_size,
        test_size,
        test_images,
        test_labels,
    )


def prepare(input_dir: str, output_dir: str) -> None:
    """Placeholder: copy/transform raw data into processed directory."""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing data from {in_dir} -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    prepare(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
