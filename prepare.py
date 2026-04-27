#!/usr/bin/env python3
"""Load prepared MNIST files from the local samples directory."""

from mnist_cnn.data import MnistTensorDataset, load_mnist, prepare_data

__all__ = ["MnistTensorDataset", "load_mnist", "prepare_data"]


def main() -> None:
    train_loader, test_loader, train_size, test_size = prepare_data()
    print(f"train_size: {train_size}")
    print(f"test_size: {test_size}")
    print(f"train_batches: {len(train_loader)}")
    print(f"test_batches: {len(test_loader)}")


if __name__ == "__main__":
    main()
