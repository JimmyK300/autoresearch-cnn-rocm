#!/usr/bin/env python3
"""Train the MNIST CNN experiment."""

from mnist_cnn.model import CNN
from mnist_cnn.training import main, train

__all__ = ["CNN", "train"]


if __name__ == "__main__":
    main()
