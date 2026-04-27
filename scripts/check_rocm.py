#!/usr/bin/env python3
"""Check PyTorch device visibility for CPU/CUDA/ROCm environments."""

from mnist_cnn.cli import check_device


if __name__ == "__main__":
    check_device()
