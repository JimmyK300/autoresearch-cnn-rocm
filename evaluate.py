#!/usr/bin/env python3
"""Evaluate MNIST CNN models and saved run summaries."""

from mnist_cnn.evaluation import evaluate, evaluate_model
from mnist_cnn.runs import main

__all__ = ["evaluate", "evaluate_model"]


if __name__ == "__main__":
    main()
