"""Metrics used by the MNIST CNN artifact."""

from __future__ import annotations

import math


def geometric_mean_loss(train_loss: float, val_loss: float) -> float:
    train_loss = float(train_loss)
    val_loss = float(val_loss)
    if train_loss < 0 or val_loss < 0:
        raise ValueError("Loss values must be non-negative.")
    return math.sqrt(train_loss * val_loss)
