"""CNN architecture for MNIST."""

from __future__ import annotations

import torch
from torch import nn


class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        conv_channels: list[int],
        kernel_sizes: list[int],
        paddings: list[int],
        hidden_dims: list[int],
        num_classes: int,
    ) -> None:
        super().__init__()
        if not (len(conv_channels) == len(kernel_sizes) == len(paddings)):
            raise ValueError("conv_channels, kernel_sizes, and paddings must align.")

        layers: list[nn.Module] = []
        current_channels = in_channels
        for out_channels, kernel_size, padding in zip(
            conv_channels, kernel_sizes, paddings
        ):
            layers.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            current_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        flattened_dim = self._infer_flatten_size(in_channels)

        classifier_layers: list[nn.Module] = []
        current_dim = flattened_dim
        for hidden_dim in hidden_dims:
            classifier_layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            current_dim = hidden_dim
        classifier_layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def _infer_flatten_size(self, in_channels: int) -> int:
        with torch.no_grad():
            sample = torch.zeros(1, in_channels, 28, 28)
            return int(self.features(sample).numel())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(self.flatten(features))
