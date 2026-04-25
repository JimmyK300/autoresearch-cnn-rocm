#!/usr/bin/env python3

import torch
from torch import nn
from prepare import PrepareMnist
from evaluate import EvaluateMnist

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

import json
import copy
import time
import os
import math

# ---------------------------------------------------------------------------
# CNN Model
# ---------------------------------------------------------------------------


class CNN(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_channels,
        kernel_sizes,
        paddings,
        hidden_dims,
        num_classes,
    ):
        super().__init__()

        layers = []
        current_channels = in_channels

        # Build conv stack dynamically
        for out_channels, k, p in zip(conv_channels, kernel_sizes, paddings):
            layers.append(
                nn.Conv2d(current_channels, out_channels, kernel_size=k, padding=p)
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            current_channels = out_channels

        self.features = nn.Sequential(*layers)

        self.flatten = nn.Flatten()

        # Linear layer
        self.linearDim = None
        self._infer_flatten_size()
        linearLayers = []

        for dim in hidden_dims:
            linearLayers.append(nn.Linear(self.linearDim, dim)),
            linearLayers.append(nn.ReLU())
            self.linearDim = dim
        linearLayers.append(nn.Linear(self.linearDim, num_classes))

        self.classifier = nn.Sequential(*linearLayers)

    def _infer_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)
            x = self.features(x)
            self.linearDim = x.numel()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Optimizer: torch.optim.AdamW
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

HIDDEN_DIM_L1 = 128
HIDDEN_DIM_L2 = 32

CONV_CHANNEL_L1 = 32
CONV_CHANNEL_L2 = 32

KERNEL_SIZE_L1 = 5
KERNEL_SIZE_L2 = 3

PADDING_L1 = 2
PADDING_L2 = 1

DEVICE_BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

# ---------------------------------------------------------------------------
# CONSTANT (DO NOT edit these)
# ---------------------------------------------------------------------------
IN_CHANNEL = 1
NUM_CLASS = 10
# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()

# Check device (torch also uses cuda to check amd GPU)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
H100_BF16_PEAK_FLOPS = 989.5e12

# Init model
model = CNN(
    IN_CHANNEL,
    [CONV_CHANNEL_L1, CONV_CHANNEL_L2],
    [KERNEL_SIZE_L1, KERNEL_SIZE_L2],
    [PADDING_L1, PADDING_L2],
    [HIDDEN_DIM_L1, HIDDEN_DIM_L2],
    NUM_CLASS,
).to(device)

# Init criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Data preprocessing
(train_dataloader, test_dataloader, train_size, test_size, test_images, test_labels) = PrepareMnist(
    DEVICE_BATCH_SIZE
)

# ---------------------------------------------------------------------------
# Train Loop
# ---------------------------------------------------------------------------

t_start_training = time.time()

t_eval_during_train = 0

loss_history = []
test_loss_history = []
accuracy_history = []
train_accuracy_history = []

for epoch in trange(1, EPOCHS + 1, desc=f"Training progress"):
    model.train()

    running_loss = 0
    train_correct = 0
    for batch, (X, y) in enumerate(train_dataloader):
        optimizer.zero_grad()

        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = criterion(pred, y)
        
        loss_f = loss.item()
        
        # Fast fail: abort if loss is exploding or NaN
        if math.isnan(loss_f) or loss_f > 100:
            print("FAIL")
            exit(1)
        
        loss.backward()
        optimizer.step()

        running_loss += loss_f * X.size(0)
        train_correct += (pred.argmax(1) == y).sum().item()
    loss_history.append(running_loss / train_size)
    train_accuracy_history.append(train_correct / train_size)

    t_start_eval_during_train = time.time()

    model.eval()

    test_running_loss = 0
    correct = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            test_running_loss += loss.item() * X.size(0)
            correct += (pred.argmax(1) == y).sum().item()

        test_loss_history.append(test_running_loss / test_size)
        accuracy_history.append(correct / test_size)
    
    t_end_eval_during_train = time.time()
    
    t_eval_during_train += t_end_eval_during_train - t_start_eval_during_train

t_end_training = time.time()

total_training_time = t_end_training - t_start_training - t_eval_during_train
train_acc = np.sum(train_accuracy_history) / len(train_accuracy_history)



# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

model.eval()
val_loss, val_acc = EvaluateMnist(model, device, criterion, test_images, test_labels, test_size)



# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
num_params = sum(p.numel() for p in model.parameters())

print("---")
print(f"train_acc:        {train_acc:.6f}")
print(f"val_acc:          {val_acc:.6f}")
print(f"val_loss:         {val_loss:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
# print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
