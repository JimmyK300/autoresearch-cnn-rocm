#!/usr/bin/env python3
"""Evaluation entrypoint for autoresearch-cnn-rocm."""
import argparse
from pathlib import Path

# different evaluation for mnist
import torch

def EvaluateMnist(model, device, criterion, test_images, test_labels, test_size):
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (X, y) in enumerate(zip(test_images, test_labels)):
            X, y = X.unsqueeze(0).to(device), y.unsqueeze(0).to(device)

            output = model(X)

            loss = criterion(output, y)

            total_loss += loss.item() * X.size(0)

            pred = output.argmax(dim=1)
            correct += pred.eq(y.view_as(pred)).sum().item()

            
    avg_loss = total_loss / test_size
    accuracy = correct / test_size
    return avg_loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--model", required=False, default="latest")
    args = parser.parse_args()
    print(f"Evaluating model: {args.model}")


if __name__ == "__main__":
    main()
