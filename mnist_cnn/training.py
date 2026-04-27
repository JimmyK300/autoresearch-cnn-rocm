"""Training entrypoint for the MNIST CNN artifact."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import trange

from mnist_cnn.config import load_config
from mnist_cnn.data import load_mnist
from mnist_cnn.evaluation import evaluate
from mnist_cnn.metrics import geometric_mean_loss
from mnist_cnn.model import CNN
from mnist_cnn.runs import append_research_log, update_best_run


def select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(model_config: dict[str, Any]) -> CNN:
    return CNN(
        in_channels=int(model_config["in_channels"]),
        conv_channels=list(model_config["conv_channels"]),
        kernel_sizes=list(model_config["kernel_sizes"]),
        paddings=list(model_config["paddings"]),
        hidden_dims=list(model_config["hidden_dims"]),
        num_classes=int(model_config["num_classes"]),
    )


def train(config_path: str | Path, hypothesis: str = "") -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_config(config_path)
    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]
    artifact_config = config["artifacts"]

    seed = int(training_config["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    device = select_device()

    started_at = time.time()
    model = build_model(model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training_config["learning_rate"])
    )

    train_loader, val_loader, train_size, _val_size = load_mnist(
        batch_size=int(training_config["batch_size"]),
        samples_dir=data_config["samples_dir"],
    )

    loss_history: list[float] = []
    train_accuracy_history: list[float] = []
    val_loss_history: list[float] = []
    val_accuracy_history: list[float] = []
    training_started_at = time.time()
    eval_seconds = 0.0

    for _epoch in trange(1, int(training_config["epochs"]) + 1, desc="Training"):
        model.train()
        running_loss = 0.0
        correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss_value = loss.item()
            if math.isnan(loss_value) or loss_value > 100:
                raise RuntimeError("Training failed because loss exploded or became NaN.")

            loss.backward()
            optimizer.step()
            running_loss += loss_value * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

        loss_history.append(running_loss / train_size)
        train_accuracy_history.append(correct / train_size)

        eval_started_at = time.time()
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        eval_seconds += time.time() - eval_started_at
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_acc)

    training_seconds = time.time() - training_started_at - eval_seconds
    train_loss = float(loss_history[-1])
    train_acc = float(train_accuracy_history[-1])
    val_loss = float(val_loss_history[-1])
    val_acc = float(val_accuracy_history[-1])
    score = geometric_mean_loss(train_loss, val_loss)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = Path(artifact_config["results_dir"])
    best_run_path = Path(artifact_config["best_run_path"])
    research_log_path = Path(artifact_config["research_log_path"])
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{run_id}.json"

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device == "cuda" else 0.0
    )
    num_params = sum(parameter.numel() for parameter in model.parameters())
    summary: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": run_id,
        "config_path": config_path.as_posix(),
        "device": device,
        "hypothesis": hypothesis,
        "seed": seed,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "geometric_mean_loss": score,
        "training_seconds": training_seconds,
        "total_seconds": time.time() - started_at,
        "peak_vram_mb": peak_vram_mb,
        "num_params": num_params,
        "num_params_m": num_params / 1e6,
        "config": config,
        "loss_history": loss_history,
        "train_accuracy_history": train_accuracy_history,
        "val_loss_history": val_loss_history,
        "val_accuracy_history": val_accuracy_history,
        "results_json": result_path.as_posix(),
        "is_best_run": False,
    }

    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    is_best_run, best = update_best_run(best_run_path, summary)
    summary["is_best_run"] = is_best_run
    summary["best_run_id"] = best["run_id"]
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_research_log(research_log_path, summary)
    print_summary(summary, best_run_path)
    return summary


def print_summary(summary: dict[str, Any], best_run_path: Path) -> None:
    print("---")
    print(f"run_id:              {summary['run_id']}")
    print(f"config_path:         {summary['config_path']}")
    print(f"hypothesis:          {summary['hypothesis']}")
    print(f"device:              {summary['device']}")
    print(f"train_loss:          {summary['train_loss']:.6f}")
    print(f"train_acc:           {summary['train_acc']:.6f}")
    print(f"val_acc:             {summary['val_acc']:.6f}")
    print(f"val_loss:            {summary['val_loss']:.6f}")
    print(f"geometric_mean_loss: {summary['geometric_mean_loss']:.6f}")
    print(f"training_seconds:    {summary['training_seconds']:.1f}")
    print(f"total_seconds:       {summary['total_seconds']:.1f}")
    print(f"peak_vram_mb:        {summary['peak_vram_mb']:.1f}")
    print(f"num_params_m:        {summary['num_params_m']:.3f}")
    print(f"results_json:        {summary['results_json']}")
    print(f"is_best_run:         {summary['is_best_run']}")
    print(f"best_run_json:       {best_run_path.as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MNIST CNN experiment.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--hypothesis", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args.config, args.hypothesis)
