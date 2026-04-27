"""Configuration loading and writing helpers."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "in_channels": 1,
        "conv_channels": [32, 32],
        "kernel_sizes": [5, 5],
        "paddings": [2, 1],
        "hidden_dims": [128, 32],
        "num_classes": 10,
    },
    "training": {
        "batch_size": 64,
        "learning_rate": 0.0008,
        "epochs": 7,
        "seed": 42,
    },
    "data": {
        "samples_dir": "samples",
    },
    "artifacts": {
        "results_dir": "results",
        "best_run_path": "best_run.json",
        "research_log_path": "research_log.md",
    },
}


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")

        while stack and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        if value.strip() == "":
            current[key] = {}
            stack.append((indent, current[key]))
        else:
            current[key] = parse_scalar(value)

    return root


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(DEFAULT_CONFIG)
    model = config.get("model", {})
    training = config.get("training", {})
    data = config.get("data", {})
    artifacts = config.get("artifacts", {})

    normalized["model"].update(
        {
            "in_channels": model.get("in_channels", 1),
            "conv_channels": model.get(
                "conv_channels",
                [
                    model.get("CONV_CHANNEL_L1", 32),
                    model.get("CONV_CHANNEL_L2", 32),
                ],
            ),
            "kernel_sizes": model.get(
                "kernel_sizes",
                [
                    model.get("KERNEL_SIZE_L1", 5),
                    model.get("KERNEL_SIZE_L2", 5),
                ],
            ),
            "paddings": model.get(
                "paddings",
                [
                    model.get("PADDING_L1", 2),
                    model.get("PADDING_L2", 1),
                ],
            ),
            "hidden_dims": model.get(
                "hidden_dims",
                [
                    model.get("HIDDEN_DIM_L1", 128),
                    model.get("HIDDEN_DIM_L2", 32),
                ],
            ),
            "num_classes": model.get("num_classes", 10),
        }
    )
    normalized["training"].update(
        {
            "batch_size": training.get(
                "batch_size", training.get("DEVICE_BATCH_SIZE", 64)
            ),
            "learning_rate": training.get(
                "learning_rate", training.get("LEARNING_RATE", 0.0008)
            ),
            "epochs": training.get("epochs", training.get("EPOCHS", 7)),
            "seed": training.get("seed", 42),
        }
    )
    normalized["data"].update(
        {
            "samples_dir": data.get(
                "samples_dir", data.get("raw_dir", DEFAULT_CONFIG["data"]["samples_dir"])
            )
        }
    )
    normalized["artifacts"].update(artifacts)
    return normalized


def load_config(path: str | Path) -> dict[str, Any]:
    return normalize_config(load_simple_yaml(path))


def format_yaml_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(str(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_simple_yaml(config: dict[str, Any], path: str | Path) -> None:
    lines: list[str] = []
    for section, values in config.items():
        lines.append(f"{section}:")
        for key, value in values.items():
            lines.append(f"  {key}: {format_yaml_value(value)}")
        lines.append("")
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def set_nested_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current: dict[str, Any] = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value
