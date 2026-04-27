"""Config sweep orchestration."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from mnist_cnn.config import load_config, parse_scalar, set_nested_value, write_simple_yaml
from mnist_cnn.training import train


def load_search_plan(path: str | Path) -> list[dict[str, Any]]:
    plan_path = Path(path)
    if not plan_path.exists():
        raise FileNotFoundError(f"Search plan not found: {plan_path.as_posix()}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, list) or not plan:
        raise ValueError("Search plan must be a non-empty JSON list.")
    return plan


def run_sweep(
    config_path: str | Path = "configs/default.yaml",
    search_plan_path: str | Path = "configs/search_plan.json",
    pause_seconds: int = 1,
) -> None:
    config_path = Path(config_path)
    plan_items = load_search_plan(search_plan_path)

    for experiment in plan_items:
        key = experiment["key"]
        reason = experiment.get("reason", "")
        candidates = experiment.get("candidates", [])
        for raw_candidate in candidates:
            original_config = load_config(config_path)
            candidate_config = deepcopy(original_config)
            candidate = parse_scalar(str(raw_candidate))
            set_nested_value(candidate_config, key, candidate)
            write_simple_yaml(candidate_config, config_path)

            print("")
            print(f"=== Trying {key} -> {raw_candidate} ===")
            if reason:
                print(f"Reason: {reason}")

            hypothesis = f"config sweep: set {key} to {raw_candidate}"
            summary = train(config_path=config_path, hypothesis=hypothesis)
            if not summary["is_best_run"]:
                write_simple_yaml(original_config, config_path)
                print(f"Reverted {key}; run did not improve the objective.")
            else:
                print(f"Kept {key}; run is the new best objective.")

            if pause_seconds > 0:
                time.sleep(pause_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a config sweep for MNIST CNN.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--search-plan", default="configs/search_plan.json")
    parser.add_argument("--pause-seconds", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sweep(args.config, args.search_plan, args.pause_seconds)
