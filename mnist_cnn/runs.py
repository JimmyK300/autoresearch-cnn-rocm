"""Run artifact reading, comparison, and reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mnist_cnn.metrics import geometric_mean_loss


def load_json(path: str | Path) -> dict[str, Any]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path.as_posix()}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def objective_value(summary: dict[str, Any]) -> float:
    if "geometric_mean_loss" in summary:
        return float(summary["geometric_mean_loss"])
    if "train_loss" in summary and "val_loss" in summary:
        return geometric_mean_loss(summary["train_loss"], summary["val_loss"])
    raise KeyError("Run summary is missing loss fields needed for comparison.")


def is_better_run(candidate: dict[str, Any], current_best: dict[str, Any] | None) -> bool:
    if current_best is None:
        return True

    candidate_objective = objective_value(candidate)
    current_objective = objective_value(current_best)
    if candidate_objective != current_objective:
        return candidate_objective < current_objective
    if candidate["val_loss"] != current_best["val_loss"]:
        return candidate["val_loss"] < current_best["val_loss"]
    if candidate["val_acc"] != current_best["val_acc"]:
        return candidate["val_acc"] > current_best["val_acc"]
    return candidate["training_seconds"] < current_best["training_seconds"]


def best_payload(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "run_id",
        "timestamp_utc",
        "config_path",
        "device",
        "hypothesis",
        "train_loss",
        "train_acc",
        "val_acc",
        "val_loss",
        "geometric_mean_loss",
        "training_seconds",
        "total_seconds",
        "peak_vram_mb",
        "num_params",
        "num_params_m",
        "seed",
        "results_json",
    ]
    return {key: summary[key] for key in keys if key in summary}


def update_best_run(best_run_path: str | Path, summary: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    path = Path(best_run_path)
    current_best = load_json(path) if path.exists() else None
    if is_better_run(summary, current_best):
        payload = best_payload(summary)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True, payload
    return False, current_best


def resolve_run(
    selector: str,
    results_dir: str | Path = "results",
    best_run_path: str | Path = "best_run.json",
) -> tuple[Path, dict[str, Any]]:
    results_path = Path(results_dir)
    best_path = Path(best_run_path)

    if selector == "latest":
        result_files = sorted(
            results_path.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not result_files:
            raise FileNotFoundError(f"No result files found in {results_path.as_posix()}")
        selected_path = result_files[0]
        return selected_path, load_json(selected_path)

    if selector == "best":
        best_summary = load_json(best_path)
        result_file = best_summary.get("results_json")
        if result_file:
            selected_path = Path(result_file)
            if not selected_path.is_absolute():
                selected_path = Path.cwd() / selected_path
            if selected_path.exists():
                return selected_path, load_json(selected_path)
        fallback = results_path / f"{best_summary['run_id']}.json"
        if fallback.exists():
            return fallback, load_json(fallback)
        return best_path, best_summary

    selected_path = Path(selector)
    if selected_path.exists():
        return selected_path, load_json(selected_path)

    run_id = selected_path.stem if selector.endswith(".json") else selector
    selected_path = results_path / f"{run_id}.json"
    return selected_path, load_json(selected_path)


def append_research_log(log_path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(log_path)
    lines = [
        "",
        f"## {summary['run_id']}",
        f"- config: `{summary['config_path']}`",
        f"- hypothesis: `{summary['hypothesis']}`",
        f"- device: `{summary['device']}`",
        f"- train_loss: `{summary['train_loss']:.6f}`",
        f"- train_acc: `{summary['train_acc']:.6f}`",
        f"- val_acc: `{summary['val_acc']:.6f}`",
        f"- val_loss: `{summary['val_loss']:.6f}`",
        f"- geometric_mean_loss: `{summary['geometric_mean_loss']:.6f}`",
        f"- training_seconds: `{summary['training_seconds']:.1f}`",
        f"- total_seconds: `{summary['total_seconds']:.1f}`",
        f"- num_params_m: `{summary['num_params_m']:.3f}`",
        f"- is_best_run: `{summary['is_best_run']}`",
        f"- results_json: `{summary['results_json']}`",
    ]
    if not path.exists():
        path.write_text("# Research Log\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _as_float(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _print_metric(label: str, value: float | str | None, fmt: str = "{:.6f}") -> None:
    if value is None:
        print(f"{label:<21} n/a")
    elif isinstance(value, float):
        print(f"{label:<21} {fmt.format(value)}")
    else:
        print(f"{label:<21} {value}")


def _print_delta(label: str, value: float | None, lower_is_better: bool) -> None:
    if value is None:
        print(f"{label:<21} n/a")
        return
    if value == 0:
        print(f"{label:<21} {value:+.6f} (equal)")
        return
    better = (value < 0 and lower_is_better) or (value > 0 and not lower_is_better)
    print(f"{label:<21} {value:+.6f} ({'better' if better else 'worse'})")


def print_run_summary(
    selector: str,
    results_dir: str | Path = "results",
    best_run_path: str | Path = "best_run.json",
    print_json: bool = False,
) -> None:
    selected_path, summary = resolve_run(selector, results_dir, best_run_path)
    print(f"Selected run source: {selected_path.as_posix()}")
    print("Selection objective: geometric_mean_loss (lower is better)")
    print("---")
    _print_metric("run_id", summary.get("run_id"))
    _print_metric("timestamp_utc", summary.get("timestamp_utc"))
    _print_metric("config_path", summary.get("config_path"))
    _print_metric("hypothesis", summary.get("hypothesis"))

    objective = objective_value(summary)
    train_loss = _as_float(summary, "train_loss")
    train_acc = _as_float(summary, "train_acc")
    val_acc = _as_float(summary, "val_acc")
    val_loss = _as_float(summary, "val_loss")
    _print_metric("objective", objective)
    _print_metric("train_loss", train_loss)
    _print_metric("train_acc", train_acc)
    _print_metric("val_acc", val_acc)
    _print_metric("val_loss", val_loss)
    _print_metric("training_seconds", _as_float(summary, "training_seconds"), "{:.1f}")
    _print_metric("total_seconds", _as_float(summary, "total_seconds"), "{:.1f}")
    _print_metric("is_best_run", str(summary.get("is_best_run")))

    best_path = Path(best_run_path)
    if best_path.exists():
        best_summary = load_json(best_path)
        best_objective = objective_value(best_summary)
        print("---")
        print(f"Current best run:      {best_summary.get('run_id', 'n/a')}")
        _print_metric("best_objective", best_objective)
        _print_metric("best_val_acc", _as_float(best_summary, "val_acc"))
        _print_metric("best_val_loss", _as_float(best_summary, "val_loss"))
        print("---")
        print("Delta vs best run")
        _print_delta("objective_delta", objective - best_objective, lower_is_better=True)
        best_val_acc = _as_float(best_summary, "val_acc")
        best_val_loss = _as_float(best_summary, "val_loss")
        _print_delta(
            "val_acc_delta",
            val_acc - best_val_acc if val_acc is not None and best_val_acc is not None else None,
            lower_is_better=False,
        )
        _print_delta(
            "val_loss_delta",
            val_loss - best_val_loss if val_loss is not None and best_val_loss is not None else None,
            lower_is_better=True,
        )

    if print_json:
        print("---")
        print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show MNIST CNN run metrics.")
    parser.add_argument("--run", default=None, help="latest, best, run_id, or result JSON path.")
    parser.add_argument("--model", dest="legacy_model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--best-run", default="best_run.json")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selector = (args.run or args.legacy_model or "latest").strip()
    print_run_summary(selector, args.results_dir, args.best_run, args.json)
