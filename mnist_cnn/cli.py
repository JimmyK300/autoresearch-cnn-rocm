"""Command-line interface for the MNIST CNN artifact."""

from __future__ import annotations

import argparse

import torch

from mnist_cnn import __version__
from mnist_cnn.runs import print_run_summary
from mnist_cnn.sweep import run_sweep
from mnist_cnn.training import train


def check_device() -> None:
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("hip version:", getattr(torch.version, "hip", None))
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
    else:
        print("device: cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mnist-cnn", description="MNIST CNN experiments.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train from a config file.")
    train_parser.add_argument("--config", default="configs/default.yaml")
    train_parser.add_argument("--hypothesis", default="")

    eval_parser = subparsers.add_parser("evaluate", help="Show saved run metrics.")
    eval_parser.add_argument("--run", default="latest")
    eval_parser.add_argument("--results-dir", default="results")
    eval_parser.add_argument("--best-run", default="best_run.json")
    eval_parser.add_argument("--json", action="store_true")

    sweep_parser = subparsers.add_parser("sweep", help="Run a config search plan.")
    sweep_parser.add_argument("--config", default="configs/default.yaml")
    sweep_parser.add_argument("--search-plan", default="configs/search_plan.json")
    sweep_parser.add_argument("--pause-seconds", type=int, default=1)

    subparsers.add_parser("check-device", help="Show PyTorch CPU/GPU visibility.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        train(args.config, args.hypothesis)
    elif args.command == "evaluate":
        print_run_summary(args.run, args.results_dir, args.best_run, args.json)
    elif args.command == "sweep":
        run_sweep(args.config, args.search_plan, args.pause_seconds)
    elif args.command == "check-device":
        check_device()
