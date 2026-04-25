#!/usr/bin/env bash
set -euo pipefail

while true; do
  python3 train.py --config configs/default.yaml
  python3 evaluate.py --model latest
  sleep 1
done
