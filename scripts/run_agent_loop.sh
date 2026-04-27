#!/usr/bin/env bash
set -euo pipefail

while true; do
  python3 -m mnist_cnn train --config configs/default.yaml
  python3 -m mnist_cnn evaluate --run latest
  sleep 1
done
