#!/usr/bin/env bash
set -euo pipefail

python3 -m mnist_cnn train --config configs/default.yaml
