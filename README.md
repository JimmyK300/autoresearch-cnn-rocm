# autoresearch-cnn-rocm

Config-driven MNIST CNN experiments with simple run tracking and agent-friendly commands.

The current `main` branch is intentionally narrow: train and evaluate a CNN on MNIST, record each run, and make small config sweeps easy to repeat. ROCm support is a later differentiator; for now the code uses PyTorch's normal `torch.cuda` check when a GPU is available and otherwise runs on CPU.

## Why This Shape

The repo is designed so humans and AI agents mostly edit configuration instead of rewriting training code. The core Python package owns data loading, model construction, metrics, run tracking, and sweeps. PowerShell and Bash scripts stay available as thin convenience wrappers.

## Install

Use an environment with PyTorch and the MNIST `samples/` files available locally.

```powershell
py -m pip install -e .
```

The raw MNIST files are expected in `samples/`:

```text
samples/
  train-images-idx3-ubyte
  train-labels-idx1-ubyte
  t10k-images-idx3-ubyte
  t10k-labels-idx1-ubyte
```

## Commands

Train the baseline config:

```powershell
py train.py --config configs/default.yaml
```

Or use the package command:

```powershell
py -m mnist_cnn train --config configs/default.yaml
```

Show the latest run:

```powershell
py evaluate.py --run latest
```

Show the current best run:

```powershell
py -m mnist_cnn evaluate --run best
```

Run a config sweep:

```powershell
py -m mnist_cnn sweep --config configs/default.yaml --search-plan configs/search_plan.json
```

Check PyTorch device visibility:

```powershell
py -m mnist_cnn check-device
```

Existing wrappers remain available:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_agent_loop.ps1
powershell -ExecutionPolicy Bypass -File scripts/auto_sweep.ps1
```

## Project Layout

- `mnist_cnn/`: shared implementation for config, data, model, training, evaluation, runs, and sweeps.
- `train.py`: familiar training entrypoint.
- `evaluate.py`: familiar saved-run reporting entrypoint.
- `prepare.py`: local MNIST sample loader check.
- `metrics.py`: public metric helpers.
- `configs/default.yaml`: baseline CNN/training configuration.
- `configs/search_plan.json`: small sweep candidates.
- `scripts/`: shell and PowerShell convenience wrappers.

Generated outputs such as `results/`, `best_run.json`, `research_log.md`, `weights/`, and submission packages are local artifacts and are ignored on `main`.

## Optimization Target

The default comparison objective is:

```text
geometric_mean_loss = sqrt(train_loss * val_loss)
```

Lower is better. Ties are resolved by lower validation loss, then higher validation accuracy, then shorter training time.
