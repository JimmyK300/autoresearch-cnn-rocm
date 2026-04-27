# Agent Operating Contract

This branch is a front-facing MNIST CNN artifact. Keep it boring, legible, and reproducible.

## Mission

- Train and evaluate a CNN on MNIST.
- Prefer small config-driven experiments over code rewrites.
- Keep generated outputs local and out of the final repo.
- Preserve CPU-first behavior; use `cuda` only when PyTorch reports it is available.

## Editing Rules

- Prefer editing `configs/default.yaml` and `configs/search_plan.json`.
- Do not casually edit model, data, metric, or run-tracking code unless there is a clear bug or cleanup reason.
- Keep each file purposeful. If two files do the same job, merge the behavior or turn one into a wrapper.
- Do not reintroduce writer-bias optimization on `main`; that work is archived on `archive/writer-bias-current`.

## Experiment Loop

1. State the hypothesis.
2. Change one config value or one small search-plan group.
3. Run `py -m mnist_cnn train --config configs/default.yaml` or `py -m mnist_cnn sweep`.
4. Compare against `best_run.json` using geometric mean loss.
5. Keep config changes only when the run improves the objective without obvious validation collapse.

## Required Outputs

Every training run should write:

- `results/<run_id>.json`
- `best_run.json`
- `research_log.md`

These are local artifacts and should remain ignored by Git on `main`.
