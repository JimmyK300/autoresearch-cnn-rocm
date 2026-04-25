# Mission

This repo has two phases.

Phase 1 is the minimal baseline:

- get a plain MNIST CNN to train successfully
- report train accuracy, validation accuracy, and validation loss
- keep the code path simple and stable on AMD ROCm

Phase 2 is the research phase:

- improve CNN performance while reducing writer bias
- add writer-aware metrics and selection criteria
- run a disciplined experiment loop

The agent should optimize for completing Phase 1 first unless the human explicitly says to enter Phase 2.

The agent may edit only `train.py` unless explicitly told otherwise.

# Hardware Target

This repo targets AMD GPUs through PyTorch ROCm.
Under ROCm, PyTorch usually exposes the AMD GPU through the CUDA-compatible API surface.

Use these checks:

- use `torch.cuda.is_available()` to confirm the GPU is visible to PyTorch
- use `device = "cuda"` when that check is true
- otherwise fall back to CPU without crashing

Before spending time on model tuning, verify the runtime first with `scripts/check_rocm.py`.

# Phase 1 Rules

Phase 1 is intentionally narrow. The goal is one boring, repeatable baseline run.

Required metrics in Phase 1:

- train accuracy
- validation accuracy
- validation loss
- training time

Acceptable changes in Phase 1:

- learning rate
- batch size
- weight decay
- CNN depth
- CNN width
- batch normalization
- dropout
- evaluation/reporting cleanup
- ROCm-safe device handling

Do not add Phase 2 complexity yet:

- writer metrics
- group-balanced sampling
- writer-aware loss reweighting
- fairness-based acceptance rules
- large multi-file rewrites

# Phase 2 Rules

Enter Phase 2 only after Phase 1 is stable and reproducible.

Required metrics in Phase 2:

- train accuracy
- validation accuracy
- validation loss
- writer_gap = max writer accuracy - min writer accuracy
- min_writer_acc
- max_writer_acc
- mean_writer_acc
- generalization_gap = train_acc - val_acc
- research_score

Optimization rule in Phase 2:

Prefer changes that improve `research_score`.

Do not accept a change if:

- validation accuracy collapses
- writer_gap improves only because all writers become bad
- min_writer_acc gets worse
- generalization_gap becomes much larger
- the result cannot be reproduced across at least 2 seeds after a major improvement

# Experiment Loop

For every experiment:

1. State the hypothesis.
2. Modify only `train.py`.
3. Run the training command.
4. Record metrics.
5. Compare against current best for the active phase.
6. Keep the change only if it improves the active phase objective.
7. Append result to `research_log.md`.

# First Experiment Ideas

Use this order:

Phase 1 first:

- safe GPU detection and fallback
- stable loss/accuracy reporting
- learning rate
- batch size
- CNN width
- CNN depth
- dropout
- batch normalization

Only in Phase 2:

- group-balanced sampling by writer
- loss reweighting for weak writers
- early stopping based on research_score

# Forbidden Initially

Do not:

- change the dataset split
- leak writer identity into the model input
- optimize only for average validation accuracy during Phase 2
- delete metrics that belong to the active phase
- make large multi-file rewrites
