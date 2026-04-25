Yes. Build it as **`autoresearch-cnn-rocm`**: a Karpathy-style autonomous experiment loop, but for **CNN fairness/generalization experiments on AMD GPUs**.

The contribution angle is real because Karpathy’s original repo is deliberately LLM-focused, single-file, fixed-time-budget, and currently requires a single NVIDIA GPU; it uses `prepare.py`, editable `train.py`, and `program.md` as the human-written “research org code.” ([GitHub][1]) AMD support should target **PyTorch ROCm**, preferably through Docker first, since AMD’s ROCm docs recommend prebuilt PyTorch Docker images to avoid install problems. ([ROCm Documentation][2])

## 1. Repo concept

Name:

```text
autoresearch-cnn-rocm
```

Core promise:

```text
Autonomous CNN experiment loop for AMD/ROCm GPUs, with first-class support for group-level metrics such as writer accuracy disparity.
```

Not just “Karpathy but CNN.” The real differentiator is:

```text
optimize model quality under constraints:
- overall validation accuracy/loss
- writer fairness / writer bias
- generalization gap
- fixed wall-clock budget
- AMD ROCm compatibility
```

## 2. Architecture

Use this structure:

```text
autoresearch-cnn-rocm/
  README.md
  program.md
  pyproject.toml
  prepare.py
  train.py
  evaluate.py
  metrics.py
  research_log.md
  configs/
    default.yaml
    writer_bias.yaml
  data/
    raw/
    processed/
  scripts/
    check_rocm.py
    run_baseline.sh
    run_agent_loop.sh
  examples/
    mnist_writer_like/
    your_project_adapter/
```

Keep Karpathy’s key design:

```text
prepare.py  = stable data prep, not edited by agent
train.py    = agent-editable experiment surface
program.md  = human-written research instructions
metrics.py  = fixed scoring logic, not casually edited
```

Difference from Karpathy:

```text
CNN task, not LLM.
ROCm/AMD target, not NVIDIA-only.
Fairness metric, not only validation loss.
```

## 3. AMD GPU / ROCm layer

Use PyTorch ROCm. Important weird detail: even on AMD GPU, PyTorch often still uses the `torch.cuda` namespace as the generic GPU interface; AMD’s docs explicitly say `torch.cuda.is_available()` is the check for AMD GPU availability under ROCm. ([ROCm Documentation][2]) PyTorch on ROCm uses AMD libraries like MIOpen and RCCL for deep learning workloads. ([ROCm Documentation][2])

Add this file:

```python
# scripts/check_rocm.py
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("hip version:", getattr(torch.version, "hip", None))

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("matmul ok:", y.mean().item())
else:
    raise SystemExit("No ROCm/AMD GPU detected by PyTorch.")
```

Recommended Docker command:

```bash
docker run -it \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 8G \
  -v "$PWD:/workspace" \
  -w /workspace \
  rocm/pytorch:latest
```

AMD’s current docs list `rocm/pytorch:latest` and specific ROCm/PyTorch tags, with Docker as the recommended setup path. ([ROCm Documentation][2])

## 4. Writer-bias metric design

Your metric:

```text
writer_gap = max(writer_accuracy) - min(writer_accuracy)
```

Good, but incomplete. Add safeguards.

Use:

```python
# metrics.py
from collections import defaultdict
import numpy as np

def writer_metrics(y_true, y_pred, writer_ids, min_samples=5):
    correct_by_writer = defaultdict(int)
    total_by_writer = defaultdict(int)

    for yt, yp, wid in zip(y_true, y_pred, writer_ids):
        total_by_writer[wid] += 1
        correct_by_writer[wid] += int(yt == yp)

    acc_by_writer = {
        wid: correct_by_writer[wid] / total
        for wid, total in total_by_writer.items()
        if total >= min_samples
    }

    if not acc_by_writer:
        return {
            "writer_gap": None,
            "min_writer_acc": None,
            "max_writer_acc": None,
            "mean_writer_acc": None,
            "num_writers_used": 0,
        }

    values = list(acc_by_writer.values())

    return {
        "writer_gap": max(values) - min(values),
        "min_writer_acc": min(values),
        "max_writer_acc": max(values),
        "mean_writer_acc": float(np.mean(values)),
        "num_writers_used": len(values),
        "acc_by_writer": acc_by_writer,
    }
```

Then define a single accept/reject score:

```python
def research_score(val_acc, writer_gap, train_acc=None, lambda_gap=0.5, lambda_gen=0.25):
    if writer_gap is None:
        return -999

    gen_gap = 0.0
    if train_acc is not None:
        gen_gap = max(0.0, train_acc - val_acc)

    return val_acc - lambda_gap * writer_gap - lambda_gen * gen_gap
```

Meaning:

```text
Higher score is better.
Accuracy matters.
Writer disparity is punished.
Overfitting is punished.
```

This prevents the agent from “solving” writer bias by making the model equally bad for everyone.

## 5. `program.md` for your agent

Use this as the first version:

```md
# Mission

Improve CNN performance while reducing writer bias.

The agent may edit only `train.py` unless explicitly told otherwise.

# Hardware Target

This repo targets AMD GPUs through PyTorch ROCm.
Use `device = "cuda"` when `torch.cuda.is_available()` is true, because PyTorch ROCm exposes AMD GPUs through the CUDA-compatible namespace.

# Primary Metrics

Each run must report:

- train accuracy
- validation accuracy
- validation loss
- writer_gap = max writer accuracy - min writer accuracy
- min_writer_acc
- max_writer_acc
- mean_writer_acc
- generalization_gap = train_acc - val_acc
- research_score

# Optimization Rule

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
5. Compare against current best.
6. Keep the change only if research_score improves.
7. Append result to `research_log.md`.

# First Experiment Ideas

Try small, controlled changes first:

- learning rate
- batch size
- weight decay
- dropout
- data augmentation
- CNN depth
- CNN width
- batch normalization
- group-balanced sampling by writer
- loss reweighting for weak writers
- early stopping based on research_score

# Forbidden Initially

Do not:
- change the dataset split
- leak writer identity into the model input
- optimize only for average validation accuracy
- delete metrics
- make large multi-file rewrites
```

## 6. Your project-specific adapter

Your dataset needs to return this shape:

```python
image, label, writer_id
```

So your `Dataset.__getitem__` should look like:

```python
def __getitem__(self, idx):
    image = ...
    label = ...
    writer_id = ...
    return image, label, writer_id
```

Then the evaluation loop collects:

```python
all_true = []
all_pred = []
all_writer_ids = []

model.eval()
with torch.no_grad():
    for images, labels, writer_ids in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        all_true.extend(labels.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())
        all_writer_ids.extend(writer_ids)
```

Then:

```python
wm = writer_metrics(all_true, all_pred, all_writer_ids)
score = research_score(
    val_acc=val_acc,
    writer_gap=wm["writer_gap"],
    train_acc=train_acc,
)
```

## 7. MVP build order

Do it in this order:

```text
Phase 1 — Make the generic repo
1. Copy the Karpathy pattern, not the LLM code.
2. Add ROCm check script.
3. Add a toy CNN task.
4. Add writer-group metric support using fake writer IDs.
5. Make one baseline run work.

Phase 2 — Add your real project
6. Add your dataset adapter.
7. Ensure each sample has writer_id.
8. Run baseline.
9. Produce writer accuracy table.
10. Start agent loop.

Phase 3 — Contribution polish
11. Write README: "AutoResearch for CNNs on AMD/ROCm."
12. Add examples.
13. Add Docker instructions.
14. Add warning about metric misuse.
15. Add research_log format.
```

## 8. The biggest design risk

Your optimization target is not standard. That is fine, but you must avoid this trap:

```text
Bad objective:
minimize writer_gap only
```

Because a model with 20% accuracy for every writer has excellent writer_gap but terrible usefulness.

Use this instead:

```text
maximize:
validation accuracy
- writer disparity penalty
- overfitting penalty
```

Or stricter:

```text
Accept change only if:
research_score improves
AND val_acc does not drop by more than X
AND min_writer_acc does not drop
```

My recommendation:

```text
Start with lambda_gap = 0.5
Start with lambda_gen = 0.25
Minimum writer sample count = 5 or 10
Require 2-seed confirmation for major improvements
```

## 9. Clean first repo goal

Your first public contribution should say:

```text
A minimal autonomous CNN research harness inspired by Karpathy's autoresearch, targeting AMD/ROCm GPUs and group-level fairness metrics such as writer accuracy disparity.
```

That is specific, useful, and not overclaiming.

[1]: https://github.com/karpathy/autoresearch "GitHub - karpathy/autoresearch: AI agents running research on single-GPU nanochat training automatically · GitHub"
[2]: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html "PyTorch on ROCm installation — ROCm installation (Linux)"
