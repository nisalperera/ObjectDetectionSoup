# ObjectDetectionSoup

**Model Soups experiments for Object Detection using YOLOF-R50 and Detectron2.**

ObjectDetectionSoup applies the *model soups* technique (Wortsman et al., 2022) to the COCO object-detection task. Instead of picking the single best fine-tuned checkpoint, multiple checkpoints that were trained with different hyperparameters are *merged* by averaging their weights. This repository implements a two-stage, branch-aware merging procedure for the YOLOF-R50 architecture and provides four merging conditions, a loss-landscape analysis module, a full 12-step data-collection pipeline, and a test suite.

---

## Table of Contents

- [Background](#background)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Merging Conditions](#merging-conditions)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Running Individual Pipeline Steps](#running-individual-pipeline-steps)
- [Experiment Configuration](#experiment-configuration)
- [Loss Landscape Analysis](#loss-landscape-analysis)
- [Evaluation](#evaluation)
- [Utilities](#utilities)
- [Running the Tests](#running-the-tests)
- [Project Structure](#project-structure)

---

## Background

*Model soups* (Wortsman et al., 2022) averages the weights of several fine-tuned models that all started from the same pre-trained checkpoint θ₀. This typically improves accuracy and robustness over any single ingredient model.

ObjectDetectionSoup extends this idea to YOLOF-R50 on COCO by partitioning the decoder into two branches—a classification branch (`cls`) and a regression branch (`reg`)—and applying independent per-branch weight averaging. Four merging conditions are studied:

| Condition | Name | Description |
|-----------|------|-------------|
| 1 | Global uniform soup | Entire decoder averaged with equal weights (baseline) |
| 2 | Branch uniform soup | `cls` and `reg` averaged separately, still with equal weights |
| 3 | Branch Dirichlet search | Per-branch coefficients found by a two-stage Dirichlet random search |
| 4 | Fisher-weighted branch soup | Coefficients proportional to each model's empirical Fisher information trace |

---

## Repository Layout

```
soup/
  config.py          – Experiment configuration registry (nine runs, Table 3.1)
  merger.py          – Core merging module (all four conditions)
  loss_landscape.py  – Per-branch loss-barrier & Hutchinson Hessian-trace measurement
  train.py           – Detectron2 training helpers
  evaluate.py        – Ingredient quality audit & COCOEvaluator wrapper
  pipeline.py        – 12-step data-collection orchestrator
  utils.py           – SHA-256 hashing, seed helpers, JSON/CSV logging
tests/               – pytest test suite
requirements.txt     – Python dependencies
```

---

## Requirements

**Core (always needed)**

| Package | Version |
|---------|---------|
| Python | ≥ 3.9 |
| numpy | ≥ 1.24 |

**Heavy dependencies (required for training and evaluation on a GPU machine)**

| Package | Notes |
|---------|-------|
| PyTorch | ≥ 2.0, with matching CUDA toolkit |
| torchvision | ≥ 0.15 |
| Detectron2 | Install from source – see the [official guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) |
| pyhessian | ≥ 0.1 – Hutchinson Hessian-trace estimation |
| pycocotools | ≥ 2.0 – COCO evaluation |

The core merging and utility functions work with plain NumPy arrays and do **not** require PyTorch or Detectron2. All heavy-dependency code is guarded with `try/except ImportError` so the test suite and merging logic run in a lightweight environment.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/nisalperera/ObjectDetectionSoup.git
cd ObjectDetectionSoup

# 2. Install core dependencies
pip install -r requirements.txt

# 3. (Optional) Install heavy dependencies for training / evaluation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install pyhessian pycocotools
```

---

## Quick Start

The merging functions operate on plain Python dictionaries that map parameter names to NumPy arrays. When using Detectron2 checkpoints, extract the state dict first:

```python
import torch
import numpy as np

def load_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    return {k: v.numpy() for k, v in ckpt["model"].items()}

state_dicts = [load_state_dict(p) for p in checkpoint_paths]
```

### Condition 1 – Global uniform soup

```python
from soup import merge_uniform

soup = merge_uniform(state_dicts)
```

### Condition 2 – Branch uniform soup

```python
from soup import merge_branch_uniform

soup = merge_branch_uniform(state_dicts)
```

### Condition 3 – Branch Dirichlet search

```python
from soup import merge_branch_dirichlet

def eval_fn(state_dict: dict) -> float:
    # Load state_dict into your model, run COCO val evaluation,
    # and return mAP50:95 (0–100 scale).
    ...

soup, lambda_cls, lambda_reg = merge_branch_dirichlet(
    state_dicts,
    eval_fn=eval_fn,
    seed=42,
)
print("Best λ_cls:", lambda_cls)
print("Best λ_reg:", lambda_reg)
```

You can also supply a faster mini-val evaluation function for the Stage A exploration phase and reserve the full val2017 evaluation for Stage B confirmation:

```python
soup, lambda_cls, lambda_reg = merge_branch_dirichlet(
    state_dicts,
    eval_fn=full_val_eval_fn,       # used in Stage B (5 000 images)
    mini_val_eval_fn=mini_val_fn,   # used in Stage A (500 images)
    m=100,   # Stage A: number of Dirichlet candidates per branch
    k=10,    # Stage B: top-K candidates to confirm
    seed=42,
)
```

### Condition 4 – Fisher-weighted branch soup

First collect per-model, per-branch Fisher traces (empirical Fisher information, estimated from gradient samples on a calibration set):

```python
from soup import merge_branch_fisher

# fisher_traces = {run_id: {"cls": float, "reg": float}}
fisher_traces = {
    "L1": {"cls": 12.4, "reg": 8.1},
    "L2": {"cls": 10.9, "reg": 9.3},
    # ...
}

soup, lambda_cls, lambda_reg = merge_branch_fisher(state_dicts, fisher_traces)
```

The order of keys in `fisher_traces` must match the order of `state_dicts`.

---

## Running the Full Pipeline

The `run_full_pipeline` helper orchestrates all 12 steps end-to-end:

```python
from soup.pipeline import run_full_pipeline

run_full_pipeline(
    output_dir="./experiments",
    theta0_path="./checkpoints/theta0.pth",
)
```

`output_dir` will be created with the following subdirectories:

```
experiments/
  checkpoints/        – per-run .pth files
  logs/               – training logs
  merges/             – merged model state dicts
  barriers/           – loss-barrier .npy arrays
  hessian/            – Hutchinson trace outputs
  lambda_vectors/     – best λ_cls and λ_reg vectors
  sha256/             – checkpoint SHA-256 manifest
```

---

## Running Individual Pipeline Steps

Each step is a standalone function. Import and call only the steps you need:

```python
from soup.pipeline import (
    step1_setup,
    step2_download_pretrained,
    step2b_pilot_check,
    step3_train_l_runs,
    step4_train_c_runs,
    step5_ingredient_audit,
    step6_loss_landscape,
    step7_merging,
    step8_preregister_sources,
    step9_head_finetune,
    step10_final_pipeline,
    step11_archive,
)

# Step 1 – create output directory structure
output_dir = step1_setup("./experiments")

# Step 2 – download/copy pre-trained θ₀ and record its SHA-256
theta0_path, sha256 = step2_download_pretrained(
    output_dir,
    weights_url_or_path="https://dl.fbaipublicfiles.com/detectron2/YOLOF/...",
)

# Step 2b – pilot divergence check (abort early if LR is badly misconfigured)
step2b_pilot_check(
    output_dir,
    theta0_metrics={"AP": 37.5},
    pilot_eval_fn=my_pilot_fn,
)

# Step 3 – train ingredient runs L1–L4
def my_trainer(run_id, run_output_dir, theta0_path):
    ...  # launch Detectron2 training

step3_train_l_runs(output_dir, theta0_path, trainer_fn=my_trainer)

# Step 4 – train extended pool runs C1–C2
step4_train_c_runs(output_dir, theta0_path, trainer_fn=my_trainer)
```

---

## Experiment Configuration

Nine experimental runs are pre-configured in `soup/config.py` (Table 3.1):

| Run | GPU | Type | Epochs | Batch | Changed hyperparameter | Purpose |
|-----|-----|------|--------|-------|------------------------|---------|
| L1 | RTX 5070 Ti | Full fine-tune | 8 | 8 | Base config (anchor) | Ingredient pool |
| L2 | RTX 5070 Ti | Full fine-tune | 8 | 8 | Learning rate ×2 | Ingredient pool |
| L3 | RTX 5070 Ti | Full fine-tune | 8 | 8 | Weight decay ×5 | Ingredient pool |
| L4 | RTX 5070 Ti | Full fine-tune | 10 | 8 | Training epochs +2 | Ingredient pool |
| C1 | RTX 5090 | Full fine-tune | 8 | 16 | Batch size ×2 | Extended pool |
| C2 | RTX 5090 | Full fine-tune | 8 | 16 | LR schedule (cosine) | Extended pool |
| D1 | RTX 5070 Ti | Head fine-tune | 2 | 8 | – | Post-merge FT (weak) |
| D2 | RTX 5070 Ti | Head fine-tune | 2 | 8 | – | Post-merge FT (strong) |
| C3 | RTX 5090 | Head fine-tune | 2–3 | 16 | – | Final soup |

Access configurations programmatically:

```python
from soup.config import RUN_CONFIGS, get_ingredient_configs, get_finetune_configs

# All nine configs
print(RUN_CONFIGS["L1"])

# Only the six ingredient runs (L1-L4, C1-C2)
ingredient_configs = get_ingredient_configs()

# Post-merge fine-tuning runs (D1, D2, C3)
finetune_configs = get_finetune_configs()
```

---

## Loss Landscape Analysis

`soup/loss_landscape.py` provides two analysis tools described in Section 3.3.3:

### Loss barrier

Measures the worst-case loss increase when linearly interpolating between two model's component parameters:

```python
from soup.loss_landscape import measure_all_barriers, save_barriers_npy

def loss_fn(state_dict: dict) -> float:
    # Evaluate total loss on a calibration set and return a scalar.
    ...

# Measure barriers for all pairs across all components
barriers = measure_all_barriers(state_dicts, loss_fn=loss_fn)
# barriers = {(i, j): {"backbone": float, "encoder": float, "cls": float, "reg": float}}

# Persist as .npy file
save_barriers_npy(barriers, path="./experiments/barriers/barriers.npy")
```

### Hutchinson Hessian-trace estimation

Estimates the trace of the Hessian for each component using 50 Rademacher random vectors (requires PyHessian):

```python
from soup.loss_landscape import hutchinson_trace_all

traces = hutchinson_trace_all(state_dicts, loss_fn=loss_fn)
# traces = [{component: float}, ...]  – one dict per model
```

---

## Evaluation

`soup/evaluate.py` provides COCO evaluation helpers (requires Detectron2):

```python
from soup.evaluate import evaluate_checkpoint, ingredient_quality_audit

# Evaluate a single checkpoint
metrics = evaluate_checkpoint(
    checkpoint_path="./experiments/checkpoints/L1/model_final.pth",
    cfg=my_detectron2_cfg,
    dataset_name="coco_2017_val",
    output_dir="./experiments/logs",
)
print(metrics)  # {"AP": 39.2, "AP50": 57.1, "AR100": 52.4}

# Audit the full ingredient pool and flag weak models
audit_results = ingredient_quality_audit(
    metrics_per_run={"L1": metrics_l1, "L2": metrics_l2, ...},
)
# Returns a dict with per-run quality flags and the pool maximum mAP.
```

Models whose mAP50:95 falls more than 3 percentage points below the pool maximum are automatically flagged for exclusion (following Wortsman et al., 2022).

---

## Utilities

```python
from soup.utils import compute_sha256, setup_seed, log_lambda_vectors

# Reproducibility
setup_seed(42)

# Verify checkpoint integrity
digest = compute_sha256("./checkpoints/theta0.pth")

# Persist the best lambda vectors found by Condition 3 or 4
log_lambda_vectors(
    lambda_cls=lambda_cls,
    lambda_reg=lambda_reg,
    path="./experiments/lambda_vectors/cond3.json",
)
```

---

## Running the Tests

```bash
# Install test dependencies (already in requirements.txt)
pip install pytest pytest-cov

# Run the full test suite
pytest

# With coverage report
pytest --cov=soup --cov-report=term-missing
```

The tests are designed to run without PyTorch or Detectron2 installed. All heavy-dependency code is mocked or guarded so the suite passes in a lightweight CI environment.

---

## References

- Wortsman, M., et al. (2022). *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.* ICML 2022. [arXiv:2203.05482](https://arxiv.org/abs/2203.05482)
- Matena, M. & Raffel, C. (2022). *Merging models with Fisher-weighted averaging.* NeurIPS 2022. [arXiv:2111.09832](https://arxiv.org/abs/2111.09832)
- Frankle, J., et al. (2020). *Linear mode connectivity and the lottery ticket hypothesis.* ICML 2020. [arXiv:1912.05671](https://arxiv.org/abs/1912.05671)
- Yao, Z., et al. (2020). *PyHessian: Neural networks through the lens of the Hessian.* IEEE Big Data 2020. [arXiv:1912.07145](https://arxiv.org/abs/1912.07145)

---

## License

See [LICENSE](LICENSE).
