# ObjectDetectionSoup

**Component-specific model soups for YOLOF object detection**

Master's thesis implementation — IU International University of Applied Sciences

---

## Overview

This repository implements the experimental programme for the thesis:
*"Learned Model Soups for Object Detection: Does Component-Specific Parameter Averaging
Outperform Global Uniform Weight Averaging?"*

The central question: can we improve COCO mAP by applying **separate mixing coefficients
to the backbone, encoder, and decoder** of YOLOF-R50, rather than treating the model as
a monolithic parameter vector (Wortsman et al., 2022)?

---

## Repository Structure

```
ObjectDetectionSoup/
├── soup/               # Core merging library (Conditions 1-4)
├── shared/             # Shared YOLOF I/O, component partition, evaluator utilities
├── rq1_component_soup/ # RQ1: Component-specific vs. global averaging
├── rq2_loss_landscape/ # RQ2: Per-component loss barrier & Hessian analysis
├── rq3_coeff_strategy/ # RQ3: Mixing coefficient learning strategies
├── rq4_cross_family/   # RQ4: Cross-family generalisation (stub, deferred)
├── analysis/           # Jupyter notebooks and plotting scripts
├── configs/            # YAML experiment configurations
└── tests/              # Unit tests
```

---

## Branches

| Branch | Purpose |
|---|---|
| `main` | Stable code |
| `feature/experiments` | Full RQ1–RQ4 experiment packages (active development) |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/nisalperera/ObjectDetectionSoup.git
cd ObjectDetectionSoup
git checkout feature/experiments

# 2. Install all packages in editable mode
pip install -e soup/
pip install -e shared/
pip install -e rq1_component_soup/
pip install -e rq2_loss_landscape/
pip install -e rq3_coeff_strategy/

# 3. Configure paths
cp configs/paths.yaml configs/paths.local.yaml
# Edit configs/paths.local.yaml to point at your COCO data and checkpoints

# 4. Run RQ1
python rq1_component_soup/rq1_component_soup/run_experiment.py \
    --config configs/experiments/rq1_component_soup.yaml \
    --paths configs/paths.yaml

# 5. Run RQ2 (barrier only — Hessian is overnight)
python rq2_loss_landscape/rq2_loss_landscape/run_experiment.py \
    --config configs/experiments/rq2_loss_landscape.yaml \
    --skip-hessian

# 6. Run RQ3 (validate proxy first)
python rq3_coeff_strategy/rq3_coeff_strategy/run_experiment.py \
    --config configs/experiments/rq3_coeff_strategy.yaml \
    --validate-proxy-only        # Run this first to validate entropy ↔ mAP correlation
```

---

## YOLOF-R50 Component Partition

| Component | Detectron2 key prefix | Thesis role |
|---|---|---|
| Backbone | `model.backbone.*` | ResNet-50 C5 feature extractor |
| Encoder | `model.proposal_generator.encoder.*` | Dilated Encoder (novel SiSo neck) |
| Cls head | `model.proposal_generator.head.cls_subnet.*`, `cls_score.*` | Classification branch |
| Reg head | `model.proposal_generator.head.bbox_subnet.*`, `bbox_pred.*`, `object_pred.*` | Regression + objectness branch |

---

## Requirements

See `requirements.txt`. Key dependencies:
- Python ≥ 3.9
- PyTorch ≥ 2.0
- Detectron2 (GitHub install)
- YOLOF (GitHub install)
- numpy, scipy, tqdm, pyyaml

---

## Citation

If you use this code, please cite:
```
Perera, N. (2026). Learned Model Soups for Object Detection.
Master's Thesis, IU International University of Applied Sciences.
```
