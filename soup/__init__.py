"""
ObjectDetectionSoup
===================
Model-soup merging experiments for YOLOF-R50 object detection using Detectron2.

Package layout
--------------
config          Experiment Configuration Registry (Table 3.1, nine runs).
merger          Core merging module – four merging conditions.
loss_landscape  Per-branch loss-barrier and Hutchinson Hessian-trace measurement.
train           Per-run Detectron2 training helpers.
evaluate        Ingredient quality audit and COCOEvaluator wrapper.
pipeline        12-step data collection orchestrator.
utils           SHA-256 hashing, seed utilities, JSON logging.
"""

from soup.config import RUN_CONFIGS, PARAMETER_PREFIXES, INGREDIENT_RUN_IDS
from soup.merger import (
    merge_uniform,
    merge_branch_uniform,
    merge_branch_dirichlet,
    merge_branch_fisher,
)
from soup.utils import compute_sha256, setup_seed, log_lambda_vectors

__all__ = [
    "RUN_CONFIGS",
    "PARAMETER_PREFIXES",
    "INGREDIENT_RUN_IDS",
    "merge_uniform",
    "merge_branch_uniform",
    "merge_branch_dirichlet",
    "merge_branch_fisher",
    "compute_sha256",
    "setup_seed",
    "log_lambda_vectors",
]
