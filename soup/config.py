"""
Experiment Configuration Registry
==================================
Implements Table 3.1 from Section 3.2.4 of the thesis methodology.

Nine runs are defined:
  - Six ingredient training runs (L1-L4, C1-C2) – full fine-tune from θ₀.
  - Two head-only post-merge fine-tuning runs (D1, D2).
  - One final pipeline run (C3).

All six ingredient runs share the same pre-trained initialisation θ₀ and the
same base Detectron2 YOLOF-R50 configuration; exactly one hyperparameter
differs per run from the base.

YOLOF-R50 named parameter prefixes are verified against the Detectron2
checkpoint state dictionary and listed in PARAMETER_PREFIXES.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# YOLOF-R50 named parameter-key prefixes
# (verified against the Detectron2 YOLOF implementation)
# ---------------------------------------------------------------------------

PARAMETER_PREFIXES: Dict[str, List[str]] = {
    "backbone": ["model.backbone."],
    "encoder": ["model.proposal_generator.encoder."],
    "cls": [
        "model.proposal_generator.head.cls_subnet.",
        "model.proposal_generator.head.cls_score.",
    ],
    "reg": [
        "model.proposal_generator.head.bbox_subnet.",
        "model.proposal_generator.head.bbox_pred.",
        "model.proposal_generator.head.object_pred.",
    ],
}

# Convenience flat list of all ingredient run IDs
INGREDIENT_RUN_IDS: Tuple[str, ...] = ("L1", "L2", "L3", "L4", "C1", "C2")

# Run IDs for head-only post-merge fine-tuning
FINETUNE_RUN_IDS: Tuple[str, ...] = ("D1", "D2")

# Final pipeline run
FINAL_RUN_ID: str = "C3"


# ---------------------------------------------------------------------------
# Base Detectron2 YOLOF-R50 hyperparameter values
# These are the shared defaults; each ingredient run changes exactly one.
# ---------------------------------------------------------------------------

BASE_CONFIG: Dict[str, object] = {
    "BASE_LR": 0.01,
    "WEIGHT_DECAY": 1e-4,
    "MAX_ITER": None,  # Derived from epochs × steps_per_epoch
    "EPOCHS": 8,
    "BATCH_SIZE": 8,
    "LR_SCHEDULE": "WarmupMultiStepLR",
    "SEED": 1,
}


# ---------------------------------------------------------------------------
# Run configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Configuration for a single experimental run (Table 3.1)."""

    run_id: str
    gpu: str
    run_type: str  # "Full fine-tune" or "Head fine-tune"
    epochs: int  # 2, 8, or 2-3 (stored as int; C3 uses epochs_max)
    epochs_max: Optional[int]  # For C3 (2-3 epochs)
    batch_size: int
    changed_hyperparameter: Optional[str]  # None for D1/D2/C3
    seed: int
    purpose: str
    frozen_layers: Optional[List[str]]  # None = no frozen layers

    # Hyperparameter overrides relative to BASE_CONFIG
    hyperparameter_overrides: Dict[str, object] = field(default_factory=dict)

    @property
    def is_ingredient(self) -> bool:
        """True for the six ingredient pool runs (L1-L4, C1-C2)."""
        return self.run_id in INGREDIENT_RUN_IDS

    @property
    def is_finetune(self) -> bool:
        """True for the head-only fine-tuning runs (D1, D2, C3)."""
        return self.run_id in (*FINETUNE_RUN_IDS, FINAL_RUN_ID)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RunConfig(run_id={self.run_id!r}, gpu={self.gpu!r}, "
            f"type={self.run_type!r}, epochs={self.epochs}, "
            f"batch_size={self.batch_size}, seed={self.seed})"
        )


# ---------------------------------------------------------------------------
# Table 3.1 – Experiment Configuration Registry
# ---------------------------------------------------------------------------

RUN_CONFIGS: Dict[str, RunConfig] = {
    # ---- L-runs: RTX 5070 Ti, full fine-tune, batch 8 ----
    "L1": RunConfig(
        run_id="L1",
        gpu="RTX 5070 Ti",
        run_type="Full fine-tune",
        epochs=8,
        epochs_max=None,
        batch_size=8,
        changed_hyperparameter="Base config (anchor)",
        seed=1,
        purpose="Pool / RQ1-3",
        frozen_layers=None,
        hyperparameter_overrides={},
    ),
    "L2": RunConfig(
        run_id="L2",
        gpu="RTX 5070 Ti",
        run_type="Full fine-tune",
        epochs=8,
        epochs_max=None,
        batch_size=8,
        changed_hyperparameter="Learning rate",
        seed=2,
        purpose="Pool / RQ1-3",
        frozen_layers=None,
        hyperparameter_overrides={"BASE_LR": 0.02},
    ),
    "L3": RunConfig(
        run_id="L3",
        gpu="RTX 5070 Ti",
        run_type="Full fine-tune",
        epochs=8,
        epochs_max=None,
        batch_size=8,
        changed_hyperparameter="Weight decay",
        seed=3,
        purpose="Pool / RQ1-3",
        frozen_layers=None,
        hyperparameter_overrides={"WEIGHT_DECAY": 5e-4},
    ),
    "L4": RunConfig(
        run_id="L4",
        gpu="RTX 5070 Ti",
        run_type="Full fine-tune",
        epochs=10,  # Changed hyperparameter: training epochs
        epochs_max=None,
        batch_size=8,
        changed_hyperparameter="Training epochs",
        seed=4,
        purpose="Pool / RQ1-3",
        frozen_layers=None,
        hyperparameter_overrides={"EPOCHS": 10},
    ),
    # ---- C-runs: RTX 5090, full fine-tune, batch 16 ----
    "C1": RunConfig(
        run_id="C1",
        gpu="RTX 5090",
        run_type="Full fine-tune",
        epochs=8,
        epochs_max=None,
        batch_size=16,
        changed_hyperparameter="Batch size",
        seed=11,
        purpose="Pool / best single",
        frozen_layers=None,
        hyperparameter_overrides={"BATCH_SIZE": 16},
    ),
    "C2": RunConfig(
        run_id="C2",
        gpu="RTX 5090",
        run_type="Full fine-tune",
        epochs=8,
        epochs_max=None,
        batch_size=16,
        changed_hyperparameter="LR schedule",
        seed=22,
        purpose="Pool / best single",
        frozen_layers=None,
        hyperparameter_overrides={
            "BATCH_SIZE": 16,
            "LR_SCHEDULE": "WarmupCosineLR",
        },
    ),
    # ---- Post-merge head fine-tuning runs ----
    "D1": RunConfig(
        run_id="D1",
        gpu="RTX 5070 Ti",
        run_type="Head fine-tune",
        epochs=2,
        epochs_max=None,
        batch_size=8,
        changed_hyperparameter=None,
        seed=7,
        purpose="RQ3b: weak merge + FT",
        frozen_layers=["backbone", "encoder"],
        hyperparameter_overrides={},
    ),
    "D2": RunConfig(
        run_id="D2",
        gpu="RTX 5070 Ti",
        run_type="Head fine-tune",
        epochs=2,
        epochs_max=None,
        batch_size=8,
        changed_hyperparameter=None,
        seed=7,
        purpose="RQ3b: strong merge + FT",
        frozen_layers=["backbone", "encoder"],
        hyperparameter_overrides={},
    ),
    # ---- Final pipeline run ----
    "C3": RunConfig(
        run_id="C3",
        gpu="RTX 5090",
        run_type="Head fine-tune",
        epochs=2,
        epochs_max=3,
        batch_size=16,
        changed_hyperparameter=None,
        seed=33,
        purpose="RQ4: final soup",
        frozen_layers=["backbone", "encoder"],
        hyperparameter_overrides={"BATCH_SIZE": 16},
    ),
}


def get_ingredient_configs() -> Dict[str, RunConfig]:
    """Return the six ingredient run configurations (L1-L4, C1-C2)."""
    return {k: v for k, v in RUN_CONFIGS.items() if v.is_ingredient}


def get_finetune_configs() -> Dict[str, RunConfig]:
    """Return D1, D2, and C3 run configurations."""
    return {k: v for k, v in RUN_CONFIGS.items() if v.is_finetune}
