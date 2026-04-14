"""
Training Helpers
================
Per-run Detectron2 configuration builder and training launcher for all nine
experimental runs (Table 3.1, Section 3.2.4).

When Detectron2 is not installed the module is still importable; calling any
function that needs Detectron2 will raise ``ImportError`` with a clear message.

Typical usage::

    from soup.train import build_run_config, launch_training

    cfg = build_run_config("L2", output_dir="./output/L2")
    launch_training(cfg, run_id="L2")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from soup.config import BASE_CONFIG, INGREDIENT_RUN_IDS, RUN_CONFIGS, RunConfig
from soup.utils import setup_detectron2_seed, setup_seed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

# Detectron2 YOLOF-R50 model-zoo config (YAML) and pretrained checkpoint URL.
# Updated URLs/names can be overridden at runtime via build_run_config().
YOLOF_R50_CONFIG_FILE: str = (
    "COCO-Detection/yolof_R_50_DC5_1x.yaml"
)
YOLOF_R50_WEIGHTS_URL: str = (
    "detectron2://COCO-Detection/yolof_R_50_DC5_1x/142999717/model_final_dc5d9e.pkl"
)


def build_run_config(
    run_id: str,
    output_dir: Union[str, Path],
    base_weights: Optional[str] = None,
    config_file: Optional[str] = None,
    extra_opts: Optional[list] = None,
) -> "CfgNode":  # type: ignore[name-defined]  # noqa: F821
    """Build a Detectron2 ``CfgNode`` for the given *run_id*.

    Starts from the base YOLOF-R50 config and applies the single
    hyperparameter override specified in Table 3.1 for that run.

    Parameters
    ----------
    run_id:
        One of the nine run IDs: L1-L4, C1-C2, D1, D2, C3.
    output_dir:
        Directory to write training artefacts (checkpoints, logs).
    base_weights:
        Path or URL to the pre-trained θ₀ checkpoint.  Defaults to the
        Detectron2 YOLOF-R50 model-zoo checkpoint.
    config_file:
        Detectron2 YAML config file.  Defaults to YOLOF-R50 DC5-1x.
    extra_opts:
        Additional ``key value`` pairs passed to ``cfg.merge_from_list``.

    Returns
    -------
    CfgNode
        A fully-configured Detectron2 CfgNode ready for training.

    Raises
    ------
    ImportError
        If Detectron2 is not installed.
    KeyError
        If *run_id* is not in :data:`soup.config.RUN_CONFIGS`.
    """
    try:
        from detectron2.config import get_cfg  # type: ignore[import]
        from detectron2.model_zoo import get_config_file  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Detectron2 is required to build run configurations. "
            "Install it following https://detectron2.readthedocs.io/."
        ) from exc

    if run_id not in RUN_CONFIGS:
        raise KeyError(f"Unknown run_id {run_id!r}. Valid IDs: {list(RUN_CONFIGS.keys())}.")

    run_cfg: RunConfig = RUN_CONFIGS[run_id]
    cfg = get_cfg()

    _config_file = config_file or YOLOF_R50_CONFIG_FILE
    cfg.merge_from_file(get_config_file(_config_file))

    _weights = base_weights or YOLOF_R50_WEIGHTS_URL
    cfg.MODEL.WEIGHTS = _weights
    cfg.OUTPUT_DIR = str(output_dir)

    # ---- Apply hyperparameter overrides ----
    overrides = run_cfg.hyperparameter_overrides

    if "BASE_LR" in overrides:
        cfg.SOLVER.BASE_LR = float(overrides["BASE_LR"])

    if "WEIGHT_DECAY" in overrides:
        cfg.SOLVER.WEIGHT_DECAY = float(overrides["WEIGHT_DECAY"])

    if "BATCH_SIZE" in overrides:
        cfg.SOLVER.IMS_PER_BATCH = int(overrides["BATCH_SIZE"])
    else:
        cfg.SOLVER.IMS_PER_BATCH = int(BASE_CONFIG["BATCH_SIZE"])

    if "EPOCHS" in overrides:
        _epochs = int(overrides["EPOCHS"])
    else:
        _epochs = int(run_cfg.epochs)

    # Convert epochs to iterations (COCO train2017 ≈ 118 287 images)
    steps_per_epoch = max(1, 118287 // cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = _epochs * steps_per_epoch

    if "LR_SCHEDULE" in overrides and overrides["LR_SCHEDULE"] == "WarmupCosineLR":
        cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"

    # ---- Seed ----
    cfg.SEED = run_cfg.seed

    # ---- Frozen layers for head-only fine-tuning (D1, D2, C3) ----
    if run_cfg.frozen_layers:
        # Detectron2 uses FREEZE_AT for the backbone; encoder/head freezing is
        # handled programmatically in launch_training via requires_grad=False.
        cfg.MODEL.BACKBONE.FREEZE_AT = 5  # freeze all ResNet stages

    if extra_opts:
        cfg.merge_from_list(extra_opts)

    cfg.freeze()
    return cfg


def _freeze_component_parameters(model: "torch.nn.Module", component: str) -> None:  # noqa: F821
    """Set ``requires_grad=False`` for all parameters in *component*.

    Parameters
    ----------
    model:
        Detectron2 YOLOF model.
    component:
        ``"backbone"`` or ``"encoder"`` (the frozen components for D1/D2/C3).
    """
    from soup.config import PARAMETER_PREFIXES

    prefixes = PARAMETER_PREFIXES[component]
    frozen = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad_(False)
            frozen += 1
    logger.info("Froze %d parameters in component '%s'.", frozen, component)


def launch_training(
    cfg: "CfgNode",  # type: ignore[name-defined]  # noqa: F821
    run_id: str,
    resume: bool = False,
) -> None:
    """Launch Detectron2 training for the given *run_id*.

    Applies per-run seed setup, freezes the appropriate layers for head-only
    fine-tuning runs, and starts the Detectron2 ``DefaultTrainer``.

    Parameters
    ----------
    cfg:
        A Detectron2 CfgNode produced by :func:`build_run_config`.
    run_id:
        Run identifier from Table 3.1.
    resume:
        If ``True``, resume from the latest checkpoint in ``cfg.OUTPUT_DIR``.

    Raises
    ------
    ImportError
        If Detectron2 is not installed.
    """
    try:
        import torch
        from detectron2.engine import DefaultTrainer  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Detectron2 and PyTorch are required to run training."
        ) from exc

    run_config: RunConfig = RUN_CONFIGS[run_id]

    # Reproducibility anchoring (Step 1)
    setup_detectron2_seed(run_config.seed)

    # Build trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume)

    # Freeze backbone + encoder for head-only fine-tuning runs
    if run_config.frozen_layers:
        for layer in run_config.frozen_layers:
            _freeze_component_parameters(trainer.model, layer)

    logger.info(
        "Starting training for run %s  (seed=%d, epochs=%d, batch=%d).",
        run_id,
        run_config.seed,
        run_config.epochs,
        run_config.batch_size,
    )
    trainer.train()


def build_all_ingredient_configs(
    output_base: Union[str, Path],
    base_weights: Optional[str] = None,
) -> dict:
    """Build Detectron2 CfgNodes for all six ingredient runs (L1-L4, C1-C2).

    Parameters
    ----------
    output_base:
        Base output directory; each run writes to ``output_base/<run_id>``.
    base_weights:
        Shared θ₀ checkpoint URL/path.

    Returns
    -------
    dict
        ``{run_id: CfgNode}`` for all six ingredient runs.
    """
    configs = {}
    for run_id in INGREDIENT_RUN_IDS:
        output_dir = Path(output_base) / run_id
        configs[run_id] = build_run_config(
            run_id,
            output_dir=output_dir,
            base_weights=base_weights,
        )
    return configs
