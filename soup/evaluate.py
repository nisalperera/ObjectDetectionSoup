"""
Evaluation Utilities
====================
Ingredient quality audit (Step 5) and a thin wrapper around Detectron2's
``COCOEvaluator`` for mAP50:95, mAP50, and AR@100 measurement.

When Detectron2 is not installed every function that needs it raises a clear
``ImportError``; the module remains importable so non-Detectron2 tests pass.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from soup.config import INGREDIENT_RUN_IDS

logger = logging.getLogger(__name__)

# Models whose mAP50:95 falls more than this many percentage points below the
# pool maximum are flagged for exclusion (Wortsman et al., 2022, p. 23970).
QUALITY_THRESHOLD_PP: float = 3.0


# ---------------------------------------------------------------------------
# Detectron2 COCO evaluation wrapper
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    checkpoint_path: Union[str, Path],
    cfg: "CfgNode",  # type: ignore[name-defined]  # noqa: F821
    dataset_name: str = "coco_2017_val",
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """Evaluate a Detectron2 checkpoint on a COCO dataset split.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pth`` checkpoint file.
    cfg:
        Detectron2 ``CfgNode`` (modified in-place to point at
        *checkpoint_path*).
    dataset_name:
        Detectron2 dataset name, e.g. ``"coco_2017_val"`` or
        ``"coco_2017_test"`` for test-dev.
    output_dir:
        Directory to write the COCOEvaluator JSON results.

    Returns
    -------
    dict
        ``{"AP": float, "AP50": float, "AR100": float}`` where AP is
        mAP50:95 (0–100 scale).

    Raises
    ------
    ImportError
        If Detectron2 is not installed.
    """
    try:
        import torch
        from detectron2.checkpoint import DetectionCheckpointer  # type: ignore[import]
        from detectron2.data import build_detection_test_loader  # type: ignore[import]
        from detectron2.engine import DefaultPredictor  # type: ignore[import]
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset  # type: ignore[import]
        from detectron2.modeling import build_model  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Detectron2 is required for checkpoint evaluation."
        ) from exc

    cfg = cfg.clone()
    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(checkpoint_path)
    cfg.freeze()

    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(str(checkpoint_path))

    _output_dir = str(output_dir) if output_dir else cfg.OUTPUT_DIR
    evaluator = COCOEvaluator(
        dataset_name,
        output_dir=_output_dir,
        tasks=("bbox",),
    )
    val_loader = build_detection_test_loader(cfg, dataset_name)

    results = inference_on_dataset(model, val_loader, evaluator)
    bbox_results = results.get("bbox", {})

    metrics = {
        "AP": float(bbox_results.get("AP", 0.0)),
        "AP50": float(bbox_results.get("AP50", 0.0)),
        "AR100": float(bbox_results.get("AR100", 0.0)),
    }
    logger.info(
        "Checkpoint %s: AP=%.2f, AP50=%.2f, AR100=%.2f",
        Path(checkpoint_path).name,
        metrics["AP"],
        metrics["AP50"],
        metrics["AR100"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Step 5 – Ingredient quality audit
# ---------------------------------------------------------------------------


def ingredient_quality_audit(
    checkpoint_metrics: Dict[str, Dict[str, float]],
    threshold_pp: float = QUALITY_THRESHOLD_PP,
) -> Tuple[List[str], List[str]]:
    """Identify pool checkpoints that fall below the quality threshold.

    A checkpoint is flagged for exclusion if its ``AP`` (mAP50:95) falls more
    than *threshold_pp* percentage points below the pool maximum, following
    the greedy inclusion principle of Wortsman et al. (2022, p. 23970).

    Parameters
    ----------
    checkpoint_metrics:
        ``{run_id: {"AP": float, "AP50": float, "AR100": float}}``.
        Typically the output of :func:`evaluate_checkpoint` for each run.
    threshold_pp:
        Exclusion threshold in percentage points (default 3.0).

    Returns
    -------
    included : List[str]
        Run IDs that pass the quality check.
    excluded : List[str]
        Run IDs flagged for exclusion.
    """
    if not checkpoint_metrics:
        return [], []

    max_ap = max(v["AP"] for v in checkpoint_metrics.values())
    included: List[str] = []
    excluded: List[str] = []

    for run_id, metrics in checkpoint_metrics.items():
        ap = metrics["AP"]
        if max_ap - ap > threshold_pp:
            logger.warning(
                "Run %s flagged for exclusion: AP=%.2f  (pool max=%.2f, "
                "threshold=%.1f pp).",
                run_id, ap, max_ap, threshold_pp,
            )
            excluded.append(run_id)
        else:
            included.append(run_id)

    logger.info(
        "Ingredient audit complete: %d included, %d excluded (threshold %.1f pp).",
        len(included), len(excluded), threshold_pp,
    )
    return included, excluded


# ---------------------------------------------------------------------------
# Pilot divergence check (Step 2b)
# ---------------------------------------------------------------------------


def pilot_divergence_check(
    pilot_metrics: Dict[str, float],
    theta0_metrics: Dict[str, float],
    min_ap_delta: float = 0.3,
) -> bool:
    """Check whether the pilot run produces sufficient weight-space movement.

    Returns ``True`` if the pilot passes the divergence check; ``False`` if
    the base learning rate should be doubled and the check repeated.

    Parameters
    ----------
    pilot_metrics:
        ``{"AP": float, …}`` for the 2-epoch pilot checkpoint.
    theta0_metrics:
        ``{"AP": float, …}`` for the pre-trained θ₀ checkpoint.
    min_ap_delta:
        Minimum required |ΔmAP50:95| in percentage points (default 0.3).

    Returns
    -------
    bool
        ``True`` if ``|AP_pilot − AP_theta0| >= min_ap_delta``.
    """
    delta = abs(pilot_metrics["AP"] - theta0_metrics["AP"])
    passed = delta >= min_ap_delta
    logger.info(
        "Pilot divergence check: |ΔAP| = %.4f pp  (min required %.2f pp) → %s.",
        delta, min_ap_delta, "PASS" if passed else "FAIL – increase LR × 2",
    )
    return passed


# ---------------------------------------------------------------------------
# Best-single-model selection
# ---------------------------------------------------------------------------


def best_single_model(
    checkpoint_metrics: Dict[str, Dict[str, float]],
) -> Tuple[str, float]:
    """Return the run ID and AP of the best individual ingredient model.

    Parameters
    ----------
    checkpoint_metrics:
        ``{run_id: {"AP": float, …}}``.

    Returns
    -------
    best_id : str
        Run ID with the highest mAP50:95.
    best_ap : float
        The corresponding mAP50:95 value.
    """
    best_id = max(checkpoint_metrics, key=lambda r: checkpoint_metrics[r]["AP"])
    best_ap = checkpoint_metrics[best_id]["AP"]
    logger.info(
        "Best single model: %s  AP=%.4f.", best_id, best_ap
    )
    return best_id, best_ap
