"""
12-Step Data Collection Pipeline
==================================
Orchestrates all twelve steps of the data collection procedure described in
Section 3.4 of the thesis methodology, from environment setup through
archiving.

Each step is implemented as a standalone function that can be run
independently or called in sequence via :func:`run_full_pipeline`.

Usage::

    from soup.pipeline import run_full_pipeline

    run_full_pipeline(
        output_dir="./experiments",
        theta0_path="./checkpoints/theta0.pth",
    )

Or step by step::

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
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from soup.config import (
    INGREDIENT_RUN_IDS,
    RUN_CONFIGS,
    RunConfig,
)
from soup.evaluate import (
    best_single_model,
    ingredient_quality_audit,
    pilot_divergence_check,
)
from soup.loss_landscape import measure_all_barriers, save_barriers_npy
from soup.merger import (
    StateDict,
    merge_branch_dirichlet,
    merge_branch_fisher,
    merge_branch_uniform,
    merge_uniform,
)
from soup.utils import (
    compute_sha256,
    log_checkpoint_hash,
    log_lambda_vectors,
    save_metrics_csv,
    setup_detectron2_seed,
    setup_seed,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1 – Environment setup and reproducibility anchoring
# ---------------------------------------------------------------------------


def step1_setup(output_dir: Union[str, Path]) -> Path:
    """Step 1: environment setup and reproducibility anchoring.

    Creates the output directory structure required by subsequent steps.
    Records the setup in a manifest JSON alongside the directory.

    Parameters
    ----------
    output_dir:
        Root experiment output directory.

    Returns
    -------
    Path
        The created output directory.
    """
    output_dir = Path(output_dir)
    for sub in (
        "checkpoints", "logs", "merges", "barriers", "hessian",
        "lambda_vectors", "sha256",
    ):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    manifest = {
        "output_dir": str(output_dir),
        "torch_deterministic": True,
        "torch_benchmark": False,
    }

    try:
        import torch
        manifest["torch_version"] = torch.__version__
        manifest["cuda_available"] = torch.cuda.is_available()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        manifest["torch_version"] = "not installed"

    try:
        import detectron2
        manifest["detectron2_version"] = detectron2.__version__
    except ImportError:
        manifest["detectron2_version"] = "not installed"

    manifest_path = output_dir / "setup_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info("Step 1 complete. Output dir: %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Step 2 – Pre-trained weight retrieval
# ---------------------------------------------------------------------------


def step2_download_pretrained(
    output_dir: Union[str, Path],
    weights_url_or_path: str,
) -> Tuple[Path, str]:
    """Step 2: retrieve and hash the shared pre-trained θ₀ checkpoint.

    If *weights_url_or_path* is a URL, the checkpoint is downloaded using
    Detectron2's model zoo utilities; otherwise it is copied to
    ``output_dir/checkpoints/theta0.pth``.

    Parameters
    ----------
    output_dir:
        Root experiment output directory.
    weights_url_or_path:
        URL or local path of the YOLOF-R50 pre-trained checkpoint.

    Returns
    -------
    theta0_path : Path
        Local path of the downloaded/copied checkpoint.
    sha256 : str
        SHA-256 hex digest of the checkpoint.
    """
    output_dir = Path(output_dir)
    dest = output_dir / "checkpoints" / "theta0.pth"

    if weights_url_or_path.startswith("http"):
        try:
            from detectron2.utils.file_io import PathManager  # type: ignore[import]
            local = PathManager.get_local_path(weights_url_or_path)
            shutil.copy2(local, dest)
        except ImportError as exc:
            raise ImportError(
                "Detectron2 is required to download model-zoo checkpoints."
            ) from exc
    else:
        shutil.copy2(weights_url_or_path, dest)

    sha256 = log_checkpoint_hash(
        dest,
        log_path=output_dir / "sha256" / "checkpoint_hashes.json",
    )
    logger.info("Step 2 complete. θ₀ saved to %s  (SHA-256: %s).", dest, sha256)
    return dest, sha256


# ---------------------------------------------------------------------------
# Step 2b – Pilot divergence check
# ---------------------------------------------------------------------------


def step2b_pilot_check(
    output_dir: Union[str, Path],
    theta0_metrics: Dict[str, float],
    pilot_eval_fn: Callable[[], Dict[str, float]],
    min_ap_delta: float = 0.3,
    max_attempts: int = 5,
) -> bool:
    """Step 2b: pilot divergence check before full training.

    Executes a 2-epoch pilot run using the L1 base configuration and compares
    mAP50:95 against θ₀.  If the absolute delta is < *min_ap_delta* pp, the
    base LR is doubled and the check is repeated (up to *max_attempts* times).

    Parameters
    ----------
    output_dir:
        Root experiment output directory.
    theta0_metrics:
        ``{"AP": float, …}`` for the pre-trained θ₀ checkpoint.
    pilot_eval_fn:
        Callable that runs the 2-epoch pilot and returns metrics dict.
        Receives no arguments; any LR doubling must be handled externally.
    min_ap_delta:
        Minimum |ΔmAP50:95| required (default 0.3 pp).
    max_attempts:
        Maximum number of LR-doubling attempts (default 5).

    Returns
    -------
    bool
        ``True`` once the pilot passes; raises ``RuntimeError`` if all
        attempts are exhausted.

    Raises
    ------
    RuntimeError
        If the pilot fails to pass after *max_attempts* attempts.
    """
    for attempt in range(1, max_attempts + 1):
        pilot_metrics = pilot_eval_fn()
        passed = pilot_divergence_check(pilot_metrics, theta0_metrics, min_ap_delta)
        if passed:
            logger.info("Step 2b: pilot passed on attempt %d.", attempt)
            return True
        logger.warning(
            "Step 2b: pilot attempt %d failed – caller should double base LR.",
            attempt,
        )

    raise RuntimeError(
        f"Pilot divergence check failed after {max_attempts} attempts.  "
        "The ingredient pool may collapse; review base LR and configuration."
    )


# ---------------------------------------------------------------------------
# Step 3 – Hyperparameter-diverse pool training (L1-L4)
# ---------------------------------------------------------------------------


def step3_train_l_runs(
    output_dir: Union[str, Path],
    theta0_path: Union[str, Path],
    trainer_fn: Callable[[str, Path, str], None],
) -> Dict[str, Path]:
    """Step 3: train runs L1-L4 on the RTX 5070 Ti.

    Parameters
    ----------
    output_dir:
        Root experiment output directory.
    theta0_path:
        Path to shared θ₀ checkpoint.
    trainer_fn:
        ``(run_id, output_dir, theta0_path) → None``.
        Caller-supplied function that actually launches the training run
        (typically :func:`soup.train.launch_training` wrapped with
        :func:`soup.train.build_run_config`).

    Returns
    -------
    dict
        ``{run_id: checkpoint_path}`` for L1–L4.
    """
    output_dir = Path(output_dir)
    checkpoints: Dict[str, Path] = {}
    for run_id in ("L1", "L2", "L3", "L4"):
        run_output = output_dir / "checkpoints" / run_id
        run_output.mkdir(parents=True, exist_ok=True)
        setup_seed(RUN_CONFIGS[run_id].seed)
        trainer_fn(run_id, run_output, str(theta0_path))
        # The trainer writes model_final.pth or model_best.pth
        ckpt = run_output / "model_final.pth"
        if ckpt.exists():
            log_checkpoint_hash(ckpt, output_dir / "sha256" / "checkpoint_hashes.json")
        checkpoints[run_id] = ckpt
        logger.info("Step 3: %s training complete → %s.", run_id, ckpt)
    return checkpoints


# ---------------------------------------------------------------------------
# Step 4 – Extended pool training (C1-C2)
# ---------------------------------------------------------------------------


def step4_train_c_runs(
    output_dir: Union[str, Path],
    theta0_path: Union[str, Path],
    trainer_fn: Callable[[str, Path, str], None],
) -> Dict[str, Path]:
    """Step 4: train runs C1-C2 on the RTX 5090 (batch 16).

    Parameters
    ----------
    output_dir:
        Root experiment output directory.
    theta0_path:
        Path to shared θ₀ checkpoint.
    trainer_fn:
        Training launcher function.

    Returns
    -------
    dict
        ``{run_id: checkpoint_path}`` for C1–C2.
    """
    output_dir = Path(output_dir)
    checkpoints: Dict[str, Path] = {}
    for run_id in ("C1", "C2"):
        run_output = output_dir / "checkpoints" / run_id
        run_output.mkdir(parents=True, exist_ok=True)
        setup_seed(RUN_CONFIGS[run_id].seed)
        trainer_fn(run_id, run_output, str(theta0_path))
        ckpt = run_output / "model_final.pth"
        if ckpt.exists():
            log_checkpoint_hash(ckpt, output_dir / "sha256" / "checkpoint_hashes.json")
        checkpoints[run_id] = ckpt
        logger.info("Step 4: %s training complete → %s.", run_id, ckpt)
    return checkpoints


# ---------------------------------------------------------------------------
# Step 5 – Ingredient quality audit
# ---------------------------------------------------------------------------


def step5_ingredient_audit(
    checkpoint_metrics: Dict[str, Dict[str, float]],
    output_dir: Union[str, Path],
    threshold_pp: float = 3.0,
) -> Tuple[List[str], List[str]]:
    """Step 5: evaluate all six pool checkpoints and flag low-quality models.

    Parameters
    ----------
    checkpoint_metrics:
        ``{run_id: {"AP": float, "AP50": float, "AR100": float}}``.
    output_dir:
        Root experiment output directory (results saved as Table 4.1 JSON).
    threshold_pp:
        Exclusion threshold in mAP percentage points (default 3.0).

    Returns
    -------
    included, excluded : Tuple[List[str], List[str]]
    """
    included, excluded = ingredient_quality_audit(checkpoint_metrics, threshold_pp)

    table_path = Path(output_dir) / "logs" / "table_4_1_ingredient_audit.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w") as fh:
        json.dump(
            {
                "checkpoint_metrics": checkpoint_metrics,
                "included": included,
                "excluded": excluded,
                "threshold_pp": threshold_pp,
            },
            fh,
            indent=2,
        )
    logger.info("Step 5 complete. Table 4.1 written to %s.", table_path)
    return included, excluded


# ---------------------------------------------------------------------------
# Step 6 – Per-branch loss landscape measurement
# ---------------------------------------------------------------------------


def step6_loss_landscape(
    state_dicts: Dict[str, StateDict],
    loss_fn: Callable[[StateDict], float],
    output_dir: Union[str, Path],
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Step 6: per-branch loss barriers for all C(N,2) pairs.

    Parameters
    ----------
    state_dicts:
        ``{run_id: state_dict}`` for all included ingredient models.
    loss_fn:
        Calibration loss callable (state dict → scalar loss on 1 000-image
        COCO subset).
    output_dir:
        Root experiment output directory.

    Returns
    -------
    dict
        Barrier results keyed by ``(run_id_a, run_id_b, component)``.
    """
    run_ids = list(state_dicts.keys())
    sds = [state_dicts[r] for r in run_ids]
    barriers = measure_all_barriers(sds, run_ids, loss_fn)

    barriers_path = Path(output_dir) / "barriers" / "loss_barriers.npy"
    save_barriers_npy(barriers, barriers_path)

    logger.info("Step 6 complete. Barrier arrays saved to %s.", barriers_path)
    return barriers


# ---------------------------------------------------------------------------
# Step 7 – Merging experiments: Conditions 1-4
# ---------------------------------------------------------------------------


def step7_merging(
    state_dicts: Dict[str, StateDict],
    output_dir: Union[str, Path],
    eval_fn: Callable[[StateDict], float],
    mini_val_eval_fn: Optional[Callable[[StateDict], float]],
    fisher_traces: Optional[Dict[str, Dict[str, float]]] = None,
    dirichlet_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Step 7: run all four merging conditions and return their metrics.

    Parameters
    ----------
    state_dicts:
        ``{run_id: state_dict}`` for all included ingredient models.
    output_dir:
        Root experiment output directory.
    eval_fn:
        Full COCO val2017 evaluation callable (state dict → mAP50:95).
    mini_val_eval_fn:
        500-image mini-val evaluation callable for Condition 3 Stage A.
    fisher_traces:
        ``{run_id: {"cls": float, "reg": float}}`` for Condition 4.
        If ``None``, Condition 4 is skipped.
    dirichlet_seed:
        Optional seed for the Dirichlet sampler.

    Returns
    -------
    dict
        ``{condition: {"soup": StateDict, "ap": float, "lambda_cls": …,
        "lambda_reg": …}}``.
    """
    run_ids = list(state_dicts.keys())
    sds = [state_dicts[r] for r in run_ids]
    output_dir = Path(output_dir)
    lambda_dir = output_dir / "lambda_vectors"
    merge_dir = output_dir / "merges"
    lambda_dir.mkdir(parents=True, exist_ok=True)
    merge_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    n = len(run_ids)
    uniform_lam = np.full(n, 1.0 / n)

    # ---- Condition 1: global uniform soup ----
    soup_c1 = merge_uniform(sds)
    ap_c1 = eval_fn(soup_c1)
    results["condition_1"] = {
        "soup": soup_c1,
        "ap": ap_c1,
        "lambda_cls": uniform_lam.tolist(),
        "lambda_reg": uniform_lam.tolist(),
    }
    log_lambda_vectors(
        "condition_1_global_uniform",
        run_ids,
        uniform_lam.tolist(),
        uniform_lam.tolist(),
        lambda_dir / "condition_1.json",
    )
    logger.info("Condition 1 (global uniform) AP = %.4f.", ap_c1)

    # ---- Condition 2: branch uniform soup ----
    soup_c2 = merge_branch_uniform(sds)
    ap_c2 = eval_fn(soup_c2)
    results["condition_2"] = {
        "soup": soup_c2,
        "ap": ap_c2,
        "lambda_cls": uniform_lam.tolist(),
        "lambda_reg": uniform_lam.tolist(),
    }
    log_lambda_vectors(
        "condition_2_branch_uniform",
        run_ids,
        uniform_lam.tolist(),
        uniform_lam.tolist(),
        lambda_dir / "condition_2.json",
    )
    logger.info("Condition 2 (branch uniform) AP = %.4f.", ap_c2)

    # ---- Condition 3: branch Dirichlet search ----
    soup_c3, lam_cls_c3, lam_reg_c3 = merge_branch_dirichlet(
        sds,
        eval_fn=eval_fn,
        mini_val_eval_fn=mini_val_eval_fn,
        seed=dirichlet_seed,
    )
    ap_c3 = eval_fn(soup_c3)
    results["condition_3"] = {
        "soup": soup_c3,
        "ap": ap_c3,
        "lambda_cls": lam_cls_c3.tolist(),
        "lambda_reg": lam_reg_c3.tolist(),
    }
    log_lambda_vectors(
        "condition_3_dirichlet",
        run_ids,
        lam_cls_c3.tolist(),
        lam_reg_c3.tolist(),
        lambda_dir / "condition_3.json",
    )
    logger.info("Condition 3 (Dirichlet) AP = %.4f.", ap_c3)

    # ---- Condition 4: Fisher-weighted branch soup (optional) ----
    if fisher_traces is not None:
        soup_c4, lam_cls_c4, lam_reg_c4 = merge_branch_fisher(sds, fisher_traces)
        ap_c4 = eval_fn(soup_c4)
        results["condition_4"] = {
            "soup": soup_c4,
            "ap": ap_c4,
            "lambda_cls": lam_cls_c4.tolist(),
            "lambda_reg": lam_reg_c4.tolist(),
        }
        log_lambda_vectors(
            "condition_4_fisher",
            run_ids,
            lam_cls_c4.tolist(),
            lam_reg_c4.tolist(),
            lambda_dir / "condition_4.json",
        )
        logger.info("Condition 4 (Fisher) AP = %.4f.", ap_c4)
    else:
        logger.info("Condition 4 skipped (no Fisher traces provided).")

    # Summary
    summary = {k: v["ap"] for k, v in results.items()}
    summary_path = output_dir / "logs" / "merging_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Step 7 complete. Merging summary: %s.", summary)
    return results


# ---------------------------------------------------------------------------
# Step 8 – Source checkpoint pre-registration for D1 and D2
# ---------------------------------------------------------------------------


def step8_preregister_sources(
    merging_results: Dict[str, Any],
    output_dir: Union[str, Path],
) -> Dict[str, str]:
    """Step 8: pre-register D1 and D2 source checkpoints before training.

    D1 source = soup_condition2_N6.pth  (Condition 2, branch uniform soup)
    D2 source = soup_condition_best_N6.pth  (best of Conditions 3-4 by AP)

    Parameters
    ----------
    merging_results:
        Output of :func:`step7_merging`.
    output_dir:
        Root experiment output directory.

    Returns
    -------
    dict
        ``{"D1": str_source_label, "D2": str_source_label}``.
    """
    output_dir = Path(output_dir)

    d1_source = "condition_2"

    learned_conditions = {
        k: v for k, v in merging_results.items()
        if k in ("condition_3", "condition_4")
    }
    if not learned_conditions:
        raise ValueError(
            "At least one of Condition 3 or Condition 4 must be present in "
            "merging_results to pre-register D2 source."
        )
    d2_source = max(learned_conditions, key=lambda k: learned_conditions[k]["ap"])

    registration = {
        "D1_source": d1_source,
        "D2_source": d2_source,
        "D1_ap": merging_results[d1_source]["ap"],
        "D2_ap": merging_results[d2_source]["ap"],
    }

    reg_path = output_dir / "logs" / "step8_source_preregistration.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_path, "w") as fh:
        json.dump(registration, fh, indent=2)

    logger.info(
        "Step 8: D1 source=%s (AP=%.4f), D2 source=%s (AP=%.4f).  "
        "Committed to %s.",
        d1_source, registration["D1_ap"],
        d2_source, registration["D2_ap"],
        reg_path,
    )
    return {"D1": d1_source, "D2": d2_source}


# ---------------------------------------------------------------------------
# Step 9 – Head-only fine-tuning (D1, D2)
# ---------------------------------------------------------------------------


def step9_head_finetune(
    output_dir: Union[str, Path],
    source_state_dicts: Dict[str, StateDict],
    trainer_fn: Callable[[str, Path, StateDict], None],
) -> Dict[str, Path]:
    """Step 9: head-only fine-tuning for runs D1 and D2.

    Parameters
    ----------
    output_dir:
        Root experiment output directory.
    source_state_dicts:
        ``{"D1": soup_state_dict, "D2": soup_state_dict}``.
    trainer_fn:
        ``(run_id, output_dir, source_state_dict) → None``.

    Returns
    -------
    dict
        ``{run_id: checkpoint_path}`` for D1 and D2.
    """
    output_dir = Path(output_dir)
    checkpoints: Dict[str, Path] = {}
    for run_id in ("D1", "D2"):
        run_output = output_dir / "checkpoints" / run_id
        run_output.mkdir(parents=True, exist_ok=True)
        setup_seed(RUN_CONFIGS[run_id].seed)
        trainer_fn(run_id, run_output, source_state_dicts[run_id])
        ckpt = run_output / "model_final.pth"
        if ckpt.exists():
            log_checkpoint_hash(ckpt, output_dir / "sha256" / "checkpoint_hashes.json")
        checkpoints[run_id] = ckpt
        logger.info("Step 9: %s fine-tuning complete → %s.", run_id, ckpt)
    return checkpoints


# ---------------------------------------------------------------------------
# Step 10 – Final pipeline: Run C3
# ---------------------------------------------------------------------------


def step10_final_pipeline(
    output_dir: Union[str, Path],
    best_soup_state_dict: StateDict,
    trainer_fn: Callable[[str, Path, StateDict], None],
    eval_fn: Callable[[StateDict], float],
) -> Tuple[Path, float]:
    """Step 10: final pipeline run (C3) – best merge + head fine-tune.

    Parameters
    ----------
    output_dir:
        Root experiment output directory.
    best_soup_state_dict:
        Best merged soup from Step 7 (backbone+encoder frozen).
    trainer_fn:
        Training launcher function.
    eval_fn:
        Evaluation callable returning mAP50:95.

    Returns
    -------
    c3_checkpoint_path : Path
    c3_ap : float
        Final mAP50:95 on COCO val2017.
    """
    output_dir = Path(output_dir)
    run_output = output_dir / "checkpoints" / "C3"
    run_output.mkdir(parents=True, exist_ok=True)
    setup_seed(RUN_CONFIGS["C3"].seed)
    trainer_fn("C3", run_output, best_soup_state_dict)
    ckpt = run_output / "model_final.pth"
    if ckpt.exists():
        log_checkpoint_hash(ckpt, output_dir / "sha256" / "checkpoint_hashes.json")
    c3_ap = eval_fn(best_soup_state_dict)  # placeholder; caller should re-eval ckpt
    logger.info("Step 10: C3 complete → %s  (AP=%.4f).", ckpt, c3_ap)
    return ckpt, c3_ap


# ---------------------------------------------------------------------------
# Step 11 – Data logging and archiving
# ---------------------------------------------------------------------------


def step11_archive(
    output_dir: Union[str, Path],
    per_epoch_metrics: Dict[str, List[Dict[str, Any]]],
    hessian_traces: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Step 11: write per-epoch CSV logs, Hessian traces, and finalise archive.

    Parameters
    ----------
    output_dir:
        Root experiment output directory.
    per_epoch_metrics:
        ``{run_id: [{"epoch": int, "map50_95": float, …}, …]}``.
    hessian_traces:
        ``{run_id: {component: float}}``.  Written as ``.npy`` if provided.
    """
    output_dir = Path(output_dir)

    for run_id, metrics in per_epoch_metrics.items():
        csv_path = output_dir / "logs" / f"{run_id}_metrics.csv"
        save_metrics_csv(metrics, csv_path)

    if hessian_traces:
        npy_path = output_dir / "hessian" / "hessian_traces.npy"
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, hessian_traces)
        logger.info("Hessian traces saved to %s.", npy_path)

    logger.info("Step 11 (archiving) complete.")


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------


def run_full_pipeline(
    output_dir: Union[str, Path],
    theta0_path: Union[str, Path],
    trainer_fn: Callable,
    loss_fn: Callable[[StateDict], float],
    eval_fn: Callable[[StateDict], float],
    mini_val_eval_fn: Optional[Callable[[StateDict], float]] = None,
    load_state_dict_fn: Optional[Callable[[Path], StateDict]] = None,
    fisher_traces: Optional[Dict[str, Dict[str, float]]] = None,
    theta0_metrics: Optional[Dict[str, float]] = None,
    pilot_eval_fn: Optional[Callable[[], Dict[str, float]]] = None,
    dirichlet_seed: Optional[int] = None,
) -> None:
    """Execute all twelve steps of the data collection procedure sequentially.

    Parameters
    ----------
    output_dir:
        Root experiment output directory.
    theta0_path:
        Path to the shared θ₀ pre-trained checkpoint.
    trainer_fn:
        Training launcher: ``(run_id, out_dir, source) → None``.
    loss_fn:
        Calibration loss callable (state dict → scalar loss) for Step 6.
    eval_fn:
        Full COCO val2017 mAP50:95 callable (state dict → float) for Steps
        7 and 10.
    mini_val_eval_fn:
        500-image mini-val callable for Condition 3 Stage A.  Falls back to
        *eval_fn* if ``None``.
    load_state_dict_fn:
        Function to load a ``.pth`` file into a ``StateDict``.  If ``None``,
        a simple ``torch.load`` wrapper is used.
    fisher_traces:
        Pre-computed Fisher traces for Condition 4 (optional).
    theta0_metrics:
        mAP metrics for θ₀ (required for Step 2b pilot check).
    pilot_eval_fn:
        Callable to execute the 2-epoch pilot run for Step 2b.
    dirichlet_seed:
        Seed for the Dirichlet sampler in Condition 3.
    """
    # ---- Step 1 ----
    out = step1_setup(output_dir)

    # ---- Step 2 ----
    # (theta0_path already provided)
    log_checkpoint_hash(theta0_path, out / "sha256" / "checkpoint_hashes.json")

    # ---- Step 2b ----
    if theta0_metrics is not None and pilot_eval_fn is not None:
        step2b_pilot_check(out, theta0_metrics, pilot_eval_fn)

    # ---- Steps 3 & 4 ----
    step3_train_l_runs(out, theta0_path, trainer_fn)
    step4_train_c_runs(out, theta0_path, trainer_fn)

    # ---- Load ingredient state dicts ----
    if load_state_dict_fn is None:
        try:
            import torch

            def load_state_dict_fn(p: Path) -> StateDict:  # type: ignore[misc]
                ckpt = torch.load(p, map_location="cpu")
                return {k: v.numpy() for k, v in ckpt.get("model", ckpt).items()}
        except ImportError:
            raise ImportError(
                "PyTorch is required to load checkpoints in the default pipeline. "
                "Supply a custom load_state_dict_fn."
            )

    ingredient_state_dicts: Dict[str, StateDict] = {}
    ingredient_metrics: Dict[str, Dict[str, float]] = {}
    for run_id in INGREDIENT_RUN_IDS:
        ckpt_path = out / "checkpoints" / run_id / "model_final.pth"
        if ckpt_path.exists():
            ingredient_state_dicts[run_id] = load_state_dict_fn(ckpt_path)

    # ---- Step 5 ----
    included, _ = step5_ingredient_audit(ingredient_metrics, out)
    active_sds = {r: ingredient_state_dicts[r] for r in included if r in ingredient_state_dicts}

    # ---- Step 6 ----
    step6_loss_landscape(active_sds, loss_fn, out)

    # ---- Step 7 ----
    merging_results = step7_merging(
        active_sds,
        out,
        eval_fn,
        mini_val_eval_fn,
        fisher_traces=fisher_traces,
        dirichlet_seed=dirichlet_seed,
    )

    # ---- Step 8 ----
    source_map = step8_preregister_sources(merging_results, out)

    # ---- Step 9 ----
    source_state_dicts = {
        "D1": merging_results[source_map["D1"]]["soup"],
        "D2": merging_results[source_map["D2"]]["soup"],
    }
    step9_head_finetune(out, source_state_dicts, trainer_fn)

    # ---- Step 10 ----
    best_condition = max(merging_results, key=lambda k: merging_results[k]["ap"])
    step10_final_pipeline(out, merging_results[best_condition]["soup"], trainer_fn, eval_fn)

    # ---- Step 11 ----
    step11_archive(out, per_epoch_metrics={})

    logger.info("Full pipeline complete. All outputs in %s.", out)
