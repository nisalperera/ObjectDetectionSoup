"""
Utility helpers
===============
SHA-256 hashing, random-seed setup, and JSON logging utilities used
across all experimental steps.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SHA-256 hashing
# ---------------------------------------------------------------------------


def compute_sha256(filepath: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    """Return the hex-encoded SHA-256 digest of *filepath*.

    Parameters
    ----------
    filepath:
        Path to the file to hash.
    chunk_size:
        Read chunk size in bytes (default 1 MiB).

    Returns
    -------
    str
        Lowercase hex digest, e.g. ``"d41d8cd98f00b204e9800998ecf8427e"``.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def log_checkpoint_hash(
    checkpoint_path: Union[str, Path],
    log_path: Union[str, Path, None] = None,
) -> str:
    """Compute and optionally persist the SHA-256 hash of *checkpoint_path*.

    If *log_path* is provided, the mapping ``{checkpoint_name: hash}`` is
    appended (or created) in a JSON file at that path.

    Returns
    -------
    str
        The SHA-256 hex digest.
    """
    checkpoint_path = Path(checkpoint_path)
    digest = compute_sha256(checkpoint_path)
    logger.info("SHA-256 %s  %s", digest, checkpoint_path.name)

    if log_path is not None:
        log_path = Path(log_path)
        existing: Dict[str, str] = {}
        if log_path.exists():
            with open(log_path, "r") as fh:
                existing = json.load(fh)
        existing[checkpoint_path.name] = digest
        with open(log_path, "w") as fh:
            json.dump(existing, fh, indent=2)

    return digest


# ---------------------------------------------------------------------------
# Reproducibility / seed setup
# ---------------------------------------------------------------------------


def setup_seed(seed: int) -> None:
    """Fix random seeds for Python, NumPy, and (if available) PyTorch.

    This is the pure-Python / NumPy portion of reproducibility anchoring.
    When Detectron2 is available, ``seed_all_rng(seed)`` from
    ``detectron2.utils.env`` should be called **in addition** to this
    function (see :func:`setup_detectron2_seed`).

    Parameters
    ----------
    seed:
        Integer seed value from Table 3.1.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch  # noqa: F401 – optional dependency

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("PyTorch seed set to %d (deterministic=True, benchmark=False)", seed)
    except ImportError:
        logger.debug("PyTorch not available; skipping torch seed setup.")


def setup_detectron2_seed(seed: int) -> None:
    """Call Detectron2's ``seed_all_rng`` before any data loading or model init.

    This wraps the Detectron2-specific seeding utility and is separated from
    :func:`setup_seed` so that the base seed setup remains testable without
    a Detectron2 installation.

    Parameters
    ----------
    seed:
        Integer seed value from Table 3.1.
    """
    try:
        from detectron2.utils.env import seed_all_rng  # type: ignore[import]

        seed_all_rng(seed)
        logger.info("Detectron2 seed_all_rng called with seed=%d", seed)
    except ImportError:
        logger.warning(
            "Detectron2 not installed; falling back to generic seed setup only."
        )
        setup_seed(seed)


# ---------------------------------------------------------------------------
# JSON lambda-vector logging
# ---------------------------------------------------------------------------


def log_lambda_vectors(
    condition: str,
    run_ids: List[str],
    lambda_cls: List[float],
    lambda_reg: List[float],
    output_path: Union[str, Path],
) -> None:
    """Persist the λ_cls and λ_reg coefficient vectors to a JSON file.

    The vectors are logged as required by Section 3.3.4 and Step 7 of the
    data collection procedure.

    Parameters
    ----------
    condition:
        Human-readable condition label, e.g. ``"condition_3_dirichlet"``.
    run_ids:
        Ordered list of ingredient run IDs corresponding to each coefficient.
    lambda_cls:
        Per-model classification-branch coefficients (must sum to ≈ 1).
    lambda_reg:
        Per-model regression-branch coefficients (must sum to ≈ 1).
    output_path:
        Destination JSON file path.

    Raises
    ------
    ValueError
        If the lengths of *run_ids*, *lambda_cls*, and *lambda_reg* differ.
    """
    if not (len(run_ids) == len(lambda_cls) == len(lambda_reg)):
        raise ValueError(
            "run_ids, lambda_cls, and lambda_reg must have the same length; "
            f"got {len(run_ids)}, {len(lambda_cls)}, {len(lambda_reg)}."
        )

    payload: Dict[str, Any] = {
        "condition": condition,
        "run_ids": run_ids,
        "lambda_cls": list(lambda_cls),
        "lambda_reg": list(lambda_reg),
        "sum_lambda_cls": float(sum(lambda_cls)),
        "sum_lambda_reg": float(sum(lambda_reg)),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    logger.info(
        "λ vectors for %s written to %s  (sum_cls=%.6f, sum_reg=%.6f)",
        condition,
        output_path,
        payload["sum_lambda_cls"],
        payload["sum_lambda_reg"],
    )


def save_metrics_csv(
    metrics: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> None:
    """Append per-epoch metric rows to a CSV file.

    Creates the file (with header) if it does not exist; appends rows
    otherwise.  Each dict in *metrics* must share the same keys.

    Parameters
    ----------
    metrics:
        List of dicts, one per epoch, e.g.
        ``[{"epoch": 1, "map50_95": 42.1, "map50": 60.3, "ar100": 55.2}]``.
    output_path:
        Destination CSV path.
    """
    import csv

    if not metrics:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(metrics[0].keys())
    file_exists = output_path.exists()

    with open(output_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(metrics)

    logger.info("Metrics written to %s (%d rows)", output_path, len(metrics))
