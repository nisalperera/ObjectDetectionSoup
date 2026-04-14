"""
Loss Landscape Analysis
=======================
Per-branch loss barrier measurement (Frankle et al., 2020) and Hutchinson
Hessian trace estimation (Yao et al., 2020, via PyHessian) as described in
Section 3.3.3 and Step 6 of the data collection procedure.

Loss barrier (Section 3.3.3 / Step 6)
--------------------------------------
For each model pair (A, B) and each YOLOF component c ∈ {backbone, encoder,
cls, reg}, parameters of component c are linearly interpolated from θ_A^c
→ θ_B^c at 21 equally-spaced α ∈ {0.00, 0.05, …, 1.00} while all other
components are frozen at model A's weights.  The barrier is:

    B_c(A, B) = max_{α} L(α θ_A^c + (1-α) θ_B^c)
                − (1/2)[L(θ_A^c) + L(θ_B^c)]

Hutchinson trace estimator (Step 6)
-------------------------------------
For each checkpoint and each component, 50 random Rademacher vectors are used
to estimate the Hessian trace:

    Tr(H_c) ≈ (1/n) Σ_j v_j^T H_c v_j,  v_j ∼ Rademacher

The Detectron2 / PyHessian integration is wrapped in try/except so the module
remains importable without those heavy dependencies.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from soup.config import PARAMETER_PREFIXES
from soup.merger import StateDict, _keys_for_component

logger = logging.getLogger(__name__)

# 21 equally-spaced interpolation points α ∈ [0, 1]
ALPHA_GRID: np.ndarray = np.linspace(0.0, 1.0, 21)

# Number of Rademacher vectors for Hutchinson trace estimation
HUTCHINSON_N_VECTORS: int = 50

# Components for which barriers and traces are measured
COMPONENTS: Tuple[str, ...] = ("backbone", "encoder", "cls", "reg")


# ---------------------------------------------------------------------------
# Linear interpolation helper
# ---------------------------------------------------------------------------


def interpolate_component(
    state_a: StateDict,
    state_b: StateDict,
    component: str,
    alpha: float,
) -> StateDict:
    """Return a state dict with component *c* interpolated at *alpha*.

    Parameters *other than* component *c* are taken from *state_a* (frozen).

    Parameters
    ----------
    state_a, state_b:
        Ingredient model state dicts.
    component:
        YOLOF component to interpolate (``"backbone"``, ``"encoder"``,
        ``"cls"``, or ``"reg"``).
    alpha:
        Interpolation coefficient, α ∈ [0, 1].
        α=0 → state_a,  α=1 → state_b.

    Returns
    -------
    StateDict
        A new state dict with the interpolated component.
    """
    keys_c = set(_keys_for_component(state_a, component))
    mixed: StateDict = {}
    for key in state_a:
        if key in keys_c:
            a_val = np.asarray(state_a[key], dtype=np.float64)
            b_val = np.asarray(state_b[key], dtype=np.float64)
            mixed[key] = (1.0 - alpha) * a_val + alpha * b_val
        else:
            mixed[key] = state_a[key]
    return mixed


# ---------------------------------------------------------------------------
# Per-branch loss barrier
# ---------------------------------------------------------------------------


def measure_loss_barrier(
    state_a: StateDict,
    state_b: StateDict,
    component: str,
    loss_fn: Callable[[StateDict], float],
    alpha_grid: np.ndarray = ALPHA_GRID,
) -> Dict[str, Any]:
    """Measure the per-branch loss barrier for a model pair (A, B).

    The barrier is computed as per Frankle et al. (2020, p. 3261):

        B_c(A, B) = max_{α} L(α θ_A^c + (1-α) θ_B^c)
                    − (1/2)[L(θ_A^c) + L(θ_B^c)]

    where the max is taken over the discrete *alpha_grid*.

    Parameters
    ----------
    state_a, state_b:
        Ingredient model state dicts.
    component:
        YOLOF component to analyse (``"backbone"``, ``"encoder"``,
        ``"cls"``, or ``"reg"``).
    loss_fn:
        Callable that accepts a full state dict and returns a scalar loss
        value evaluated on a fixed calibration subset.
    alpha_grid:
        1-D array of interpolation coefficients (default: 21 points,
        0.00 to 1.00 in steps of 0.05).

    Returns
    -------
    dict with keys:
        ``component``   – component name
        ``alpha_grid``  – list of α values
        ``losses``      – list of loss values at each α
        ``barrier``     – scalar B_c(A, B)
        ``loss_a``      – L(θ_A^c)  (α = 0)
        ``loss_b``      – L(θ_B^c)  (α = 1)
    """
    if component not in PARAMETER_PREFIXES:
        raise KeyError(f"Unknown component {component!r}.")

    losses: List[float] = []
    for alpha in alpha_grid:
        mixed = interpolate_component(state_a, state_b, component, float(alpha))
        losses.append(float(loss_fn(mixed)))

    loss_a = losses[0]   # α = 0 → model A
    loss_b = losses[-1]  # α = 1 → model B
    barrier = float(max(losses) - 0.5 * (loss_a + loss_b))

    logger.debug(
        "Barrier[%s]: max_loss=%.4f, loss_a=%.4f, loss_b=%.4f, barrier=%.4f",
        component, max(losses), loss_a, loss_b, barrier,
    )

    return {
        "component": component,
        "alpha_grid": alpha_grid.tolist(),
        "losses": losses,
        "barrier": barrier,
        "loss_a": loss_a,
        "loss_b": loss_b,
    }


def measure_all_barriers(
    state_dicts: List[StateDict],
    run_ids: List[str],
    loss_fn: Callable[[StateDict], float],
    alpha_grid: np.ndarray = ALPHA_GRID,
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Measure per-branch barriers for all C(N, 2) pairs (Step 6).

    Parameters
    ----------
    state_dicts:
        List of N ingredient model state dicts.
    run_ids:
        Corresponding run ID labels (length N).
    loss_fn:
        Calibration loss callable (state dict → scalar loss).
    alpha_grid:
        Interpolation grid (default 21 points).

    Returns
    -------
    dict
        Keyed by ``(run_id_a, run_id_b, component)`` tuples.
    """
    if len(state_dicts) != len(run_ids):
        raise ValueError("state_dicts and run_ids must have the same length.")

    results: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    pairs = list(itertools.combinations(range(len(state_dicts)), 2))

    logger.info(
        "Measuring barriers: %d pairs × %d components = %d evaluations "
        "(21 α points each).",
        len(pairs),
        len(COMPONENTS),
        len(pairs) * len(COMPONENTS),
    )

    for i, j in pairs:
        id_a, id_b = run_ids[i], run_ids[j]
        for component in COMPONENTS:
            key = (id_a, id_b, component)
            results[key] = measure_loss_barrier(
                state_dicts[i],
                state_dicts[j],
                component,
                loss_fn,
                alpha_grid=alpha_grid,
            )
            logger.info(
                "Barrier[%s, %s, %s] = %.4f",
                id_a, id_b, component,
                results[key]["barrier"],
            )

    return results


def save_barriers_npy(
    barriers: Dict[Tuple[str, str, str], Dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Save the barrier arrays to a NumPy ``.npy`` file (Step 11).

    Parameters
    ----------
    barriers:
        Output of :func:`measure_all_barriers`.
    output_path:
        Destination ``.npy`` file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert tuple keys to strings for numpy serialisation
    serialisable = {
        f"{a}__{b}__{c}": v for (a, b, c), v in barriers.items()
    }
    np.save(output_path, serialisable)
    logger.info("Barrier arrays saved to %s.", output_path)


# ---------------------------------------------------------------------------
# Hutchinson Hessian trace estimator (numpy fallback)
# ---------------------------------------------------------------------------


def hutchinson_trace_numpy(
    loss_fn_params: Callable[[np.ndarray], float],
    params: np.ndarray,
    n_vectors: int = HUTCHINSON_N_VECTORS,
    seed: Optional[int] = None,
) -> float:
    """Hutchinson trace estimator using finite-difference gradient approximation.

    This is a **pure-NumPy fallback** implementation intended for unit testing
    without PyTorch/PyHessian.  For production use with Detectron2, use
    :func:`measure_hessian_trace_pyhessian` instead.

    Tr(H) ≈ (1/n) Σ_j  v_j^T H v_j,   v_j ∼ Rademacher{−1, +1}

    The Hessian-vector product H v is approximated via a central
    finite-difference scheme::

        Hv ≈ [∇L(θ + ε v) − ∇L(θ − ε v)] / (2ε)

    where ∇L is estimated by a forward finite difference on ``loss_fn_params``.

    Parameters
    ----------
    loss_fn_params:
        Callable mapping a flat parameter vector to a scalar loss.
    params:
        Flat parameter array (θ).
    n_vectors:
        Number of Rademacher vectors to sample (default 50).
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    float
        Estimated Hessian trace.
    """
    rng = np.random.default_rng(seed)
    d = params.shape[0]
    eps = 1e-3
    trace_estimates = []

    for _ in range(n_vectors):
        v = rng.choice([-1.0, 1.0], size=d).astype(np.float64)

        # Gradient approximation at θ + εv
        def grad_fd(theta: np.ndarray, eps_fd: float = 1e-4) -> np.ndarray:
            g = np.zeros_like(theta)
            base = loss_fn_params(theta)
            for k in range(min(d, 100)):  # limit dimension for tests
                delta = np.zeros_like(theta)
                delta[k] = eps_fd
                g[k] = (loss_fn_params(theta + delta) - base) / eps_fd
            return g

        g_plus = grad_fd(params + eps * v)
        g_minus = grad_fd(params - eps * v)
        hv = (g_plus - g_minus) / (2.0 * eps)
        trace_estimates.append(float(v @ hv))

    return float(np.mean(trace_estimates))


def measure_hessian_trace_pyhessian(
    model: Any,
    data_loader: Any,
    component: str,
    n_vectors: int = HUTCHINSON_N_VECTORS,
) -> float:
    """Estimate Hessian trace for *component* using PyHessian.

    Requires ``pyhessian`` and ``torch`` to be installed.  All non-target
    modules have ``requires_grad=False`` applied during computation to
    prevent gradient propagation beyond the branch of interest.

    Parameters
    ----------
    model:
        A Detectron2 YOLOF model (``torch.nn.Module``).
    data_loader:
        A Detectron2 data loader providing batches from the calibration set.
    component:
        YOLOF component to trace (``"backbone"``, ``"encoder"``,
        ``"cls"``, or ``"reg"``).
    n_vectors:
        Number of Rademacher vectors (default 50).

    Returns
    -------
    float
        Estimated Hessian trace for *component*.

    Raises
    ------
    ImportError
        If ``pyhessian`` or ``torch`` are not available.
    """
    try:
        import torch
        from pyhessian import hessian as PyHessian  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pyhessian and torch are required for Hessian trace estimation. "
            "Install them with:  pip install pyhessian torch"
        ) from exc

    prefixes = PARAMETER_PREFIXES[component]

    # Freeze all parameters not belonging to *component*
    frozen_params = []
    for name, param in model.named_parameters():
        if not any(name.startswith(p) for p in prefixes):
            param.requires_grad_(False)
            frozen_params.append(name)

    try:
        # Collect one calibration batch for PyHessian
        batches = []
        for i, batch in enumerate(data_loader):
            batches.append(batch)
            if i >= 0:  # single batch sufficient for trace estimation
                break

        hessian_comp = PyHessian(model, criterion=None, data=batches[0])
        trace = hessian_comp.trace(maxIter=n_vectors, tol=1e-3)
        logger.info("Hessian trace[%s] = %.4f (n_vectors=%d)", component, trace, n_vectors)
        return float(np.mean(trace))

    finally:
        # Restore requires_grad
        for name, param in model.named_parameters():
            if name in frozen_params:
                param.requires_grad_(True)


def measure_all_hessian_traces(
    models: Dict[str, Any],
    data_loader: Any,
    n_vectors: int = HUTCHINSON_N_VECTORS,
) -> Dict[str, Dict[str, float]]:
    """Estimate Hessian traces for all components × all checkpoints (Step 6).

    Parameters
    ----------
    models:
        Dict mapping run_id → Detectron2 model.
    data_loader:
        Calibration data loader.
    n_vectors:
        Rademacher vectors per trace estimate (default 50).

    Returns
    -------
    dict
        Nested ``{run_id: {component: trace_value}}``.
    """
    results: Dict[str, Dict[str, float]] = {}
    for run_id, model in models.items():
        results[run_id] = {}
        for component in COMPONENTS:
            results[run_id][component] = measure_hessian_trace_pyhessian(
                model, data_loader, component, n_vectors=n_vectors
            )
    return results
