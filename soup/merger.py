"""
Core Merging Module
===================
Implements the two-stage parameter averaging procedure from Section 3.3.1
and all four merging conditions from Section 3.3.2.

Merging conditions
------------------
Condition 1  Global uniform soup          – entire decoder averaged uniformly.
Condition 2  Branch uniform soup          – cls / reg averaged separately with
                                            equal per-model weights.
Condition 3  Branch Dirichlet search      – coefficients found by two-stage
                                            Dirichlet random search on a
                                            mini-val split.
Condition 4  Fisher-weighted branch soup  – coefficients proportional to the
                                            trace of the empirical Fisher
                                            information matrix per branch.

State-dict format
-----------------
Each checkpoint is expected to be a plain Python ``dict`` mapping parameter
key strings to ``numpy.ndarray`` (or any array-like supporting arithmetic).
When used with Detectron2, the state dict is obtained via::

    import torch
    ckpt = torch.load(path, map_location="cpu")
    state = {k: v.numpy() for k, v in ckpt["model"].items()}

The merging functions are deliberately framework-agnostic (they work on
``numpy.ndarray`` objects) so they can be tested without Detectron2 or PyTorch.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from soup.config import PARAMETER_PREFIXES

logger = logging.getLogger(__name__)

# Type alias for a state dictionary (parameter name → array)
StateDict = Dict[str, np.ndarray]

# Number of Dirichlet candidates per branch (Stage A)
DIRICHLET_M: int = 100

# Number of top candidates promoted to Stage B confirmation
DIRICHLET_K: int = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_key_coverage(state_dicts: List[StateDict]) -> None:
    """Verify that all state dicts share the same parameter keys."""
    if len(state_dicts) < 2:
        return
    reference = set(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        if set(sd.keys()) != reference:
            extra = set(sd.keys()) - reference
            missing = reference - set(sd.keys())
            raise ValueError(
                f"State dict {i} has mismatched keys. "
                f"Extra: {extra}. Missing: {missing}."
            )


def _keys_for_component(state_dict: StateDict, component: str) -> List[str]:
    """Return all parameter keys belonging to *component*.

    Parameters
    ----------
    state_dict:
        A model state dictionary.
    component:
        One of ``"backbone"``, ``"encoder"``, ``"cls"``, ``"reg"``.

    Raises
    ------
    KeyError
        If *component* is not in :data:`soup.config.PARAMETER_PREFIXES`.
    """
    if component not in PARAMETER_PREFIXES:
        raise KeyError(
            f"Unknown component {component!r}. "
            f"Valid components: {list(PARAMETER_PREFIXES.keys())}."
        )
    prefixes = PARAMETER_PREFIXES[component]
    return [k for k in state_dict if any(k.startswith(p) for p in prefixes)]


def _validate_simplex(lambdas: np.ndarray, name: str = "lambda", tol: float = 1e-5) -> None:
    """Assert that *lambdas* satisfies the unit-simplex constraints."""
    if np.any(lambdas < -tol):
        raise ValueError(f"{name} contains negative values: {lambdas}.")
    total = float(np.sum(lambdas))
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"{name} does not sum to 1 (sum={total:.8f}, tol={tol})."
        )


def _uniform_lambdas(n: int) -> np.ndarray:
    """Return uniform simplex weights [1/N, …, 1/N]."""
    return np.full(n, 1.0 / n)


def _weighted_average(
    state_dicts: List[StateDict],
    keys: List[str],
    lambdas: np.ndarray,
) -> StateDict:
    """Return a partial state dict containing the weighted average for *keys*.

    Parameters
    ----------
    state_dicts:
        Ingredient model state dicts.
    keys:
        Parameter keys to average.
    lambdas:
        Per-model weights (must satisfy unit-simplex constraints).

    Returns
    -------
    StateDict
        A new dict containing only the averaged *keys*.
    """
    _validate_simplex(lambdas)
    result: StateDict = {}
    for key in keys:
        result[key] = sum(
            lambdas[i] * np.asarray(state_dicts[i][key], dtype=np.float64)
            for i in range(len(state_dicts))
        )
    return result


# ---------------------------------------------------------------------------
# Stage 1 – Uniform backbone + encoder merge (shared across all conditions)
# ---------------------------------------------------------------------------


def merge_backbone_encoder(state_dicts: List[StateDict]) -> StateDict:
    """Stage 1: uniform average of backbone and encoder parameters.

    Applies the formula:

        θ_backbone^soup = (1/N) Σ θ_backbone_i
        θ_encoder^soup  = (1/N) Σ θ_encoder_i

    as defined in Section 3.3.1.

    Parameters
    ----------
    state_dicts:
        List of N ingredient model state dicts (same key set required).

    Returns
    -------
    StateDict
        Partial state dict containing averaged backbone and encoder keys.
    """
    if not state_dicts:
        raise ValueError("state_dicts must not be empty.")
    _check_key_coverage(state_dicts)

    n = len(state_dicts)
    lambdas = _uniform_lambdas(n)
    merged: StateDict = {}

    for component in ("backbone", "encoder"):
        keys = _keys_for_component(state_dicts[0], component)
        merged.update(_weighted_average(state_dicts, keys, lambdas))

    return merged


# ---------------------------------------------------------------------------
# Stage 2 helpers – head merging with explicit λ vectors
# ---------------------------------------------------------------------------


def merge_head_weighted(
    state_dicts: List[StateDict],
    lambda_cls: np.ndarray,
    lambda_reg: np.ndarray,
) -> StateDict:
    """Stage 2: weighted merge of cls and reg branch parameters.

    Applies:

        θ_head^soup = Σ_i ( λ_cls_i · θ_cls_i  +  λ_reg_i · θ_reg_i )

    with the constraint that both λ_cls and λ_reg lie on their respective
    unit simplices (Section 3.3.1).

    Parameters
    ----------
    state_dicts:
        List of N ingredient model state dicts.
    lambda_cls:
        Per-model weights for the cls branch (length N, sums to 1).
    lambda_reg:
        Per-model weights for the reg branch (length N, sums to 1).

    Returns
    -------
    StateDict
        Partial state dict containing merged cls and reg branch keys.
    """
    if not state_dicts:
        raise ValueError("state_dicts must not be empty.")
    _check_key_coverage(state_dicts)

    lambda_cls = np.asarray(lambda_cls, dtype=np.float64)
    lambda_reg = np.asarray(lambda_reg, dtype=np.float64)
    _validate_simplex(lambda_cls, "lambda_cls")
    _validate_simplex(lambda_reg, "lambda_reg")

    merged: StateDict = {}
    for component, lambdas in (("cls", lambda_cls), ("reg", lambda_reg)):
        keys = _keys_for_component(state_dicts[0], component)
        merged.update(_weighted_average(state_dicts, keys, lambdas))

    return merged


def assemble_soup(
    stage1: StateDict,
    stage2: StateDict,
) -> StateDict:
    """Concatenate Stage-1 and Stage-2 partial state dicts into a full soup.

    Implements:

        θ_soup = θ_backbone^soup ⊕ θ_encoder^soup ⊕ θ_head^soup

    Parameters
    ----------
    stage1:
        Output of :func:`merge_backbone_encoder`.
    stage2:
        Output of :func:`merge_head_weighted` (or equivalent).

    Returns
    -------
    StateDict
        Complete merged model state dict.

    Raises
    ------
    ValueError
        If *stage1* and *stage2* share any parameter keys (would indicate
        double-counting).
    """
    overlap = set(stage1.keys()) & set(stage2.keys())
    if overlap:
        raise ValueError(
            f"stage1 and stage2 share {len(overlap)} key(s): "
            f"{sorted(overlap)[:5]} …"
        )
    return {**stage1, **stage2}


# ---------------------------------------------------------------------------
# Condition 1 – Global uniform soup
# ---------------------------------------------------------------------------


def merge_uniform(state_dicts: List[StateDict]) -> StateDict:
    """Condition 1: global uniform soup (no branch partition).

    The entire decoder (cls + reg branches) is averaged as a single block,
    identical to the standard model soup of Wortsman et al. (2022).

        λ_i = 1/N  ∀ i   (applied to backbone, encoder, cls, reg)

    Parameters
    ----------
    state_dicts:
        List of N ingredient model state dicts.

    Returns
    -------
    StateDict
        Fully merged state dict.
    """
    if not state_dicts:
        raise ValueError("state_dicts must not be empty.")
    _check_key_coverage(state_dicts)

    n = len(state_dicts)
    lambdas = _uniform_lambdas(n)
    merged: StateDict = {}

    for component in ("backbone", "encoder", "cls", "reg"):
        keys = _keys_for_component(state_dicts[0], component)
        merged.update(_weighted_average(state_dicts, keys, lambdas))

    logger.info("Condition 1 (global uniform soup) assembled with N=%d models.", n)
    return merged


# ---------------------------------------------------------------------------
# Condition 2 – Branch uniform soup
# ---------------------------------------------------------------------------


def merge_branch_uniform(state_dicts: List[StateDict]) -> StateDict:
    """Condition 2: branch uniform soup.

    Backbone and encoder undergo Stage-1 uniform merging; cls and reg
    branches are averaged separately but with equal per-model weights:

        λ_cls_i = 1/N  ∀ i,   λ_reg_i = 1/N  ∀ i

    The partition structure is introduced while coefficients remain equal.
    The mAP delta between Condition 1 and Condition 2 measures the isolated
    effect of branch partitioning (Section 3.3.2).

    Parameters
    ----------
    state_dicts:
        List of N ingredient model state dicts.

    Returns
    -------
    StateDict
        Fully merged state dict.
    """
    if not state_dicts:
        raise ValueError("state_dicts must not be empty.")
    _check_key_coverage(state_dicts)

    n = len(state_dicts)
    lambdas = _uniform_lambdas(n)

    stage1 = merge_backbone_encoder(state_dicts)
    stage2 = merge_head_weighted(state_dicts, lambdas, lambdas)
    soup = assemble_soup(stage1, stage2)

    logger.info("Condition 2 (branch uniform soup) assembled with N=%d models.", n)
    return soup


# ---------------------------------------------------------------------------
# Condition 3 – Branch Dirichlet random search
# ---------------------------------------------------------------------------


def _sample_dirichlet_candidates(
    n: int,
    m: int = DIRICHLET_M,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample *m* coefficient vectors from Dir(**1**_N) over the N-simplex.

    Uses ``numpy.random.dirichlet([1]*N, size=M)`` as specified in Step 7.

    Parameters
    ----------
    n:
        Dimension of the simplex (number of ingredient models).
    m:
        Number of candidate vectors to sample.
    rng:
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (M, N) array of coefficient vectors.
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.dirichlet(np.ones(n), size=m)


def merge_branch_dirichlet(
    state_dicts: List[StateDict],
    eval_fn: Callable[[StateDict], float],
    mini_val_eval_fn: Optional[Callable[[StateDict], float]] = None,
    m: int = DIRICHLET_M,
    k: int = DIRICHLET_K,
    seed: Optional[int] = None,
) -> Tuple[StateDict, np.ndarray, np.ndarray]:
    """Condition 3: branch random simplex search via two-stage Dirichlet procedure.

    Stage A – Exploration
        *M* = 100 candidate vectors are sampled independently for λ_cls and
        λ_reg from Dir(**1**_N).  Each λ_cls candidate is combined with the
        uniform λ_reg = **1**/N and scored on the 500-image COCO mini-val; the
        procedure is repeated for λ_reg with λ_cls held at **1**/N.

    Stage B – Confirmation
        The top *K* = 10 candidates per branch are re-evaluated on the full
        COCO val2017 (5 000 images).  The globally best λ*_cls and λ*_reg are
        selected and their vectors logged.

    Parameters
    ----------
    state_dicts:
        List of N ingredient model state dicts.
    eval_fn:
        Callable that accepts a full state dict and returns a scalar score
        (higher is better; typically mAP50:95 on COCO val2017, 0–100).
        Used in Stage B.
    mini_val_eval_fn:
        Callable used in Stage A (500-image mini-val).  If ``None``,
        *eval_fn* is used for both stages.
    m:
        Number of candidate vectors sampled in Stage A (default 100).
    k:
        Number of top candidates promoted to Stage B (default 10).
    seed:
        Optional integer seed for the Dirichlet sampler.

    Returns
    -------
    StateDict
        Fully merged state dict with the best λ*_cls and λ*_reg.
    np.ndarray
        Best λ_cls vector (length N).
    np.ndarray
        Best λ_reg vector (length N).
    """
    if not state_dicts:
        raise ValueError("state_dicts must not be empty.")
    _check_key_coverage(state_dicts)

    n = len(state_dicts)
    uniform = _uniform_lambdas(n)
    rng = np.random.default_rng(seed)
    stage_a_eval = mini_val_eval_fn if mini_val_eval_fn is not None else eval_fn

    stage1 = merge_backbone_encoder(state_dicts)

    def _score_candidate(lam_cls: np.ndarray, lam_reg: np.ndarray) -> float:
        stage2 = merge_head_weighted(state_dicts, lam_cls, lam_reg)
        soup = assemble_soup(stage1, stage2)
        return stage_a_eval(soup)

    # ---- Stage A: explore λ_cls with uniform λ_reg, then vice-versa ----
    candidates_cls = _sample_dirichlet_candidates(n, m, rng)
    scores_cls_a = np.array([
        _score_candidate(lam, uniform) for lam in candidates_cls
    ])

    candidates_reg = _sample_dirichlet_candidates(n, m, rng)
    scores_reg_a = np.array([
        _score_candidate(uniform, lam) for lam in candidates_reg
    ])

    logger.info(
        "Condition 3 Stage A: cls best=%.4f, reg best=%.4f (M=%d, N=%d)",
        scores_cls_a.max(), scores_reg_a.max(), m, n,
    )

    # ---- Stage B: confirm top-K candidates on full val2017 ----
    top_k_cls_idx = np.argsort(scores_cls_a)[-k:][::-1]
    top_k_reg_idx = np.argsort(scores_reg_a)[-k:][::-1]

    def _score_full(lam_cls: np.ndarray, lam_reg: np.ndarray) -> float:
        stage2 = merge_head_weighted(state_dicts, lam_cls, lam_reg)
        soup = assemble_soup(stage1, stage2)
        return eval_fn(soup)

    scores_cls_b = np.array([
        _score_full(candidates_cls[i], uniform) for i in top_k_cls_idx
    ])
    scores_reg_b = np.array([
        _score_full(uniform, candidates_reg[i]) for i in top_k_reg_idx
    ])

    best_lambda_cls = candidates_cls[top_k_cls_idx[int(np.argmax(scores_cls_b))]]
    best_lambda_reg = candidates_reg[top_k_reg_idx[int(np.argmax(scores_reg_b))]]

    logger.info(
        "Condition 3 Stage B: cls best=%.4f, reg best=%.4f (K=%d)",
        scores_cls_b.max(), scores_reg_b.max(), k,
    )

    # Assemble final merged model
    stage2_final = merge_head_weighted(state_dicts, best_lambda_cls, best_lambda_reg)
    soup_final = assemble_soup(stage1, stage2_final)

    logger.info(
        "Condition 3 (Dirichlet search) assembled. "
        "λ_cls=%s, λ_reg=%s",
        np.array2string(best_lambda_cls, precision=4),
        np.array2string(best_lambda_reg, precision=4),
    )
    return soup_final, best_lambda_cls, best_lambda_reg


# ---------------------------------------------------------------------------
# Condition 4 – Fisher-weighted branch soup
# ---------------------------------------------------------------------------


def _compute_fisher_trace_numpy(
    state_dict: StateDict,
    component: str,
    gradients: List[Dict[str, np.ndarray]],
) -> float:
    """Estimate the empirical Fisher trace for *component* from gradient samples.

    Given *G* gradient samples {g_j} computed on calibration data, the
    empirical Fisher trace for component *c* is approximated as:

        F_c ≈ (1/G) Σ_j Σ_{k ∈ c} g_j[k]²

    This mirrors the approach of Matena and Raffel (2022, pp. 2-4).

    Parameters
    ----------
    state_dict:
        Model state dict (used only to identify component keys).
    component:
        One of ``"cls"`` or ``"reg"``.
    gradients:
        List of gradient state dicts – each has the same keys as *state_dict*,
        with values being the parameter gradients from one calibration batch.

    Returns
    -------
    float
        Scalar Fisher trace estimate for *component*.
    """
    keys = _keys_for_component(state_dict, component)
    total = 0.0
    for grad_dict in gradients:
        for key in keys:
            g = np.asarray(grad_dict[key], dtype=np.float64)
            total += float(np.sum(g ** 2))
    return total / max(len(gradients), 1)


def merge_branch_fisher(
    state_dicts: List[StateDict],
    fisher_traces: Dict[str, Dict[str, float]],
) -> Tuple[StateDict, np.ndarray, np.ndarray]:
    """Condition 4: Fisher-weighted branch soup.

    Coefficients are proportional to the empirical Fisher information trace
    for each branch, normalised to sum to 1 across models:

        λ_cls_i ∝ F_cls_i,   λ_reg_i ∝ F_reg_i

    following the Fisher merging approach of Matena and Raffel (2022).

    Parameters
    ----------
    state_dicts:
        List of N ingredient model state dicts (ordered to match
        *fisher_traces* keys).
    fisher_traces:
        Nested dict ``{run_id: {"cls": float, "reg": float}}``.
        The run-ID ordering must match the ordering of *state_dicts*.

    Returns
    -------
    StateDict
        Fully merged state dict.
    np.ndarray
        Normalised λ_cls vector (length N).
    np.ndarray
        Normalised λ_reg vector (length N).

    Raises
    ------
    ValueError
        If any Fisher trace is negative, or if normalisation would divide
        by zero (all traces are zero for a branch).
    """
    if not state_dicts:
        raise ValueError("state_dicts must not be empty.")
    _check_key_coverage(state_dicts)

    run_ids = list(fisher_traces.keys())
    if len(run_ids) != len(state_dicts):
        raise ValueError(
            f"fisher_traces has {len(run_ids)} entries but state_dicts has "
            f"{len(state_dicts)} entries."
        )

    traces_cls = np.array([fisher_traces[r]["cls"] for r in run_ids], dtype=np.float64)
    traces_reg = np.array([fisher_traces[r]["reg"] for r in run_ids], dtype=np.float64)

    if np.any(traces_cls < 0) or np.any(traces_reg < 0):
        raise ValueError("Fisher traces must be non-negative.")

    sum_cls = traces_cls.sum()
    sum_reg = traces_reg.sum()
    if sum_cls == 0.0:
        raise ValueError("All cls Fisher traces are zero; cannot normalise.")
    if sum_reg == 0.0:
        raise ValueError("All reg Fisher traces are zero; cannot normalise.")

    lambda_cls = traces_cls / sum_cls
    lambda_reg = traces_reg / sum_reg

    stage1 = merge_backbone_encoder(state_dicts)
    stage2 = merge_head_weighted(state_dicts, lambda_cls, lambda_reg)
    soup = assemble_soup(stage1, stage2)

    logger.info(
        "Condition 4 (Fisher-weighted) assembled. "
        "λ_cls=%s, λ_reg=%s",
        np.array2string(lambda_cls, precision=4),
        np.array2string(lambda_reg, precision=4),
    )
    return soup, lambda_cls, lambda_reg
