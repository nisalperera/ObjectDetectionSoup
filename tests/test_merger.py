"""
Tests for soup.merger – Core Merging Module (Sections 3.3.1–3.3.2).

Uses synthetic state dicts with YOLOF-compatible key names; no Detectron2 or
PyTorch required.
"""

from __future__ import annotations

import numpy as np
import pytest

from soup.merger import (
    StateDict,
    _check_key_coverage,
    _keys_for_component,
    _sample_dirichlet_candidates,
    _uniform_lambdas,
    _validate_simplex,
    _weighted_average,
    assemble_soup,
    merge_backbone_encoder,
    merge_branch_dirichlet,
    merge_branch_fisher,
    merge_branch_uniform,
    merge_head_weighted,
    merge_uniform,
)

# ---------------------------------------------------------------------------
# Fixtures – synthetic YOLOF-like state dicts
# ---------------------------------------------------------------------------

# Representative parameter keys for each component
_BACKBONE_KEY = "model.backbone.res2.weight"
_ENCODER_KEY = "model.proposal_generator.encoder.lateral_conv.weight"
_CLS_SUBNET_KEY = "model.proposal_generator.head.cls_subnet.0.weight"
_CLS_SCORE_KEY = "model.proposal_generator.head.cls_score.weight"
_BBOX_SUBNET_KEY = "model.proposal_generator.head.bbox_subnet.0.weight"
_BBOX_PRED_KEY = "model.proposal_generator.head.bbox_pred.weight"
_OBJ_PRED_KEY = "model.proposal_generator.head.object_pred.weight"

ALL_KEYS = [
    _BACKBONE_KEY,
    _ENCODER_KEY,
    _CLS_SUBNET_KEY,
    _CLS_SCORE_KEY,
    _BBOX_SUBNET_KEY,
    _BBOX_PRED_KEY,
    _OBJ_PRED_KEY,
]


def _make_state_dict(fill_value: float) -> StateDict:
    """Create a synthetic state dict with all weights set to *fill_value*."""
    return {k: np.full((4, 4), fill_value) for k in ALL_KEYS}


def _make_state_dicts(values: list[float]) -> list[StateDict]:
    return [_make_state_dict(v) for v in values]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestUniformLambdas:
    def test_sum_to_one(self):
        lam = _uniform_lambdas(6)
        assert abs(lam.sum() - 1.0) < 1e-10

    def test_equal_weights(self):
        lam = _uniform_lambdas(4)
        assert np.allclose(lam, 0.25)

    def test_single_model(self):
        lam = _uniform_lambdas(1)
        assert lam[0] == pytest.approx(1.0)


class TestValidateSimplex:
    def test_valid_uniform(self):
        _validate_simplex(np.array([0.25, 0.25, 0.25, 0.25]))

    def test_valid_arbitrary(self):
        _validate_simplex(np.array([0.5, 0.3, 0.2]))

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="negative"):
            _validate_simplex(np.array([-0.1, 0.6, 0.5]))

    def test_rejects_wrong_sum(self):
        with pytest.raises(ValueError, match="sum"):
            _validate_simplex(np.array([0.5, 0.5, 0.5]))


class TestKeysForComponent:
    def test_backbone_keys(self):
        sd = _make_state_dict(0.0)
        keys = _keys_for_component(sd, "backbone")
        assert _BACKBONE_KEY in keys
        for k in keys:
            assert k.startswith("model.backbone.")

    def test_cls_keys(self):
        sd = _make_state_dict(0.0)
        keys = _keys_for_component(sd, "cls")
        assert _CLS_SUBNET_KEY in keys
        assert _CLS_SCORE_KEY in keys

    def test_reg_keys(self):
        sd = _make_state_dict(0.0)
        keys = _keys_for_component(sd, "reg")
        assert _BBOX_SUBNET_KEY in keys
        assert _BBOX_PRED_KEY in keys
        assert _OBJ_PRED_KEY in keys

    def test_unknown_component_raises(self):
        sd = _make_state_dict(0.0)
        with pytest.raises(KeyError):
            _keys_for_component(sd, "unknown_component")


class TestCheckKeyCoverage:
    def test_same_keys_no_error(self):
        sds = [_make_state_dict(i) for i in range(3)]
        _check_key_coverage(sds)

    def test_mismatched_keys_raises(self):
        sd1 = _make_state_dict(0.0)
        sd2 = {"model.backbone.extra": np.ones((2, 2))}
        with pytest.raises(ValueError, match="mismatched keys"):
            _check_key_coverage([sd1, sd2])

    def test_single_dict_no_error(self):
        _check_key_coverage([_make_state_dict(1.0)])


class TestWeightedAverage:
    def test_uniform_average_values(self):
        sds = _make_state_dicts([1.0, 3.0])
        keys = [_BACKBONE_KEY]
        lam = np.array([0.5, 0.5])
        result = _weighted_average(sds, keys, lam)
        assert np.allclose(result[_BACKBONE_KEY], 2.0)

    def test_all_weight_to_first_model(self):
        sds = _make_state_dicts([7.0, 99.0])
        keys = [_BACKBONE_KEY]
        lam = np.array([1.0, 0.0])
        result = _weighted_average(sds, keys, lam)
        assert np.allclose(result[_BACKBONE_KEY], 7.0)

    def test_three_models(self):
        sds = _make_state_dicts([0.0, 3.0, 6.0])
        keys = [_BACKBONE_KEY]
        lam = _uniform_lambdas(3)
        result = _weighted_average(sds, keys, lam)
        assert np.allclose(result[_BACKBONE_KEY], 3.0)


class TestDirichletCandidates:
    def test_shape(self):
        candidates = _sample_dirichlet_candidates(n=6, m=100)
        assert candidates.shape == (100, 6)

    def test_each_row_sums_to_one(self):
        candidates = _sample_dirichlet_candidates(n=4, m=50, rng=np.random.default_rng(42))
        assert np.allclose(candidates.sum(axis=1), 1.0)

    def test_all_non_negative(self):
        candidates = _sample_dirichlet_candidates(n=3, m=50, rng=np.random.default_rng(0))
        assert np.all(candidates >= 0)


# ---------------------------------------------------------------------------
# Merging conditions
# ---------------------------------------------------------------------------


class TestMergeBackboneEncoder:
    def test_uniform_average(self):
        sds = _make_state_dicts([2.0, 4.0])
        result = merge_backbone_encoder(sds)
        assert np.allclose(result[_BACKBONE_KEY], 3.0)
        assert np.allclose(result[_ENCODER_KEY], 3.0)

    def test_only_backbone_encoder_keys(self):
        sds = _make_state_dicts([1.0, 1.0])
        result = merge_backbone_encoder(sds)
        for key in result:
            assert key.startswith("model.backbone.") or key.startswith(
                "model.proposal_generator.encoder."
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            merge_backbone_encoder([])

    def test_single_model_identity(self):
        sd = _make_state_dict(5.0)
        result = merge_backbone_encoder([sd])
        assert np.allclose(result[_BACKBONE_KEY], 5.0)


class TestMergeHeadWeighted:
    def test_uniform_average(self):
        sds = _make_state_dicts([0.0, 4.0])
        lam = np.array([0.5, 0.5])
        result = merge_head_weighted(sds, lam, lam)
        for key in (_CLS_SUBNET_KEY, _CLS_SCORE_KEY):
            assert np.allclose(result[key], 2.0)
        for key in (_BBOX_SUBNET_KEY, _BBOX_PRED_KEY, _OBJ_PRED_KEY):
            assert np.allclose(result[key], 2.0)

    def test_asymmetric_lambdas(self):
        # cls takes 100% from model 0 (val=10), reg takes 100% from model 1 (val=20)
        sds = _make_state_dicts([10.0, 20.0])
        lam_cls = np.array([1.0, 0.0])
        lam_reg = np.array([0.0, 1.0])
        result = merge_head_weighted(sds, lam_cls, lam_reg)
        assert np.allclose(result[_CLS_SCORE_KEY], 10.0)
        assert np.allclose(result[_BBOX_PRED_KEY], 20.0)

    def test_invalid_simplex_raises(self):
        sds = _make_state_dicts([1.0, 1.0])
        with pytest.raises(ValueError):
            merge_head_weighted(sds, np.array([0.6, 0.6]), np.array([0.5, 0.5]))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            merge_head_weighted([], np.array([]), np.array([]))


class TestAssembleSoup:
    def test_combines_both_stages(self):
        sds = _make_state_dicts([1.0, 1.0])
        stage1 = merge_backbone_encoder(sds)
        lam = _uniform_lambdas(2)
        stage2 = merge_head_weighted(sds, lam, lam)
        soup = assemble_soup(stage1, stage2)
        assert set(soup.keys()) == set(ALL_KEYS)

    def test_overlapping_keys_raises(self):
        sd = _make_state_dict(1.0)
        with pytest.raises(ValueError, match="share"):
            assemble_soup(sd, sd)


class TestCondition1GlobalUniform:
    def test_all_keys_present(self):
        sds = _make_state_dicts([1.0, 3.0])
        soup = merge_uniform(sds)
        assert set(soup.keys()) == set(ALL_KEYS)

    def test_average_values(self):
        sds = _make_state_dicts([0.0, 2.0])
        soup = merge_uniform(sds)
        for key in ALL_KEYS:
            assert np.allclose(soup[key], 1.0)

    def test_subsumes_branch_uniform(self):
        """Condition 1 == Condition 2 when no branch partition effect exists."""
        sds = _make_state_dicts([0.0, 2.0])
        c1 = merge_uniform(sds)
        c2 = merge_branch_uniform(sds)
        for key in ALL_KEYS:
            assert np.allclose(c1[key], c2[key]), key

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            merge_uniform([])

    def test_six_models(self):
        sds = _make_state_dicts([float(i) for i in range(6)])
        soup = merge_uniform(sds)
        expected = np.mean(np.arange(6, dtype=float))
        for key in ALL_KEYS:
            assert np.allclose(soup[key], expected)


class TestCondition2BranchUniform:
    def test_all_keys_present(self):
        sds = _make_state_dicts([1.0, 3.0])
        soup = merge_branch_uniform(sds)
        assert set(soup.keys()) == set(ALL_KEYS)

    def test_average_values(self):
        sds = _make_state_dicts([0.0, 6.0])
        soup = merge_branch_uniform(sds)
        for key in ALL_KEYS:
            assert np.allclose(soup[key], 3.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            merge_branch_uniform([])


class TestCondition3Dirichlet:
    def _make_eval_fn(self, best_lam_cls_idx: int = 0, n: int = 3):
        """Return an eval_fn that rewards model 0 for cls and model 1 for reg."""
        def eval_fn(soup: StateDict) -> float:
            cls_val = float(soup[_CLS_SCORE_KEY].mean())
            reg_val = float(soup[_BBOX_PRED_KEY].mean())
            return cls_val + reg_val
        return eval_fn

    def test_returns_three_values(self):
        sds = _make_state_dicts([1.0, 2.0, 3.0])
        eval_fn = lambda s: float(s[_CLS_SCORE_KEY].mean())
        soup, lam_cls, lam_reg = merge_branch_dirichlet(sds, eval_fn, seed=42)
        assert isinstance(soup, dict)
        assert len(lam_cls) == 3
        assert len(lam_reg) == 3

    def test_lambda_cls_simplex(self):
        sds = _make_state_dicts([1.0, 2.0, 3.0])
        eval_fn = lambda s: 0.0
        _, lam_cls, lam_reg = merge_branch_dirichlet(sds, eval_fn, seed=0)
        assert abs(lam_cls.sum() - 1.0) < 1e-6
        assert np.all(lam_cls >= 0)

    def test_lambda_reg_simplex(self):
        sds = _make_state_dicts([1.0, 2.0, 3.0])
        eval_fn = lambda s: 0.0
        _, lam_cls, lam_reg = merge_branch_dirichlet(sds, eval_fn, seed=0)
        assert abs(lam_reg.sum() - 1.0) < 1e-6
        assert np.all(lam_reg >= 0)

    def test_all_keys_present(self):
        sds = _make_state_dicts([0.0, 1.0])
        eval_fn = lambda s: 0.0
        soup, _, _ = merge_branch_dirichlet(sds, eval_fn, seed=7)
        assert set(soup.keys()) == set(ALL_KEYS)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            merge_branch_dirichlet([], lambda s: 0.0)

    def test_mini_val_eval_fn_used_in_stage_a(self):
        """If mini_val_eval_fn differs from eval_fn, ensure it is actually called."""
        sds = _make_state_dicts([1.0, 2.0])
        calls = {"mini": 0, "full": 0}

        def mini_fn(s: StateDict) -> float:
            calls["mini"] += 1
            return 0.0

        def full_fn(s: StateDict) -> float:
            calls["full"] += 1
            return 0.0

        merge_branch_dirichlet(sds, full_fn, mini_val_eval_fn=mini_fn, seed=99)
        assert calls["mini"] > 0
        assert calls["full"] > 0


class TestCondition4Fisher:
    def _make_traces(self, run_ids, cls_vals, reg_vals):
        return {r: {"cls": c, "reg": rg} for r, c, rg in zip(run_ids, cls_vals, reg_vals)}

    def test_returns_three_values(self):
        sds = _make_state_dicts([1.0, 2.0, 3.0])
        traces = self._make_traces(["L1", "L2", "L3"], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        soup, lam_cls, lam_reg = merge_branch_fisher(sds, traces)
        assert set(soup.keys()) == set(ALL_KEYS)
        assert len(lam_cls) == 3
        assert len(lam_reg) == 3

    def test_normalised_simplex(self):
        sds = _make_state_dicts([1.0, 3.0])
        traces = self._make_traces(["A", "B"], [1.0, 3.0], [4.0, 1.0])
        _, lam_cls, lam_reg = merge_branch_fisher(sds, traces)
        assert abs(lam_cls.sum() - 1.0) < 1e-10
        assert abs(lam_reg.sum() - 1.0) < 1e-10
        assert np.all(lam_cls >= 0)
        assert np.all(lam_reg >= 0)

    def test_proportional_weighting(self):
        # Model with trace 3 should get weight 0.75; model with trace 1 gets 0.25
        sds = _make_state_dicts([0.0, 4.0])
        traces = self._make_traces(["A", "B"], [1.0, 3.0], [1.0, 3.0])
        _, lam_cls, lam_reg = merge_branch_fisher(sds, traces)
        assert lam_cls[0] == pytest.approx(0.25)
        assert lam_cls[1] == pytest.approx(0.75)
        # Verify merged value: 0.25*0 + 0.75*4 = 3.0
        soup, _, _ = merge_branch_fisher(sds, traces)
        assert np.allclose(soup[_CLS_SCORE_KEY], 3.0)

    def test_negative_trace_raises(self):
        sds = _make_state_dicts([1.0, 1.0])
        traces = self._make_traces(["A", "B"], [-1.0, 2.0], [1.0, 1.0])
        with pytest.raises(ValueError, match="non-negative"):
            merge_branch_fisher(sds, traces)

    def test_all_zero_traces_raises(self):
        sds = _make_state_dicts([1.0, 1.0])
        traces = self._make_traces(["A", "B"], [0.0, 0.0], [1.0, 1.0])
        with pytest.raises(ValueError, match="zero"):
            merge_branch_fisher(sds, traces)

    def test_mismatched_length_raises(self):
        sds = _make_state_dicts([1.0, 2.0, 3.0])
        traces = self._make_traces(["A", "B"], [1.0, 1.0], [1.0, 1.0])
        with pytest.raises(ValueError, match="entries"):
            merge_branch_fisher(sds, traces)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            merge_branch_fisher([], {})


class TestSimplexSubsumption:
    """Condition 1 must be a special case of the branch formulation with λ=1/N."""

    def test_condition1_subsumes_condition2(self):
        """Global uniform soup and branch uniform soup are equivalent on uniform data."""
        sds = _make_state_dicts([1.0, 2.0, 3.0, 4.0])
        c1 = merge_uniform(sds)
        c2 = merge_branch_uniform(sds)
        for key in ALL_KEYS:
            assert np.allclose(c1[key], c2[key]), f"Key {key} differs."

    def test_condition4_with_equal_traces_equals_condition2(self):
        """Fisher weights all-equal → same as branch uniform soup."""
        sds = _make_state_dicts([0.0, 6.0])
        run_ids = ["A", "B"]
        traces = {"A": {"cls": 1.0, "reg": 1.0}, "B": {"cls": 1.0, "reg": 1.0}}
        soup_c4, _, _ = merge_branch_fisher(sds, traces)
        soup_c2 = merge_branch_uniform(sds)
        for key in ALL_KEYS:
            assert np.allclose(soup_c4[key], soup_c2[key]), f"Key {key} differs."
