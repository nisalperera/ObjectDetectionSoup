"""
Tests for soup.loss_landscape – loss barrier and Hessian trace measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from soup.loss_landscape import (
    ALPHA_GRID,
    COMPONENTS,
    interpolate_component,
    measure_all_barriers,
    measure_loss_barrier,
    save_barriers_npy,
)

# ---------------------------------------------------------------------------
# Synthetic state dicts (same key structure as test_merger.py)
# ---------------------------------------------------------------------------

_BACKBONE_KEY = "model.backbone.res2.weight"
_ENCODER_KEY = "model.proposal_generator.encoder.lateral_conv.weight"
_CLS_SUBNET_KEY = "model.proposal_generator.head.cls_subnet.0.weight"
_CLS_SCORE_KEY = "model.proposal_generator.head.cls_score.weight"
_BBOX_SUBNET_KEY = "model.proposal_generator.head.bbox_subnet.0.weight"
_BBOX_PRED_KEY = "model.proposal_generator.head.bbox_pred.weight"
_OBJ_PRED_KEY = "model.proposal_generator.head.object_pred.weight"

ALL_KEYS = [
    _BACKBONE_KEY, _ENCODER_KEY,
    _CLS_SUBNET_KEY, _CLS_SCORE_KEY,
    _BBOX_SUBNET_KEY, _BBOX_PRED_KEY, _OBJ_PRED_KEY,
]


def _make_state_dict(fill: float):
    return {k: np.full((4,), fill) for k in ALL_KEYS}


# ---------------------------------------------------------------------------
# interpolate_component
# ---------------------------------------------------------------------------


class TestInterpolateComponent:
    def test_alpha_zero_returns_a(self):
        a = _make_state_dict(0.0)
        b = _make_state_dict(4.0)
        mixed = interpolate_component(a, b, "backbone", 0.0)
        assert np.allclose(mixed[_BACKBONE_KEY], 0.0)

    def test_alpha_one_returns_b(self):
        a = _make_state_dict(0.0)
        b = _make_state_dict(4.0)
        mixed = interpolate_component(a, b, "backbone", 1.0)
        assert np.allclose(mixed[_BACKBONE_KEY], 4.0)

    def test_alpha_half(self):
        a = _make_state_dict(0.0)
        b = _make_state_dict(4.0)
        mixed = interpolate_component(a, b, "backbone", 0.5)
        assert np.allclose(mixed[_BACKBONE_KEY], 2.0)

    def test_non_target_components_frozen_at_a(self):
        a = _make_state_dict(1.0)
        b = _make_state_dict(9.0)
        mixed = interpolate_component(a, b, "backbone", 0.5)
        # encoder not interpolated → stays at a's value
        assert np.allclose(mixed[_ENCODER_KEY], 1.0)
        assert np.allclose(mixed[_CLS_SCORE_KEY], 1.0)


# ---------------------------------------------------------------------------
# measure_loss_barrier
# ---------------------------------------------------------------------------


class TestMeasureLossBarrier:
    def _flat_loss_fn(self, state):
        """Constant loss function – no barrier."""
        return 1.0

    def _convex_loss_fn(self, state):
        """Loss = mean of backbone values (linear → zero barrier for linear params)."""
        return float(state[_BACKBONE_KEY].mean())

    def _peaked_loss_fn(self, target_alpha: float = 0.5):
        """Loss peaks at alpha=target_alpha, giving a positive barrier."""
        def fn(state):
            val = float(state[_BACKBONE_KEY].mean())
            # val goes from 0 (α=0) to 4 (α=1) linearly
            # We want a peak at target_alpha: loss = -(val - 2)^2 + 5
            return -(val - 2.0) ** 2 + 5.0
        return fn

    def test_returns_dict_with_required_keys(self):
        a = _make_state_dict(0.0)
        b = _make_state_dict(4.0)
        result = measure_loss_barrier(a, b, "backbone", self._flat_loss_fn)
        assert "barrier" in result
        assert "losses" in result
        assert "alpha_grid" in result
        assert "loss_a" in result
        assert "loss_b" in result
        assert "component" in result

    def test_flat_loss_zero_barrier(self):
        a = _make_state_dict(0.0)
        b = _make_state_dict(4.0)
        result = measure_loss_barrier(a, b, "backbone", self._flat_loss_fn)
        assert result["barrier"] == pytest.approx(0.0, abs=1e-6)

    def test_losses_length_matches_alpha_grid(self):
        a = _make_state_dict(0.0)
        b = _make_state_dict(1.0)
        result = measure_loss_barrier(a, b, "backbone", self._flat_loss_fn)
        assert len(result["losses"]) == len(result["alpha_grid"])

    def test_default_alpha_grid_length(self):
        a = _make_state_dict(0.0)
        b = _make_state_dict(1.0)
        result = measure_loss_barrier(a, b, "backbone", self._flat_loss_fn)
        assert len(result["alpha_grid"]) == 21

    def test_peaked_loss_positive_barrier(self):
        a = _make_state_dict(0.0)
        b = _make_state_dict(4.0)
        result = measure_loss_barrier(
            a, b, "backbone", self._peaked_loss_fn(target_alpha=0.5)
        )
        assert result["barrier"] > 0.0

    def test_unknown_component_raises(self):
        a = _make_state_dict(0.0)
        b = _make_state_dict(1.0)
        with pytest.raises(KeyError):
            measure_loss_barrier(a, b, "unknown", self._flat_loss_fn)

    def test_loss_a_is_alpha_zero(self):
        a = _make_state_dict(0.0)
        b = _make_state_dict(4.0)
        result = measure_loss_barrier(a, b, "backbone", self._convex_loss_fn)
        assert result["loss_a"] == pytest.approx(0.0)

    def test_loss_b_is_alpha_one(self):
        a = _make_state_dict(0.0)
        b = _make_state_dict(4.0)
        result = measure_loss_barrier(a, b, "backbone", self._convex_loss_fn)
        assert result["loss_b"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# measure_all_barriers
# ---------------------------------------------------------------------------


class TestMeasureAllBarriers:
    def test_number_of_pairs(self):
        sds = [_make_state_dict(float(i)) for i in range(6)]
        run_ids = [f"M{i}" for i in range(6)]
        results = measure_all_barriers(sds, run_ids, lambda s: 1.0)
        # C(6, 2) = 15 pairs × 4 components = 60 entries
        assert len(results) == 15 * 4

    def test_keys_are_triples(self):
        sds = [_make_state_dict(0.0), _make_state_dict(1.0)]
        results = measure_all_barriers(sds, ["A", "B"], lambda s: 0.0)
        for key in results:
            assert isinstance(key, tuple)
            assert len(key) == 3

    def test_all_components_covered(self):
        sds = [_make_state_dict(0.0), _make_state_dict(1.0)]
        results = measure_all_barriers(sds, ["A", "B"], lambda s: 0.0)
        components_found = {key[2] for key in results}
        assert components_found == set(COMPONENTS)

    def test_mismatched_lengths_raises(self):
        sds = [_make_state_dict(0.0)]
        with pytest.raises(ValueError):
            measure_all_barriers(sds, ["A", "B"], lambda s: 0.0)


# ---------------------------------------------------------------------------
# save_barriers_npy
# ---------------------------------------------------------------------------


class TestSaveBarriersNpy:
    def test_creates_file(self, tmp_path):
        sds = [_make_state_dict(0.0), _make_state_dict(1.0)]
        barriers = measure_all_barriers(sds, ["A", "B"], lambda s: 0.0)
        out = tmp_path / "barriers.npy"
        save_barriers_npy(barriers, out)
        assert out.exists()

    def test_creates_parent_dirs(self, tmp_path):
        sds = [_make_state_dict(0.0), _make_state_dict(1.0)]
        barriers = measure_all_barriers(sds, ["A", "B"], lambda s: 0.0)
        out = tmp_path / "nested" / "barriers.npy"
        save_barriers_npy(barriers, out)
        assert out.exists()
