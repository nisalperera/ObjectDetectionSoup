"""
Tests for soup.evaluate – ingredient quality audit and helper functions.
"""

from __future__ import annotations

import pytest

from soup.evaluate import (
    QUALITY_THRESHOLD_PP,
    best_single_model,
    ingredient_quality_audit,
    pilot_divergence_check,
)


class TestIngredientQualityAudit:
    def test_all_pass_within_threshold(self):
        metrics = {
            "L1": {"AP": 42.0, "AP50": 60.0, "AR100": 55.0},
            "L2": {"AP": 41.0, "AP50": 59.0, "AR100": 54.0},
            "L3": {"AP": 40.5, "AP50": 58.5, "AR100": 53.5},
        }
        included, excluded = ingredient_quality_audit(metrics)
        assert set(included) == {"L1", "L2", "L3"}
        assert excluded == []

    def test_one_model_excluded(self):
        metrics = {
            "L1": {"AP": 42.0, "AP50": 60.0, "AR100": 55.0},
            "L2": {"AP": 38.0, "AP50": 56.0, "AR100": 51.0},  # > 3 pp below max
        }
        included, excluded = ingredient_quality_audit(metrics)
        assert "L1" in included
        assert "L2" in excluded

    def test_exactly_at_threshold_not_excluded(self):
        metrics = {
            "A": {"AP": 42.0, "AP50": 60.0, "AR100": 55.0},
            "B": {"AP": 39.0, "AP50": 57.0, "AR100": 52.0},  # exactly 3.0 pp below
        }
        included, excluded = ingredient_quality_audit(metrics)
        assert "B" in included
        assert excluded == []

    def test_custom_threshold(self):
        metrics = {
            "A": {"AP": 42.0, "AP50": 60.0, "AR100": 55.0},
            "B": {"AP": 40.0, "AP50": 58.0, "AR100": 53.0},  # 2 pp below
        }
        # With threshold=1.5, B should be excluded
        included, excluded = ingredient_quality_audit(metrics, threshold_pp=1.5)
        assert "B" in excluded

    def test_empty_metrics_returns_empty(self):
        included, excluded = ingredient_quality_audit({})
        assert included == []
        assert excluded == []

    def test_single_model_always_included(self):
        metrics = {"L1": {"AP": 40.0, "AP50": 58.0, "AR100": 53.0}}
        included, excluded = ingredient_quality_audit(metrics)
        assert included == ["L1"]
        assert excluded == []


class TestPilotDivergenceCheck:
    def test_passes_sufficient_delta(self):
        pilot = {"AP": 43.0}
        theta0 = {"AP": 42.5}
        assert pilot_divergence_check(pilot, theta0, min_ap_delta=0.3) is True

    def test_fails_insufficient_delta(self):
        pilot = {"AP": 42.6}
        theta0 = {"AP": 42.5}
        assert pilot_divergence_check(pilot, theta0, min_ap_delta=0.3) is False

    def test_passes_when_pilot_below_theta0(self):
        """Delta is absolute value; pilot can be below θ₀."""
        pilot = {"AP": 42.0}
        theta0 = {"AP": 42.5}
        assert pilot_divergence_check(pilot, theta0, min_ap_delta=0.3) is True

    def test_exactly_at_threshold(self):
        # Use values that are exactly representable in floating point
        pilot = {"AP": 43.0}
        theta0 = {"AP": 42.5}
        assert pilot_divergence_check(pilot, theta0, min_ap_delta=0.5) is True


class TestBestSingleModel:
    def test_returns_highest_ap(self):
        metrics = {
            "L1": {"AP": 41.0},
            "L2": {"AP": 43.5},
            "L3": {"AP": 42.0},
        }
        best_id, best_ap = best_single_model(metrics)
        assert best_id == "L2"
        assert best_ap == pytest.approx(43.5)

    def test_single_model(self):
        metrics = {"L1": {"AP": 40.0}}
        best_id, best_ap = best_single_model(metrics)
        assert best_id == "L1"
        assert best_ap == pytest.approx(40.0)
