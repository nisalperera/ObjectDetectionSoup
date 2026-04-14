"""
Integration-level tests for soup.pipeline – Step 7 merging and Step 8
source pre-registration (no Detectron2/PyTorch needed).
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from soup.merger import StateDict
from soup.pipeline import step5_ingredient_audit, step7_merging, step8_preregister_sources

# ---------------------------------------------------------------------------
# Synthetic state dicts
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


def _make_sd(fill: float) -> StateDict:
    return {k: np.full((4,), fill) for k in ALL_KEYS}


def _make_state_dicts():
    return {
        f"M{i}": _make_sd(float(i)) for i in range(3)
    }


# ---------------------------------------------------------------------------
# step5_ingredient_audit (pipeline wrapper)
# ---------------------------------------------------------------------------


class TestStep5PipelineAudit:
    def test_writes_json(self, tmp_path):
        metrics = {
            "L1": {"AP": 42.0, "AP50": 60.0, "AR100": 55.0},
            "L2": {"AP": 41.0, "AP50": 59.0, "AR100": 54.0},
        }
        (tmp_path / "logs").mkdir()
        included, excluded = step5_ingredient_audit(metrics, tmp_path)
        table_path = tmp_path / "logs" / "table_4_1_ingredient_audit.json"
        assert table_path.exists()
        data = json.loads(table_path.read_text())
        assert "included" in data
        assert "excluded" in data


# ---------------------------------------------------------------------------
# step7_merging (pipeline wrapper)
# ---------------------------------------------------------------------------


class TestStep7Merging:
    def _make_eval_fn(self):
        call_count = {"n": 0}

        def fn(soup: StateDict) -> float:
            call_count["n"] += 1
            return float(soup[_CLS_SCORE_KEY].mean())

        return fn, call_count

    def test_conditions_1_and_2_present(self, tmp_path):
        sds = _make_state_dicts()
        eval_fn, _ = self._make_eval_fn()
        results = step7_merging(sds, tmp_path, eval_fn, None, dirichlet_seed=0)
        assert "condition_1" in results
        assert "condition_2" in results

    def test_condition_3_present(self, tmp_path):
        sds = _make_state_dicts()
        eval_fn, _ = self._make_eval_fn()
        results = step7_merging(sds, tmp_path, eval_fn, None, dirichlet_seed=0)
        assert "condition_3" in results

    def test_condition_4_present_with_fisher_traces(self, tmp_path):
        sds = _make_state_dicts()
        eval_fn, _ = self._make_eval_fn()
        fisher = {
            "M0": {"cls": 1.0, "reg": 2.0},
            "M1": {"cls": 2.0, "reg": 1.0},
            "M2": {"cls": 1.5, "reg": 1.5},
        }
        results = step7_merging(sds, tmp_path, eval_fn, None, fisher_traces=fisher, dirichlet_seed=0)
        assert "condition_4" in results

    def test_condition_4_absent_without_fisher(self, tmp_path):
        sds = _make_state_dicts()
        eval_fn, _ = self._make_eval_fn()
        results = step7_merging(sds, tmp_path, eval_fn, None, dirichlet_seed=0)
        assert "condition_4" not in results

    def test_lambda_vectors_written(self, tmp_path):
        sds = _make_state_dicts()
        eval_fn, _ = self._make_eval_fn()
        step7_merging(sds, tmp_path, eval_fn, None, dirichlet_seed=0)
        assert (tmp_path / "lambda_vectors" / "condition_1.json").exists()
        assert (tmp_path / "lambda_vectors" / "condition_2.json").exists()
        assert (tmp_path / "lambda_vectors" / "condition_3.json").exists()

    def test_ap_values_in_results(self, tmp_path):
        sds = _make_state_dicts()
        eval_fn, _ = self._make_eval_fn()
        results = step7_merging(sds, tmp_path, eval_fn, None, dirichlet_seed=0)
        for cond in ("condition_1", "condition_2", "condition_3"):
            assert isinstance(results[cond]["ap"], float)


# ---------------------------------------------------------------------------
# step8_preregister_sources
# ---------------------------------------------------------------------------


class TestStep8Preregister:
    def _make_results(self, with_condition_4=False):
        dummy_soup = _make_sd(1.0)
        results = {
            "condition_2": {"soup": dummy_soup, "ap": 41.0},
            "condition_3": {"soup": dummy_soup, "ap": 42.5},
        }
        if with_condition_4:
            results["condition_4"] = {"soup": dummy_soup, "ap": 44.0}
        return results

    def test_d1_source_is_condition_2(self, tmp_path):
        results = self._make_results()
        mapping = step8_preregister_sources(results, tmp_path)
        assert mapping["D1"] == "condition_2"

    def test_d2_source_is_best_of_3_4(self, tmp_path):
        results = self._make_results(with_condition_4=True)
        mapping = step8_preregister_sources(results, tmp_path)
        assert mapping["D2"] == "condition_4"  # condition_4 AP=44 > condition_3 AP=42.5

    def test_d2_source_condition_3_when_no_4(self, tmp_path):
        results = self._make_results(with_condition_4=False)
        mapping = step8_preregister_sources(results, tmp_path)
        assert mapping["D2"] == "condition_3"

    def test_writes_registration_json(self, tmp_path):
        results = self._make_results()
        step8_preregister_sources(results, tmp_path)
        reg_path = tmp_path / "logs" / "step8_source_preregistration.json"
        assert reg_path.exists()
        data = json.loads(reg_path.read_text())
        assert "D1_source" in data
        assert "D2_source" in data

    def test_no_learned_conditions_raises(self, tmp_path):
        # Only condition_1 and condition_2 present – no condition_3 or _4
        results = {
            "condition_1": {"soup": _make_sd(1.0), "ap": 40.0},
            "condition_2": {"soup": _make_sd(1.0), "ap": 41.0},
        }
        with pytest.raises(ValueError, match="Condition 3 or Condition 4"):
            step8_preregister_sources(results, tmp_path)
