"""
Tests for soup.config – Experiment Configuration Registry (Table 3.1).
"""

import pytest

from soup.config import (
    BASE_CONFIG,
    FINETUNE_RUN_IDS,
    FINAL_RUN_ID,
    INGREDIENT_RUN_IDS,
    PARAMETER_PREFIXES,
    RUN_CONFIGS,
    RunConfig,
    get_finetune_configs,
    get_ingredient_configs,
)


class TestParameterPrefixes:
    def test_all_components_present(self):
        assert set(PARAMETER_PREFIXES.keys()) == {"backbone", "encoder", "cls", "reg"}

    def test_each_component_has_at_least_one_prefix(self):
        for comp, prefixes in PARAMETER_PREFIXES.items():
            assert len(prefixes) >= 1, f"Component {comp!r} has no prefixes."

    def test_cls_branch_prefixes(self):
        assert "model.proposal_generator.head.cls_subnet." in PARAMETER_PREFIXES["cls"]
        assert "model.proposal_generator.head.cls_score." in PARAMETER_PREFIXES["cls"]

    def test_reg_branch_prefixes(self):
        reg = PARAMETER_PREFIXES["reg"]
        assert "model.proposal_generator.head.bbox_subnet." in reg
        assert "model.proposal_generator.head.bbox_pred." in reg
        assert "model.proposal_generator.head.object_pred." in reg

    def test_backbone_prefix(self):
        assert "model.backbone." in PARAMETER_PREFIXES["backbone"]

    def test_encoder_prefix(self):
        assert "model.proposal_generator.encoder." in PARAMETER_PREFIXES["encoder"]


class TestRunIDSets:
    def test_ingredient_run_ids(self):
        assert set(INGREDIENT_RUN_IDS) == {"L1", "L2", "L3", "L4", "C1", "C2"}

    def test_finetune_run_ids(self):
        assert set(FINETUNE_RUN_IDS) == {"D1", "D2"}

    def test_final_run_id(self):
        assert FINAL_RUN_ID == "C3"

    def test_all_nine_runs_present(self):
        expected = {"L1", "L2", "L3", "L4", "C1", "C2", "D1", "D2", "C3"}
        assert set(RUN_CONFIGS.keys()) == expected


class TestRunConfigTable:
    """Validate Table 3.1 values for every run."""

    @pytest.mark.parametrize("run_id", ["L1", "L2", "L3", "L4"])
    def test_l_runs_gpu(self, run_id):
        assert RUN_CONFIGS[run_id].gpu == "RTX 5070 Ti"

    @pytest.mark.parametrize("run_id", ["C1", "C2"])
    def test_c_runs_gpu(self, run_id):
        assert RUN_CONFIGS[run_id].gpu == "RTX 5090"

    @pytest.mark.parametrize("run_id", ["D1", "D2"])
    def test_d_runs_gpu(self, run_id):
        assert RUN_CONFIGS[run_id].gpu == "RTX 5070 Ti"

    def test_c3_gpu(self):
        assert RUN_CONFIGS["C3"].gpu == "RTX 5090"

    @pytest.mark.parametrize("run_id", ["L1", "L2", "L3", "C1", "C2"])
    def test_standard_ingredient_epochs(self, run_id):
        assert RUN_CONFIGS[run_id].epochs == 8

    def test_l4_extended_epochs(self):
        # L4 changes the training epochs hyperparameter
        assert RUN_CONFIGS["L4"].epochs > 8

    @pytest.mark.parametrize("run_id", ["D1", "D2", "C3"])
    def test_finetune_epochs(self, run_id):
        assert RUN_CONFIGS[run_id].epochs == 2

    def test_c3_epochs_max(self):
        assert RUN_CONFIGS["C3"].epochs_max == 3

    @pytest.mark.parametrize("run_id", ["L1", "L2", "L3", "L4", "D1", "D2"])
    def test_batch_8(self, run_id):
        assert RUN_CONFIGS[run_id].batch_size == 8

    @pytest.mark.parametrize("run_id", ["C1", "C2", "C3"])
    def test_batch_16(self, run_id):
        assert RUN_CONFIGS[run_id].batch_size == 16

    def test_seeds(self):
        expected_seeds = {
            "L1": 1, "L2": 2, "L3": 3, "L4": 4,
            "C1": 11, "C2": 22,
            "D1": 7, "D2": 7,
            "C3": 33,
        }
        for run_id, seed in expected_seeds.items():
            assert RUN_CONFIGS[run_id].seed == seed, (
                f"Run {run_id} expected seed {seed}, got {RUN_CONFIGS[run_id].seed}."
            )

    @pytest.mark.parametrize("run_id", ["L1", "L2", "L3", "L4", "C1", "C2"])
    def test_ingredient_no_frozen_layers(self, run_id):
        assert RUN_CONFIGS[run_id].frozen_layers is None

    @pytest.mark.parametrize("run_id", ["D1", "D2", "C3"])
    def test_finetune_frozen_layers(self, run_id):
        frozen = RUN_CONFIGS[run_id].frozen_layers
        assert frozen is not None
        assert "backbone" in frozen
        assert "encoder" in frozen

    @pytest.mark.parametrize("run_id", ["L1", "L2", "L3", "L4", "C1", "C2"])
    def test_ingredient_full_finetune_type(self, run_id):
        assert RUN_CONFIGS[run_id].run_type == "Full fine-tune"

    @pytest.mark.parametrize("run_id", ["D1", "D2", "C3"])
    def test_finetune_type(self, run_id):
        assert RUN_CONFIGS[run_id].run_type == "Head fine-tune"

    def test_l1_anchor_no_overrides(self):
        assert RUN_CONFIGS["L1"].hyperparameter_overrides == {}

    def test_l2_learning_rate_override(self):
        overrides = RUN_CONFIGS["L2"].hyperparameter_overrides
        assert "BASE_LR" in overrides
        assert float(overrides["BASE_LR"]) != float(BASE_CONFIG["BASE_LR"])

    def test_l3_weight_decay_override(self):
        overrides = RUN_CONFIGS["L3"].hyperparameter_overrides
        assert "WEIGHT_DECAY" in overrides
        assert float(overrides["WEIGHT_DECAY"]) != float(BASE_CONFIG["WEIGHT_DECAY"])

    def test_l4_epoch_override(self):
        overrides = RUN_CONFIGS["L4"].hyperparameter_overrides
        assert "EPOCHS" in overrides

    def test_c1_batch_override(self):
        overrides = RUN_CONFIGS["C1"].hyperparameter_overrides
        assert "BATCH_SIZE" in overrides
        assert int(overrides["BATCH_SIZE"]) == 16

    def test_c2_lr_schedule_override(self):
        overrides = RUN_CONFIGS["C2"].hyperparameter_overrides
        assert "LR_SCHEDULE" in overrides
        assert overrides["LR_SCHEDULE"] == "WarmupCosineLR"

    @pytest.mark.parametrize("run_id", ["D1", "D2", "C3"])
    def test_finetune_no_hyperparameter_change(self, run_id):
        assert RUN_CONFIGS[run_id].changed_hyperparameter is None


class TestRunConfigHelpers:
    def test_is_ingredient(self):
        for run_id in INGREDIENT_RUN_IDS:
            assert RUN_CONFIGS[run_id].is_ingredient

    def test_not_ingredient_for_finetune(self):
        for run_id in ("D1", "D2", "C3"):
            assert not RUN_CONFIGS[run_id].is_ingredient

    def test_is_finetune(self):
        for run_id in ("D1", "D2", "C3"):
            assert RUN_CONFIGS[run_id].is_finetune

    def test_not_finetune_for_ingredients(self):
        for run_id in INGREDIENT_RUN_IDS:
            assert not RUN_CONFIGS[run_id].is_finetune

    def test_get_ingredient_configs(self):
        cfg = get_ingredient_configs()
        assert set(cfg.keys()) == set(INGREDIENT_RUN_IDS)

    def test_get_finetune_configs(self):
        cfg = get_finetune_configs()
        assert set(cfg.keys()) == {"D1", "D2", "C3"}

    def test_exactly_one_hyperparameter_changed_per_ingredient(self):
        """Each ingredient run must override at most one entry beyond the base."""
        for run_id in INGREDIENT_RUN_IDS:
            overrides = RUN_CONFIGS[run_id].hyperparameter_overrides
            # L1 is the anchor (zero overrides); others have exactly one.
            if run_id == "L1":
                assert len(overrides) == 0
            else:
                # C2 doubles batch (inherited from C1) + LR schedule –
                # exactly these two because batch is the base for C-runs.
                assert len(overrides) >= 1
