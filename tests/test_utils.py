"""
Tests for soup.utils – SHA-256, seed setup, JSON lambda logging.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from soup.utils import (
    compute_sha256,
    log_checkpoint_hash,
    log_lambda_vectors,
    save_metrics_csv,
    setup_seed,
)


class TestComputeSHA256:
    def test_known_hash(self, tmp_path):
        # SHA-256 of empty file
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        digest = compute_sha256(f)
        assert digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_non_empty_file(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello world")
        digest = compute_sha256(f)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_deterministic(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"test content 12345")
        assert compute_sha256(f) == compute_sha256(f)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"aaa")
        f2.write_bytes(b"bbb")
        assert compute_sha256(f1) != compute_sha256(f2)


class TestLogCheckpointHash:
    def test_returns_digest(self, tmp_path):
        f = tmp_path / "model.pth"
        f.write_bytes(b"fake checkpoint data")
        digest = log_checkpoint_hash(f)
        assert len(digest) == 64

    def test_writes_log_file(self, tmp_path):
        f = tmp_path / "model.pth"
        f.write_bytes(b"data")
        log_path = tmp_path / "hashes.json"
        log_checkpoint_hash(f, log_path=log_path)
        assert log_path.exists()
        data = json.loads(log_path.read_text())
        assert "model.pth" in data

    def test_appends_to_existing_log(self, tmp_path):
        f1 = tmp_path / "model1.pth"
        f2 = tmp_path / "model2.pth"
        f1.write_bytes(b"model1")
        f2.write_bytes(b"model2")
        log_path = tmp_path / "hashes.json"
        log_checkpoint_hash(f1, log_path=log_path)
        log_checkpoint_hash(f2, log_path=log_path)
        data = json.loads(log_path.read_text())
        assert "model1.pth" in data
        assert "model2.pth" in data


class TestSetupSeed:
    def test_reproducible_numpy(self):
        setup_seed(42)
        a = np.random.rand(5)
        setup_seed(42)
        b = np.random.rand(5)
        assert np.allclose(a, b)

    def test_reproducible_random_module(self):
        import random
        setup_seed(99)
        a = random.random()
        setup_seed(99)
        b = random.random()
        assert a == b

    def test_different_seeds_different_values(self):
        setup_seed(1)
        a = np.random.rand()
        setup_seed(2)
        b = np.random.rand()
        assert a != b

    def test_python_hash_seed_set(self):
        setup_seed(123)
        assert os.environ.get("PYTHONHASHSEED") == "123"


class TestLogLambdaVectors:
    def test_writes_json(self, tmp_path):
        log_lambda_vectors(
            "condition_3",
            ["L1", "L2", "L3"],
            [0.5, 0.3, 0.2],
            [0.1, 0.6, 0.3],
            tmp_path / "lambda.json",
        )
        path = tmp_path / "lambda.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["condition"] == "condition_3"
        assert data["run_ids"] == ["L1", "L2", "L3"]
        assert abs(data["sum_lambda_cls"] - 1.0) < 1e-6
        assert abs(data["sum_lambda_reg"] - 1.0) < 1e-6

    def test_creates_parent_dirs(self, tmp_path):
        log_lambda_vectors(
            "test",
            ["A"],
            [1.0],
            [1.0],
            tmp_path / "nested" / "dir" / "lambda.json",
        )
        assert (tmp_path / "nested" / "dir" / "lambda.json").exists()

    def test_mismatched_lengths_raises(self, tmp_path):
        with pytest.raises(ValueError, match="same length"):
            log_lambda_vectors(
                "cond",
                ["A", "B"],
                [0.5, 0.5],
                [1.0],  # wrong length
                tmp_path / "out.json",
            )

    def test_stored_values_match_input(self, tmp_path):
        lam_cls = [0.4, 0.6]
        lam_reg = [0.7, 0.3]
        log_lambda_vectors("c1", ["X", "Y"], lam_cls, lam_reg, tmp_path / "lv.json")
        data = json.loads((tmp_path / "lv.json").read_text())
        assert data["lambda_cls"] == pytest.approx(lam_cls)
        assert data["lambda_reg"] == pytest.approx(lam_reg)


class TestSaveMetricsCSV:
    def test_creates_csv(self, tmp_path):
        metrics = [{"epoch": 1, "map50_95": 42.1, "map50": 60.3}]
        csv_path = tmp_path / "metrics.csv"
        save_metrics_csv(metrics, csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "epoch" in content
        assert "42.1" in content

    def test_appends_rows(self, tmp_path):
        csv_path = tmp_path / "metrics.csv"
        save_metrics_csv([{"epoch": 1, "ap": 10.0}], csv_path)
        save_metrics_csv([{"epoch": 2, "ap": 11.0}], csv_path)
        lines = csv_path.read_text().strip().split("\n")
        # header + 2 data rows
        assert len(lines) == 3

    def test_empty_metrics_no_file(self, tmp_path):
        csv_path = tmp_path / "metrics.csv"
        save_metrics_csv([], csv_path)
        assert not csv_path.exists()
