"""Smoke tests for the new Phase-1 infrastructure helpers."""
import json
import os

import numpy as np
import pytest
import torch

from genrec.trainers.trainer_utils import (
    get_git_sha,
    save_run_results,
    set_seed,
)


def test_set_seed_makes_torch_deterministic():
    set_seed(123)
    a = torch.randn(8)
    set_seed(123)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_set_seed_makes_numpy_deterministic():
    set_seed(7)
    a = np.random.randn(8)
    set_seed(7)
    b = np.random.randn(8)
    assert np.array_equal(a, b)


def test_different_seeds_diverge():
    set_seed(1)
    a = torch.randn(16)
    set_seed(2)
    b = torch.randn(16)
    assert not torch.equal(a, b)


def test_save_run_results_writes_expected_schema(tmp_path):
    out = save_run_results(
        save_dir=str(tmp_path),
        model="sasrec",
        split="beauty",
        seed=42,
        metrics={"Recall@10": 0.0851, "NDCG@10": 0.0421},
        config={"epochs": 100, "batch_size": 128, "non_jsonable": object()},
        extra={"best_valid_recall@10": 0.087},
    )
    assert os.path.isfile(out)
    with open(out) as f:
        payload = json.load(f)
    assert payload["model"] == "sasrec"
    assert payload["split"] == "beauty"
    assert payload["seed"] == 42
    assert payload["metrics"]["Recall@10"] == pytest.approx(0.0851)
    assert payload["config"]["epochs"] == 100
    # Non-jsonable values must be coerced, not crash.
    assert isinstance(payload["config"]["non_jsonable"], str)
    assert payload["extra"]["best_valid_recall@10"] == pytest.approx(0.087)
    assert "timestamp" in payload
    assert "git_sha" in payload


def test_save_run_results_records_exact_source_bundle(tmp_path, monkeypatch):
    digest = "a" * 64
    monkeypatch.setenv("GENREC_SOURCE_BUNDLE_NAME", "genrec-code-test.tar.gz")
    monkeypatch.setenv("GENREC_SOURCE_BUNDLE_SHA256", digest)
    out = save_run_results(
        save_dir=str(tmp_path),
        model="sasrec",
        split="beauty",
        seed=42,
        metrics={"Recall@10": 0.1},
    )
    with open(out) as f:
        payload = json.load(f)
    assert payload["source_bundle"] == "genrec-code-test.tar.gz"
    assert payload["source_bundle_sha256"] == digest


def test_save_run_results_records_locked_execution_cell(tmp_path, monkeypatch):
    monkeypatch.setenv("GENREC_CELL_ID", "tiger.beauty.leave_one_out.s42.train")
    monkeypatch.setenv("GENREC_PHYSICAL_GPUS", "3")
    out = save_run_results(
        save_dir=str(tmp_path),
        model="tiger",
        split="beauty",
        seed=42,
        metrics={"Recall@10": 0.1},
    )
    with open(out) as f:
        payload = json.load(f)
    assert payload["execution"] == {
        "cell_id": "tiger.beauty.leave_one_out.s42.train",
        "physical_gpus": "3",
    }


def test_get_git_sha_runs_without_crashing():
    sha = get_git_sha()
    assert isinstance(sha, str)
    assert sha  # either a real sha or "unknown"


def test_get_git_sha_prefers_exact_source_override(monkeypatch):
    monkeypatch.setenv("GENREC_SOURCE_GIT_SHA", "abcdef123456")
    assert get_git_sha() == "abcdef123456"
