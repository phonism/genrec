"""Unit tests for codebook health diagnostics."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_codebook_health import collision_stats, per_layer_stats  # noqa: E402


def test_per_layer_stats_detects_collapse():
    # 100 items, 1 layer, codebook=8. 80 items assigned to code 0 -> max_freq=80%.
    sem_ids = np.zeros((100, 1), dtype=np.int64)
    sem_ids[80:, 0] = np.array([1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6])
    stats = per_layer_stats(sem_ids, codebook_size=8)[0]
    assert stats["used_codes"] == 8
    assert stats["max_freq"] == pytest.approx(0.80)
    assert stats["top_code"] == 0


def test_per_layer_stats_reports_dead_codes():
    # 50 items all assigned to code 0 out of 32 — 31 dead codes.
    sem_ids = np.zeros((50, 1), dtype=np.int64)
    stats = per_layer_stats(sem_ids, codebook_size=32)[0]
    assert stats["dead_codes"] == 31
    assert stats["usage"] == pytest.approx(1 / 32)
    assert stats["max_freq"] == pytest.approx(1.0)


def test_collision_stats_counts_duplicates():
    # 10 items, 8 unique sem ids — two collisions.
    sem_ids = np.array([[0, 0], [0, 1], [0, 1], [1, 0], [1, 1],
                        [2, 0], [2, 1], [3, 0], [3, 1], [3, 1]])
    c = collision_stats(sem_ids)
    assert c["n_items"] == 10
    assert c["unique_sids"] == 8
    assert c["collision_rate"] == pytest.approx(0.2)
    assert c["max_collision_group"] == 2
