"""Unit tests for SID collision reporting (scripts/report_sid_collisions.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from report_sid_collisions import (  # noqa: E402
    collision_report,
    comparison_table,
    load_sem_ids,
    prefix_collision_stats,
)

# 10 items, 8 unique full SIDs -> two collision groups of size 2.
SEM = np.array([[0, 0], [0, 1], [0, 1], [1, 0], [1, 1],
                [2, 0], [2, 1], [3, 0], [3, 1], [3, 1]])


def test_full_collision_matches_known_counts():
    r = collision_report(SEM, codebook_size=4)
    assert r["n_items"] == 10
    assert r["unique_sids"] == 8
    assert r["collision_rate"] == pytest.approx(0.2)
    assert r["max_collision_group"] == 2
    assert r["items_in_collisions"] == 4
    assert r["frac_items_in_collisions"] == pytest.approx(0.4)


def test_capacity_utilization():
    # 8 unique SIDs out of codebook_size**n_layers = 4**2 = 16.
    r = collision_report(SEM, codebook_size=4)
    assert r["capacity_utilization"] == pytest.approx(8 / 16)


def test_codebook_size_inferred_from_max():
    r = collision_report(SEM)  # max code is 3 -> inferred codebook_size 4
    assert r["codebook_size"] == 4


def test_prefix_collision_accumulates():
    # depth 1: prefixes {0,1,2,3} -> 4 unique of 10 -> 60% collision
    # depth 2 (full): 8 unique -> 20% collision
    stats = prefix_collision_stats(SEM)
    assert stats[0]["depth"] == 1
    assert stats[0]["unique_prefixes"] == 4
    assert stats[0]["collision_rate"] == pytest.approx(0.6)
    assert stats[1]["collision_rate"] == pytest.approx(0.2)
    # prefix collision is monotonically non-increasing with depth
    rates = [s["collision_rate"] for s in stats]
    assert rates == sorted(rates, reverse=True)


def test_group_size_histogram():
    r = collision_report(SEM, codebook_size=4)
    hist = r["group_size_histogram"]
    # 6 SIDs appear once, 2 SIDs appear twice
    assert hist[1] == 6
    assert hist[2] == 2


def test_no_collision_case():
    sem = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    r = collision_report(sem, codebook_size=2)
    assert r["collision_rate"] == pytest.approx(0.0)
    assert r["unique_sids"] == 4
    assert r["max_collision_group"] == 1
    assert r["items_in_collisions"] == 0


def test_1d_sem_ids_reshaped():
    r = collision_report(np.array([0, 0, 1, 2]))
    assert r["n_layers"] == 1
    assert r["n_items"] == 4
    assert r["unique_sids"] == 3


def test_load_npy(tmp_path):
    p = tmp_path / "sid.npy"
    np.save(p, SEM)
    loaded = load_sem_ids(str(p))
    assert np.array_equal(loaded, SEM)


def test_load_json_list(tmp_path):
    p = tmp_path / "sid.json"
    p.write_text(json.dumps(SEM.tolist()))
    loaded = load_sem_ids(str(p))
    assert np.array_equal(loaded, SEM)


def test_load_json_dict_ordered_by_int_key(tmp_path):
    p = tmp_path / "sid.json"
    # deliberately out of order; loader must sort by integer key
    p.write_text(json.dumps({"2": [9, 9], "0": [1, 1], "10": [3, 3], "1": [2, 2]}))
    loaded = load_sem_ids(str(p))
    assert loaded.tolist() == [[1, 1], [2, 2], [9, 9], [3, 3]]


def test_load_text_ints(tmp_path):
    p = tmp_path / "sid.tsv"
    p.write_text("0\t0\n0\t1\n1\t2\n")
    loaded = load_sem_ids(str(p))
    assert loaded.tolist() == [[0, 0], [0, 1], [1, 2]]


def test_load_token_strings(tmp_path):
    # OneRec-style token lines: <C0_5><C1_6><C2_2>
    p = tmp_path / "sid.txt"
    p.write_text("<C0_5><C1_6><C2_2>\n<C0_5><C1_6><C2_3>\n")
    loaded = load_sem_ids(str(p))
    assert loaded.tolist() == [[5, 6, 2], [5, 6, 3]]


def test_inconsistent_width_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([[0, 1], [0, 1, 2]]))
    with pytest.raises(ValueError):
        load_sem_ids(str(p))


def test_comparison_table_renders_rows():
    r = collision_report(SEM, codebook_size=4)
    table = comparison_table([("A", r), ("B", r)])
    assert "| A |" in table and "| B |" in table
    assert "collision_rate" in table
    # per-prefix sub-table present for multi-layer SIDs
    assert "@1" in table and "@2" in table
