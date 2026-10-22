"""Guard the evidence chain from published per-seed results to public tables."""
import json
import shutil
from pathlib import Path

import pytest

from scripts.check_benchmarks import check_snapshot
from scripts.run_matrix import build_cells, cell_done, run_cell

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = "out/reproduce/{protocol}/{model}/{split}/seed{seed}"


def test_published_tables_match_per_seed_csv():
    assert check_snapshot(ROOT) == []


@pytest.fixture
def snapshot(tmp_path):
    for name in ("README.md", "BENCHMARKS.md", "results", "config", "docs/en/index.md", "docs/zh/index.md"):
        src, dst = ROOT / name, tmp_path / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)
    return tmp_path


def test_audit_rejects_a_stale_readme_number(snapshot):
    readme = snapshot / "README.md"
    readme.write_text(readme.read_text().replace("0.0858", "0.9999", 1))
    assert "README row mismatch: sasrec/beauty" in check_snapshot(snapshot)


def test_audit_rejects_mixing_the_four_code_recipe_into_main(snapshot):
    csv = snapshot / "results/matrix_20261022.csv"
    csv.write_text(csv.read_text().replace("v2b_3sid,", "v2b_4sid,", 1))
    assert "CSV/preset recipe mismatch: onerec_sft" in check_snapshot(snapshot)


def test_published_preset_prepares_the_matching_recipes_and_dependencies():
    cells = build_cells(
        ["onerec_grpo", "cobra", "lcrec", "onerec_sft"], ["sports"], [43],
        TEMPLATE, {}, "global_temporal", "amazon2014_20261022")
    by_model = {c.model: c for c in cells}
    assert cells.index(by_model["onerec_sft"]) < cells.index(by_model["onerec_grpo"])
    assert by_model["lcrec"].config_path.endswith("lcrec_paper_recipe.gin")
    assert by_model["cobra"].config_path.endswith("cobra_optimized.gin")
    assert by_model["lcrec"].required_artifacts == (
        "out/lcrec/amazon/sports/rqvae_paper_recipe/checkpoint_best_health.pt",)
    assert by_model["onerec_grpo"].sft_checkpoint_path == (
        "out/reproduce/global_temporal/onerec_sft/sports/seed43/checkpoint_best")
    assert by_model["onerec_sft"].recipe == "v2b_3sid"
    assert all(any("wandb_logging=False" in b for b in c.gin_overrides()) for c in cells)


def test_preset_does_not_silently_accept_unpublished_cells_or_configs():
    with pytest.raises(ValueError, match="outside"):
        build_cells(["onerec_clm"], ["beauty"], [42], TEMPLATE, {},
                    preset="amazon2014_20261022")
    with pytest.raises(ValueError, match="--config"):
        build_cells(["lcrec"], ["beauty"], [42], TEMPLATE,
                    {"lcrec": "config/lcrec/amazon/lcrec.gin"},
                    preset="amazon2014_20261022")


def test_preset_fails_before_training_without_source_provenance(monkeypatch, tmp_path):
    monkeypatch.delenv("GENREC_SOURCE_BUNDLE_NAME", raising=False)
    monkeypatch.delenv("GENREC_SOURCE_BUNDLE_SHA256", raising=False)
    output = tmp_path / "not-created"
    cell = build_cells(["sasrec"], ["beauty"], [42], str(output), {},
                       preset="amazon2014_20261022")[0]
    assert run_cell(cell, dry_run=False, log_file=None) == 2
    assert not output.exists()


def test_preset_does_not_report_success_without_a_result(monkeypatch, tmp_path):
    monkeypatch.setenv("GENREC_SOURCE_BUNDLE_NAME", "source.tar.gz")
    monkeypatch.setenv("GENREC_SOURCE_BUNDLE_SHA256", "a" * 64)
    monkeypatch.setattr("scripts.run_matrix.subprocess.call", lambda *args, **kwargs: 0)
    cell = build_cells(["sasrec"], ["beauty"], [42], str(tmp_path), {},
                       preset="amazon2014_20261022")[0]
    assert run_cell(cell, dry_run=False, log_file=None) == 2
    assert (tmp_path / "matrix_cell.json").is_file()


def test_skip_rejects_wrong_protocol_and_stale_preset_plan(tmp_path):
    cell = build_cells(["onerec_sft"], ["beauty"], [42], str(tmp_path), {},
                       preset="amazon2014_20261022")[0]
    payload = {
        "model": "onerec_sft", "split": "beauty", "seed": 42,
        "config": {"evaluation_protocol": "leave_one_out"},
        "metrics": {"Recall@5": 0.04, "Recall@10": 0.06, "NDCG@10": 0.03},
        "extra": {"evaluation_level": "item", "sid_resolution": "most_popular_training_only"},
        "source_bundle": "source.tar.gz", "source_bundle_sha256": "a" * 64,
        "git_sha": "1234567",
    }
    result = tmp_path / "results.json"
    result.write_text(json.dumps(payload))
    assert not cell_done(cell)  # Missing recipe plan.
    plan = tmp_path / "matrix_cell.json"
    plan.write_text(json.dumps(cell.plan()))
    assert cell_done(cell)
    payload["config"]["evaluation_protocol"] = "global_temporal"
    result.write_text(json.dumps(payload))
    assert not cell_done(cell)
    payload["config"]["evaluation_protocol"] = "leave_one_out"
    result.write_text(json.dumps(payload))
    stale = cell.plan()
    stale["recipe"] = "four_code_ablation"
    plan.write_text(json.dumps(stale))
    assert not cell_done(cell)
