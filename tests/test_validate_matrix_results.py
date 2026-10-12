import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_matrix_results.py"
SPEC = importlib.util.spec_from_file_location("validate_matrix_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def valid_payload():
    return {
        "model": "tiger",
        "split": "beauty",
        "seed": 42,
        "git_sha": "e52cc02",
        "source_bundle": "genrec-code-test.tar.gz",
        "source_bundle_sha256": "a" * 64,
        "config": {"evaluation_protocol": "leave_one_out"},
        "extra": {
            "evaluation_level": "item",
            "sid_resolution": "most_popular_training_only",
        },
        "metrics": {"Recall@5": 0.1, "Recall@10": 0.2, "NDCG@10": 0.15},
    }


def test_valid_payload_is_accepted():
    assert MODULE.validate_payload(valid_payload()) == []


def test_sid_level_and_unknown_provenance_are_rejected():
    payload = valid_payload()
    payload["extra"]["evaluation_level"] = "sid"
    payload["git_sha"] = "unknown"
    errors = MODULE.validate_payload(payload)
    assert any("evaluation_level" in error for error in errors)
    assert any("git_sha" in error for error in errors)


def test_sid_model_requires_training_only_resolution_provenance():
    payload = valid_payload()
    del payload["extra"]["sid_resolution"]
    errors = MODULE.validate_payload(payload)
    assert any("sid_resolution" in error for error in errors)

    payload["model"] = "sasrec"
    assert not any(
        "sid_resolution" in error for error in MODULE.validate_payload(payload)
    )


def test_recall_monotonicity_is_checked():
    payload = valid_payload()
    payload["metrics"]["Recall@5"] = 0.3
    assert any("cannot exceed" in error for error in MODULE.validate_payload(payload))


def test_expected_keys_requires_a_complete_specification():
    assert MODULE.expected_keys(["tiger"], ["beauty"], [], [42]) == set()
    assert MODULE.expected_keys(
        ["tiger"], ["beauty"], ["leave_one_out"], [42]
    ) == {("tiger", "beauty", "leave_one_out", 42)}
