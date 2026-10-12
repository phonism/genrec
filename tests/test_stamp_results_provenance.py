import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "stamp_results_provenance.py"
SPEC = importlib.util.spec_from_file_location("stamp_results_provenance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _result(path: Path) -> None:
    path.write_text(
        json.dumps({"model": "cobra", "git_sha": "unknown"}),
        encoding="utf-8",
    )


def test_stamp_adds_source_and_execution_provenance(tmp_path):
    path = tmp_path / "results.json"
    _result(path)

    MODULE.stamp(
        path,
        "genrec-code-test.tar.gz",
        "a" * 64,
        "e52cc02",
        cell_id="cobra.beauty.leave_one_out.s42.train",
        physical_gpus="2",
        task_id=137,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["execution"] == {
        "cell_id": "cobra.beauty.leave_one_out.s42.train",
        "physical_gpus": "2",
        "task_id": 137,
    }


def test_stamp_rejects_conflicting_execution_provenance(tmp_path):
    path = tmp_path / "results.json"
    _result(path)
    MODULE.stamp(
        path,
        "genrec-code-test.tar.gz",
        "a" * 64,
        "e52cc02",
        task_id=137,
    )

    with pytest.raises(ValueError, match="different execution task_id"):
        MODULE.stamp(
            path,
            "genrec-code-test.tar.gz",
            "a" * 64,
            "e52cc02",
            task_id=999,
        )
