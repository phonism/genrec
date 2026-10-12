import pytest

from scripts.aggregate_results import group, render_md, summarize


def test_summarize_uses_population_standard_deviation():
    mean, sd = summarize([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert sd == pytest.approx((2.0 / 3.0) ** 0.5)


def test_aggregation_never_mixes_protocol_or_metric_level():
    records = [
        {
            "model": "onerec_sft", "split": "beauty", "seed": 42,
            "metrics": {"Recall@10": 0.09},
            "config": {"evaluation_protocol": "leave_one_out"},
            "extra": {"evaluation_level": "item"},
        },
        {
            "model": "onerec_sft", "split": "beauty", "seed": 42,
            "metrics": {"Recall@10": 0.06},
            "config": {"evaluation_protocol": "global_temporal"},
            "extra": {"evaluation_level": "item"},
        },
        {
            "model": "onerec_sft", "split": "beauty", "seed": 42,
            "metrics": {"Recall@10": 0.12},
            "config": {"evaluation_protocol": "leave_one_out"},
            "extra": {"evaluation_level": "sid"},
        },
    ]

    groups = group(records)
    assert len(groups) == 3
    table = render_md(groups, ["Recall@10"])
    assert "protocol" in table
    assert "level" in table
    assert "global_temporal" in table
