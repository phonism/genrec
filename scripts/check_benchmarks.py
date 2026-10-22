"""Audit the published October matrix without importing training dependencies.

Run from any directory. Optionally pass --artifacts out/matrix_20261022 to
also verify the local primary/ and ablations/ raw-results mirror and hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from .validate_matrix_results import validate_payload
except ImportError:
    from validate_matrix_results import validate_payload

LABELS = {
    "sasrec": "SASRec (CE)", "hstu": "HSTU (CE)",
    "tiger": "TIGER (matrix recipe)", "lcrec": "LCRec (4B, seq20, gb128)",
    "cobra": "COBRA (optimized light)", "rpg": "RPG (direct)",
    "notellm": "NoteLLM (GCL+CSFT, I2I)",
    "onerec_sft": "OneRec SFT (3 SID)",
    "onerec_grpo": "OneRec + GRPO (selected)",
}
METRICS = ("Recall@1", "Recall@5", "Recall@10", "NDCG@5", "NDCG@10")


def section(text, title):
    """Extract one Markdown section, stopping at the same or a higher level."""
    match = re.search(r"^" + re.escape(title) + r"\s*$", text, re.M)
    if not match:
        return ""
    level = len(title) - len(title.lstrip("#"))
    rest = text[match.end():]
    end = re.search(r"^#{1," + str(level) + r"} ", rest, re.M)
    return rest[:end.start()] if end else rest


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["split"], row["protocol"])].append(row)
    result = {}
    for key, records in groups.items():
        metrics = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in records if row[metric] != ""]
            if values:
                metrics[metric] = {"mean": statistics.mean(values),
                                   "std": statistics.pstdev(values)}
        result[key] = metrics
    return groups, result


def fmt(metrics, metric, with_std=True):
    if metric not in metrics:
        return "—"
    value = metrics[metric]
    mean = f'{value["mean"]:.4f}'
    return f'{mean} ± {value["std"]:.4f}' if with_std else mean


def check_snapshot(root, artifacts=None):
    root = Path(root)
    errors = []
    spec = json.loads((root / "config/matrix/amazon2014_20261022.json").read_text())
    summary = json.loads((root / "results/matrix_20261022.summary.json").read_text())
    main = list(csv.DictReader((root / spec["results_csv"]).open()))
    extras = list(csv.DictReader((root / "results/ablations_20261022.csv").open()))
    expected = set(itertools.product(spec["models"], spec["splits"],
                                     spec["protocols"], map(str, spec["seeds"])))
    keys = [(r["model"], r["split"], r["protocol"], r["seed"]) for r in main]
    if len(keys) != 162 or len(set(keys)) != 162 or set(keys) != expected:
        errors.append("main CSV must contain exactly the 162 unique expected cells")
    extra_keys = {(r["model"], r["recipe"], r["split"], r["protocol"], r["seed"]) for r in extras}
    if len(extras) != 6 or len(extra_keys) != 6:
        errors.append("supplemental CSV must contain six unique cells")
    for row in main + extras:
        if row["evaluation_level"] != "item":
            errors.append(f"non-item metric: {row['result_path']}")
        for metric in METRICS:
            value = row[metric]
            if value == "" and metric == "Recall@1" and row["model"] == "tiger":
                continue
            try:
                number = float(value)
                assert math.isfinite(number) and 0 <= number <= 1
            except (ValueError, AssertionError):
                errors.append(f"invalid {metric}: {row['result_path']}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["result_sha256"]):
            errors.append(f"invalid result checksum: {row['result_path']}")
    for row in main:
        recipe = spec["models"].get(row["model"], {})
        if row["recipe"] != recipe.get("recipe"):
            errors.append(f"CSV/preset recipe mismatch: {row['model']}")
        if int(row["max_seq_len"]) != (20 if row["model"] == "lcrec" else 50):
            errors.append(f"unexpected history length: {row['result_path']}")
    for recipe in spec["models"].values():
        for path in [recipe["config"], *recipe.get("split_configs", {}).values()]:
            if not (root / path).is_file():
                errors.append(f"preset references missing config: {path}")

    # Avoid hiding a malformed metric behind an aggregation exception.
    if errors:
        return errors
    groups, stats = aggregate(main)
    saved = {(r["model"], r["split"], r["protocol"]): r for r in summary["rows"]}
    if set(saved) != set(stats) or len(summary["rows"]) != 54:
        errors.append("summary must contain exactly 54 matching groups")
    for key, metrics in stats.items():
        record = saved.get(key, {})
        if record.get("seeds") != [42, 43, 44]:
            errors.append(f"incomplete summary seeds: {key}")
        if set(record.get("metrics", {})) != set(metrics):
            errors.append(f"summary metric names mismatch: {key}")
        for metric, values in metrics.items():
            for stat, value in values.items():
                actual = record.get("metrics", {}).get(metric, {}).get(stat, math.nan)
                if not math.isclose(actual, value, rel_tol=0, abs_tol=1e-12):
                    errors.append(f"summary {key} {metric} {stat} mismatch")
        if sorted(record.get("result_paths", [])) != sorted(r["result_path"] for r in groups[key]):
            errors.append(f"summary artifact paths mismatch: {key}")
    fallback = sum(int(r["best_epoch"]) == -1 for r in main if r["model"] == "onerec_grpo")
    if fallback != 16 or summary["grpo_sft_fallback_cells"] != fallback:
        errors.append("GRPO SFT-fallback count mismatch")

    readme = (root / "README.md").read_text()
    bench = (root / "BENCHMARKS.md").read_text()
    for (model, split, protocol), metrics in stats.items():
        if protocol == "leave_one_out":
            line = "| " + " | ".join([LABELS[model]] + [fmt(metrics, m, False) for m in METRICS[1:]]) + " |"
            if line not in section(readme, f"### Amazon 2014 {split.title()}").splitlines():
                errors.append(f"README row mismatch: {model}/{split}")
        title = ("## Leave-one-out: 27 complete rows" if protocol == "leave_one_out"
                 else "## Global temporal: 27 complete rows")
        line = "| " + " | ".join([LABELS[model], split.title(), "3"] +
                                [fmt(metrics, m) for m in (METRICS[0], METRICS[1], METRICS[2], METRICS[4])]) + " |"
        if line not in section(bench, title).splitlines():
            errors.append(f"BENCHMARKS row mismatch: {model}/{split}/{protocol}")
    for lang in ("en", "zh"):
        index = (root / f"docs/{lang}/index.md").read_text()
        for protocol, title in (("leave_one_out", "### Leave-one-out"),
                                ("global_temporal", "### Global temporal")):
            for model, label in LABELS.items():
                line = "| " + " | ".join([label] + [fmt(stats[(model, split, protocol)], "Recall@10")
                                                     for split in spec["splits"]]) + " |"
                if line not in section(index, title).splitlines():
                    errors.append(f"{lang} index row mismatch: {model}/{protocol}")
    _, extra_stats = aggregate(extras)
    for (model, split, protocol), metrics in extra_stats.items():
        label = ("HSTU (corrected Meta-local sampled softmax)" if model == "hstu"
                 else "OneRec SFT (4-code suffix ablation)")
        line = "| " + " | ".join([label, split.title(), protocol, "3"] +
                                [fmt(metrics, m) for m in (METRICS[0], METRICS[1], METRICS[2], METRICS[4])]) + " |"
        if line not in section(bench, "## Supplemental three-seed rows").splitlines():
            errors.append(f"supplemental table mismatch: {model}")

    if artifacts is not None:
        for collection, rows in (("primary", main), ("ablations", extras)):
            for row in rows:
                path = Path(artifacts) / collection / row["model"] / row["split"] / row["protocol"] / f"seed{row['seed']}" / "results.json"
                if not path.is_file():
                    errors.append(f"missing artifact: {path}")
                    continue
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != row["result_sha256"]:
                    errors.append(f"artifact checksum mismatch: {path}")
                payload = json.loads(raw)
                errors.extend(f"{path}: {e}" for e in validate_payload(payload))
                if (payload["model"], payload["split"], str(payload["seed"]), payload["config"]["evaluation_protocol"]) != (row["model"], row["split"], row["seed"], row["protocol"]):
                    errors.append(f"artifact cell mismatch: {path}")
                for metric in METRICS:
                    if row[metric] != "" and not math.isclose(float(row[metric]), payload["metrics"][metric], rel_tol=0, abs_tol=1e-12):
                        errors.append(f"artifact/CSV {metric} mismatch: {path}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    errors = check_snapshot(args.root, args.artifacts)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)
    print("PASS: 162 main + 6 supplemental cells; CSV, 54 summary rows, README, BENCHMARKS and en/zh index tables agree.")
    if args.artifacts:
        print("PASS: all 168 raw-result checksums, item-level metadata and metrics agree.")


if __name__ == "__main__":
    main()
