"""Audit the published October matrix without importing training dependencies.

Run from any directory. Optionally pass --artifacts out/matrix_20261022 to
also verify the local primary/ and ablations/ raw-results mirror and hashes.
Use --home-artifacts for the separate Home result/command-receipt mirror.
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
    from .run_matrix import Cell
except ImportError:
    from validate_matrix_results import validate_payload
    from run_matrix import Cell

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


def check_home_snapshot(root, artifacts=None):
    """Check Home tables, provenance and optional recovered raw artifacts."""
    root = Path(root)
    spec = json.loads((root / "config/matrix/amazon2014_home_20261022.json").read_text())
    summary = json.loads((root / spec["summary_json"]).read_text())
    provenance = json.loads((root / spec["provenance_json"]).read_text())
    rows = list(csv.DictReader((root / spec["results_csv"]).open()))
    errors = []
    expected = set(itertools.product(spec["models"], spec["splits"],
                                    spec["protocols"], map(str, spec["seeds"])))
    keys = [(r["model"], r["split"], r["protocol"], r["seed"]) for r in rows]
    if len(rows) != 54 or len(set(keys)) != 54 or set(keys) != expected:
        errors.append("Home CSV must contain exactly the 54 unique expected cells")
    if spec.get("status") != "complete":
        errors.append("Home preset must record completed status")
    for row in rows:
        recipe = spec["models"].get(row["model"], {})
        if row["recipe"] != recipe.get("recipe"):
            errors.append(f"Home CSV/preset recipe mismatch: {row['model']}")
        if row["evaluation_level"] != "item":
            errors.append(f"Home non-item metric: {row['result_path']}")
        if row["max_seq_len"] != ("20" if row["model"] == "lcrec" else "50"):
            errors.append(f"Home history length mismatch: {row['result_path']}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["result_sha256"]):
            errors.append(f"Home result checksum malformed: {row['result_path']}")
        for metric in METRICS:
            value = row[metric]
            if value == "" and metric == "Recall@1" and row["model"] == "tiger":
                continue
            try:
                assert math.isfinite(float(value)) and 0 <= float(value) <= 1
            except (ValueError, AssertionError):
                errors.append(f"Home invalid {metric}: {row['result_path']}")
        for field in ("git_sha", "source_bundle", "source_bundle_sha256"):
            if row[field] != provenance.get(field) or row[field] != summary.get(field):
                errors.append(f"Home source provenance mismatch: {row['result_path']}")
    if errors:
        return errors

    groups, stats = aggregate(rows)
    saved = {(r["model"], r["split"], r["protocol"]): r for r in summary["rows"]}
    if set(saved) != set(stats) or len(summary["rows"]) != 18:
        errors.append("Home summary must contain 18 unique matching groups")
    for key, metrics in stats.items():
        record = saved.get(key, {})
        if record.get("seeds") != [42, 43, 44]:
            errors.append(f"Home incomplete summary seeds: {key}")
        if set(record.get("metrics", {})) != set(metrics):
            errors.append(f"Home summary metric names mismatch: {key}")
        for metric, values in metrics.items():
            for stat, value in values.items():
                actual = record.get("metrics", {}).get(metric, {}).get(stat, math.nan)
                if not math.isclose(actual, value, rel_tol=0, abs_tol=1e-12):
                    errors.append(f"Home summary {key} {metric} {stat} mismatch")
        if sorted(record.get("result_paths", [])) != sorted(r["result_path"] for r in groups[key]):
            errors.append(f"Home summary artifact paths mismatch: {key}")
    fallback = sum(int(r["best_epoch"]) == -1 for r in rows if r["model"] == "onerec_grpo")
    if fallback != 6 or summary.get("grpo_sft_fallback_cells") != fallback:
        errors.append("Home GRPO SFT-fallback count mismatch")
    if (summary.get("main_cells"), summary.get("main_rows"),
            summary.get("queue_successful_jobs"), summary.get("standard_deviation"),
            summary.get("evaluation_level")) != (54, 18, 68, "population", "item"):
        errors.append("Home summary counts/protocol metadata mismatch")

    recorded = {(r["model"], r["split"], r["protocol"], str(r["seed"])): r
                for r in provenance["cells"]}
    if len(provenance["cells"]) != 54 or set(recorded) != expected:
        errors.append("Home provenance must contain 54 unique command receipts")
    for name, digest in provenance["source_config_sha256"].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f"Home source config checksum mismatch: {name}")
    for row in rows:
        key = (row["model"], row["split"], row["protocol"], row["seed"])
        receipt = recorded.get(key)
        if receipt is None:
            continue
        recipe = spec["models"][row["model"]]
        config = recipe.get("split_configs", {}).get(row["split"], recipe["config"])
        save_dir = provenance["save_dir_template"].format(
            protocol=row["protocol"], model=row["model"], split=row["split"],
            seed=row["seed"])
        sft = (provenance["save_dir_template"].format(
            protocol=row["protocol"], model="onerec_sft", split=row["split"],
            seed=row["seed"]) + "/checkpoint_best"
            if row["model"] == "onerec_grpo" else None)
        cell = Cell(model=row["model"], split=row["split"], seed=int(row["seed"]),
                    config_path=config, save_dir=save_dir,
                    evaluation_protocol=row["protocol"], sft_checkpoint_path=sft,
                    recipe=recipe["recipe"], extra_bindings=tuple(recipe["bindings"]))
        if (receipt["command"][1:] != cell.cmd()[1:]
                or receipt["recipe"] != cell.recipe):
            errors.append(f"Home command/preset mismatch: {row['result_path']}")
        config_hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                         for p in (config, "config/base.gin")}
        if receipt["config_sha256"] != config_hashes:
            errors.append(f"Home command config checksum mismatch: {row['result_path']}")
        if (receipt["result_path"] != row["result_path"]
                or receipt["result_sha256"] != row["result_sha256"]
                or str(Path(row["result_path"]).parent) != save_dir):
            errors.append(f"Home provenance/CSV path or checksum mismatch: {key}")

        if artifacts is not None:
            directory = (Path(artifacts) / row["protocol"] / row["model"]
                         / row["split"] / f"seed{row['seed']}")
            path = directory / "results.json"
            plan_path = directory / "matrix_cell.json"
            if not path.is_file() or not plan_path.is_file():
                errors.append(f"Home missing result or command receipt: {directory}")
                continue
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["result_sha256"]:
                errors.append(f"Home artifact checksum mismatch: {path}")
            payload = json.loads(raw)
            errors.extend(f"{path}: {e}" for e in validate_payload(payload))
            actual_key = (payload["model"], payload["split"],
                          payload["config"]["evaluation_protocol"], str(payload["seed"]))
            if actual_key != key:
                errors.append(f"Home artifact cell mismatch: {path}")
            for metric in METRICS:
                if row[metric] != "" and not math.isclose(
                        float(row[metric]), payload["metrics"][metric],
                        rel_tol=0, abs_tol=1e-12):
                    errors.append(f"Home artifact/CSV {metric} mismatch: {path}")
            if hashlib.sha256(plan_path.read_bytes()).hexdigest() != receipt["receipt_sha256"]:
                errors.append(f"Home receipt checksum mismatch: {plan_path}")
            plan = json.loads(plan_path.read_text())
            if any(plan.get(k) != receipt.get(k)
                   for k in ("recipe", "command", "config_sha256")):
                errors.append(f"Home raw/published command mismatch: {plan_path}")

    bench = section((root / "BENCHMARKS.md").read_text(),
                    "## Home extension: 54 complete cells")
    for (model, split, protocol), metrics in stats.items():
        if protocol == "leave_one_out":
            line = "| " + " | ".join([LABELS[model]] + [
                fmt(metrics, m, False) for m in METRICS[1:]]) + " |"
            for name in ("README.md", "README.zh-CN.md"):
                if line not in section((root / name).read_text(),
                                       "### Amazon 2014 Home").splitlines():
                    errors.append(f"{name} Home row mismatch: {model}")
        title = ("### Leave-one-out" if protocol == "leave_one_out"
                 else "### Global temporal")
        line = "| " + " | ".join([LABELS[model], "3"] + [
            fmt(metrics, m) for m in ("Recall@1", "Recall@5", "Recall@10", "NDCG@10")]) + " |"
        if line not in section(bench, title).splitlines():
            errors.append(f"Home BENCHMARKS row mismatch: {model}/{protocol}")
    for lang in ("en", "zh"):
        index = section((root / f"docs/{lang}/index.md").read_text(), "### Home extension")
        for model, label in LABELS.items():
            line = "| " + " | ".join([label] + [
                fmt(stats[(model, "home", protocol)], "Recall@10")
                for protocol in spec["protocols"]]) + " |"
            if line not in index.splitlines():
                errors.append(f"{lang} Home index row mismatch: {model}")
    return errors


def check_snapshot(root, artifacts=None, home_artifacts=None):
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
    chinese_readme = (root / "README.zh-CN.md").read_text()
    bench = (root / "BENCHMARKS.md").read_text()
    for (model, split, protocol), metrics in stats.items():
        if protocol == "leave_one_out":
            line = "| " + " | ".join([LABELS[model]] + [fmt(metrics, m, False) for m in METRICS[1:]]) + " |"
            if line not in section(readme, f"### Amazon 2014 {split.title()}").splitlines():
                errors.append(f"README row mismatch: {model}/{split}")
            if line not in section(chinese_readme, f"### Amazon 2014 {split.title()}").splitlines():
                errors.append(f"Chinese README row mismatch: {model}/{split}")
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
    errors.extend(check_home_snapshot(root, home_artifacts))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--home-artifacts", type=Path)
    args = parser.parse_args()
    errors = check_snapshot(args.root, args.artifacts, args.home_artifacts)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)
    print("PASS: 162 main + 6 supplemental cells; CSV, 54 summary rows, en/zh README, BENCHMARKS and en/zh index tables agree.")
    if args.artifacts:
        print("PASS: all 168 raw-result checksums, item-level metadata and metrics agree.")
    print("PASS: Home 54 cells / 18 rows; CSV, summary, 54 commands/config hashes and all en/zh tables agree.")
    if args.home_artifacts:
        print("PASS: all 54 Home raw-result and command-receipt checksums agree.")


if __name__ == "__main__":
    main()
