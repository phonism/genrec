"""
Aggregate results.json files into a benchmark table.

Walks the search roots, loads every results.json, groups runs by
(model, split, evaluation protocol, evaluation level), and reports mean +/-
std across seeds for each metric.  This prevents legacy SID-level results from
being averaged together with honest item-level runs.

Example:
    python scripts/aggregate_results.py out/ \
        --metrics Recall@10 NDCG@10 \
        --format md \
        --output BENCHMARKS.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def find_results(roots: List[str]) -> List[str]:
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if "results.json" in filenames:
                found.append(os.path.join(dirpath, "results.json"))
    return sorted(found)


def load_results(paths: List[str]) -> List[dict]:
    out = []
    for p in paths:
        try:
            with open(p) as f:
                out.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARN: skipping {p}: {e}", file=sys.stderr)
    return out


GroupKey = Tuple[str, str, str, str]


def group(records: List[dict]) -> Dict[GroupKey, List[dict]]:
    g: Dict[GroupKey, List[dict]] = defaultdict(list)
    for r in records:
        config = r.get("config", {})
        extra = r.get("extra", {})
        protocol = config.get("evaluation_protocol", "unspecified")
        evaluation_level = extra.get("evaluation_level", "unspecified")
        key = (
            r.get("model", "?"), r.get("split", "?"),
            protocol, evaluation_level,
        )
        g[key].append(r)
    return g


def summarize(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    # The benchmark matrix enumerates the complete planned seed set rather
    # than treating the completed seeds as a sample from a larger population.
    # Keep this aligned with BENCHMARKS.md's documented population standard
    # deviation convention.
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, sd


def collect_metric_keys(groups: Dict[GroupKey, List[dict]],
                        requested: List[str]) -> List[str]:
    if requested:
        return requested
    seen = set()
    for runs in groups.values():
        for r in runs:
            seen.update(r.get("metrics", {}).keys())
    return sorted(seen)


def format_cell(values: List[float]) -> str:
    if not values:
        return "n/a"
    mean, sd = summarize(values)
    if math.isnan(mean):
        return "n/a"
    if len(values) == 1:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {sd:.4f}"


def render_md(groups, metrics: List[str]) -> str:
    header = ["model", "split", "protocol", "level", "n"] + metrics
    sep = ["---"] * len(header)
    rows = [header, sep]
    for (model, split, protocol, level), runs in sorted(groups.items()):
        seeds = sorted({r.get("seed") for r in runs})
        n_str = f"{len(runs)} ({','.join(str(s) for s in seeds)})"
        row = [model, split, protocol, level, n_str]
        for m in metrics:
            vals = [r["metrics"].get(m) for r in runs
                    if isinstance(r.get("metrics", {}).get(m), (int, float))]
            row.append(format_cell([float(v) for v in vals]))
        rows.append(row)
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def render_csv(groups, metrics: List[str]) -> str:
    lines = ["model,split,protocol,level,seed,git_sha," + ",".join(metrics)]
    for (model, split, protocol, level), runs in sorted(groups.items()):
        for r in runs:
            cells = [
                model, split, protocol, level,
                str(r.get("seed", "")), r.get("git_sha", ""),
            ]
            for m in metrics:
                v = r.get("metrics", {}).get(m)
                cells.append(f"{v:.6f}" if isinstance(v, (int, float)) else "")
            lines.append(",".join(cells))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="Dirs to walk for results.json")
    ap.add_argument("--metrics", nargs="*", default=None,
                    help="Metric keys to include. Default: union across all runs.")
    ap.add_argument("--format", choices=["md", "csv"], default="md")
    ap.add_argument("--output", "-o", default=None,
                    help="Write to file instead of stdout")
    args = ap.parse_args()

    paths = find_results(args.roots)
    print(f"[aggregate] found {len(paths)} results.json under {args.roots}",
          file=sys.stderr)
    records = load_results(paths)
    if not records:
        print("[aggregate] no results found", file=sys.stderr)
        sys.exit(1)

    groups = group(records)
    metrics = collect_metric_keys(groups, args.metrics or [])
    text = (render_md if args.format == "md" else render_csv)(groups, metrics)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"[aggregate] wrote {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
