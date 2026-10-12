#!/usr/bin/env python3
"""Select COBRA epoch or BeamFusion weights from validation artifacts only."""

import argparse
import json
import os
import re
from collections import defaultdict
from statistics import mean, pstdev


SEED_RE = re.compile(r"(?:^|/)seed(\d+)(?:/|$)")
EPOCH_RE = re.compile(r"checkpoint_epoch_(\d+)\.pt$")
METRIC_NAMES = ("R@1", "R@5", "R@10", "N@5", "N@10")


def parse_seed(path, payload):
    if payload.get("seed") is not None:
        return int(payload["seed"])
    match = SEED_RE.search(path) or SEED_RE.search(payload.get("checkpoint", ""))
    if not match:
        raise ValueError(f"cannot infer seed from {path}")
    return int(match.group(1))


def iter_records(path, vary):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("eval_split") != "valid":
        raise ValueError(f"selection input is not validation: {path}")
    seed = parse_seed(path, payload)

    if vary == "epoch":
        match = EPOCH_RE.search(payload.get("checkpoint", ""))
        if not match:
            raise ValueError(f"cannot infer checkpoint epoch from {path}")
        metrics = payload.get("metrics") or {}
        yield int(match.group(1)), seed, metrics.get("item_level", {})
        return

    grid = payload.get("fusion_grid") or []
    if not grid:
        raise ValueError(f"fusion artifact has no fusion_grid: {path}")
    for row in grid:
        yield (float(row["tau"]), float(row["psi"])), seed, row["item_level"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+")
    parser.add_argument("--vary", required=True, choices=("epoch", "fusion"))
    parser.add_argument("--expected-seeds", default="42,43,44")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    expected_seeds = {int(value) for value in args.expected_seeds.split(",")}
    grouped = defaultdict(dict)
    for path in args.json_files:
        for key, seed, metrics in iter_records(path, args.vary):
            if seed in grouped[key]:
                raise ValueError(f"duplicate seed {seed} for candidate {key}")
            missing = [name for name in METRIC_NAMES if name not in metrics]
            if missing:
                raise ValueError(f"missing metrics {missing} in {path}")
            grouped[key][seed] = metrics

    incomplete = {
        str(key): sorted(expected_seeds - set(per_seed))
        for key, per_seed in grouped.items()
        if set(per_seed) != expected_seeds
    }
    if incomplete:
        raise ValueError(f"candidate groups do not cover exactly {sorted(expected_seeds)}: {incomplete}")

    candidates = []
    for key, per_seed in grouped.items():
        summary = {}
        for metric in METRIC_NAMES:
            values = [per_seed[seed][metric] for seed in sorted(expected_seeds)]
            summary[metric] = {
                "mean": mean(values),
                "std": pstdev(values),
                "values": dict(zip(map(str, sorted(expected_seeds)), values)),
            }
        candidate = {
            "candidate": key,
            "metrics": summary,
        }
        candidates.append(candidate)

    candidates.sort(
        key=lambda row: (
            row["metrics"]["R@10"]["mean"],
            row["metrics"]["R@1"]["mean"],
            row["metrics"]["N@10"]["mean"],
        ),
        reverse=True,
    )
    result = {
        "selection_split": "valid",
        "vary": args.vary,
        "expected_seeds": sorted(expected_seeds),
        "selection_rule": "R@10 mean desc, R@1 mean desc, N@10 mean desc",
        "winner": candidates[0],
        "candidates": candidates,
        "inputs": [os.path.abspath(path) for path in args.json_files],
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.output_json:
        output_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, output_path)


if __name__ == "__main__":
    main()
