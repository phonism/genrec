"""Validate honest item-level reproducibility-matrix result artifacts.

The validator is intentionally stricter than the aggregator: it rejects
missing provenance, SID-level headline metrics, duplicate cells, malformed
metric values, and (when an expected matrix is supplied) missing cells.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
import sys
from collections import defaultdict


PROTOCOLS = ("leave_one_out", "global_temporal")
REQUIRED_METRICS = ("Recall@5", "Recall@10", "NDCG@10")
SID_RESOLVED_MODELS = {
    "tiger",
    "lcrec",
    "onerec_sft",
    "onerec_grpo",
    "onerec_clm",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def find_results(roots):
    paths = []
    for root in roots:
        if os.path.isfile(root):
            paths.append(root)
            continue
        for dirpath, _, filenames in os.walk(root):
            if "results.json" in filenames:
                paths.append(os.path.join(dirpath, "results.json"))
    return sorted(set(paths))


def cell_key(payload):
    return (
        payload.get("model"),
        payload.get("split"),
        payload.get("config", {}).get("evaluation_protocol"),
        payload.get("seed"),
    )


def validate_payload(payload):
    errors = []
    if not isinstance(payload, dict):
        return ["top-level JSON value must be an object"]

    model, split, protocol, seed = cell_key(payload)
    if not isinstance(model, str) or not model:
        errors.append("model must be a non-empty string")
    if not isinstance(split, str) or not split:
        errors.append("split must be a non-empty string")
    if protocol not in PROTOCOLS:
        errors.append(f"unsupported/missing evaluation protocol: {protocol!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        errors.append(f"seed must be an integer: {seed!r}")

    extra = payload.get("extra", {})
    level = extra.get("evaluation_level")
    if level != "item":
        errors.append(f"headline evaluation_level must be 'item', got {level!r}")
    if model in SID_RESOLVED_MODELS:
        resolution = extra.get("sid_resolution")
        if resolution != "most_popular_training_only":
            errors.append(
                "SID-based model must record sid_resolution="
                f"'most_popular_training_only', got {resolution!r}"
            )

    bundle = payload.get("source_bundle")
    if not isinstance(bundle, str) or not bundle.endswith(".tar.gz"):
        errors.append("source_bundle must name the exact .tar.gz archive")
    digest = payload.get("source_bundle_sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        errors.append("source_bundle_sha256 must be 64 lowercase hex characters")
    git_sha = payload.get("git_sha")
    if not isinstance(git_sha, str) or not GIT_SHA_RE.fullmatch(git_sha):
        errors.append(f"git_sha must be a concrete 7-40 digit hex SHA, got {git_sha!r}")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics must be an object")
        return errors
    for name in REQUIRED_METRICS:
        value = metrics.get(name)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or not 0.0 <= value <= 1.0):
            errors.append(f"{name} must be finite and within [0, 1], got {value!r}")
    recall5 = metrics.get("Recall@5")
    recall10 = metrics.get("Recall@10")
    if (isinstance(recall5, (int, float))
            and isinstance(recall10, (int, float))
            and recall5 > recall10):
        errors.append(f"Recall@5 ({recall5}) cannot exceed Recall@10 ({recall10})")
    return errors


def expected_keys(models, splits, protocols, seeds):
    if not all((models, splits, protocols, seeds)):
        return set()
    return set(itertools.product(models, splits, protocols, seeds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", help="results.json files or directories to scan")
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--splits", nargs="+", default=[])
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=[])
    args = parser.parse_args()

    selected_models = set(args.models)
    records = defaultdict(list)
    failures = []
    for path in find_results(args.roots):
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append((path, [f"cannot load JSON: {exc}"]))
            continue
        if selected_models and payload.get("model") not in selected_models:
            continue
        errors = validate_payload(payload)
        if errors:
            failures.append((path, errors))
        records[cell_key(payload)].append(path)

    for key, paths in sorted(records.items(), key=lambda item: repr(item[0])):
        if len(paths) > 1:
            failures.append((repr(key), ["duplicate cell: " + ", ".join(paths)]))

    expected = expected_keys(args.models, args.splits, args.protocols, args.seeds)
    missing = sorted(expected.difference(records), key=repr)
    if missing:
        failures.append(("expected matrix", [f"missing cell {key!r}" for key in missing]))

    if failures:
        for path, errors in failures:
            print(f"FAIL {path}", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
        print(
            f"[validate] {len(records)} unique cells, {len(failures)} failing entries",
            file=sys.stderr,
        )
        return 1

    print(f"[validate] OK: {len(records)} unique honest item-level cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
