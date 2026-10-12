"""Aggregate historical and corrected COBRA item-level evaluations.

Only complete three-seed groups are summarized.  The raw per-seed JSON files
remain authoritative; this script makes the recovery comparison reproducible
without copying metrics out of task logs by hand.
"""

import argparse
import json
import os
import statistics
from pathlib import Path


PROTOCOLS = ("leave_one_out", "global_temporal")
SEEDS = (42, 43, 44)
METRICS = ("R@1", "R@5", "R@10", "N@10")


def _read(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _baseline_row(matrix_root: Path, split: str, protocol: str, seed: int):
    path = matrix_root / protocol / "cobra" / split / f"seed{seed}" / "results.json"
    payload = _read(path)
    metrics = payload["metrics"]
    return {
        "seed": seed,
        "path": str(path),
        "metrics": {
            "R@1": metrics["Recall@1"],
            "R@5": metrics["Recall@5"],
            "R@10": metrics["Recall@10"],
            "N@10": metrics["NDCG@10"],
        },
        "source_bundle_sha256": payload.get("source_bundle_sha256"),
    }


def _recovery_row(
    recovery_root: Path, split: str, protocol: str, seed: int, variant: str
):
    path = recovery_root / split / protocol / f"seed{seed}" / f"{variant}_full.json"
    payload = _read(path)
    item = payload["metrics"]["item_level"]
    return {
        "seed": seed,
        "path": str(path),
        "metrics": {metric: item[metric] for metric in METRICS},
        "n_beam": payload["n_beam"],
        "constrain_to_catalog": payload["constrain_to_catalog"],
        "source": payload.get("source"),
    }


def _optimized_row(optimized_root: Path, protocol: str, seed: int):
    path = optimized_root / protocol / f"seed{seed}" / "results.json"
    payload = _read(path)
    metrics = payload["metrics"]
    config = payload["config"]
    return {
        "seed": seed,
        "path": str(path),
        "metrics": {
            "R@1": metrics["Recall@1"],
            "R@5": metrics["Recall@5"],
            "R@10": metrics["Recall@10"],
            "N@10": metrics["NDCG@10"],
        },
        "n_beam": config["eval_n_beam"],
        "constrain_to_catalog": config["constrain_to_catalog"],
        "source": {
            "bundle": payload.get("source_bundle"),
            "bundle_sha256": payload.get("source_bundle_sha256"),
            "git_sha": payload.get("git_sha"),
            **payload.get("execution", {}),
        },
    }


def _summarize(rows):
    summary = {}
    for metric in METRICS:
        values = [row["metrics"][metric] for row in rows]
        summary[metric] = {
            "mean": statistics.fmean(values),
            "population_std": statistics.pstdev(values),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", default="out/matrix")
    parser.add_argument("--recovery-root", default="out/recovery/cobra")
    parser.add_argument("--split", default="beauty")
    parser.add_argument("--variants", nargs="+", default=["fixed", "constrained"])
    parser.add_argument(
        "--include-optimized",
        action="store_true",
        help="also aggregate optimized_light_seq50 trainer results",
    )
    parser.add_argument(
        "--optimized-root",
        help=(
            "trainer result root; defaults to "
            "<recovery-root>/<split>/optimized_light_seq50"
        ),
    )
    parser.add_argument("--output-json")
    args = parser.parse_args()

    matrix_root = Path(args.matrix_root)
    recovery_root = Path(args.recovery_root)
    optimized_root = (
        Path(args.optimized_root)
        if args.optimized_root
        else recovery_root / args.split / "optimized_light_seq50"
    )
    result = {"model": "cobra", "split": args.split, "groups": {}, "missing": []}

    for protocol in PROTOCOLS:
        candidates = ["baseline", *args.variants]
        if args.include_optimized:
            candidates.append("optimized_light_seq50")
        for variant in candidates:
            try:
                if variant == "baseline":
                    rows = [
                        _baseline_row(matrix_root, args.split, protocol, seed)
                        for seed in SEEDS
                    ]
                elif variant == "optimized_light_seq50":
                    rows = [
                        _optimized_row(optimized_root, protocol, seed)
                        for seed in SEEDS
                    ]
                else:
                    rows = [
                        _recovery_row(
                            recovery_root, args.split, protocol, seed, variant
                        )
                        for seed in SEEDS
                    ]
            except FileNotFoundError as exc:
                result["missing"].append(str(exc.filename))
                continue

            key = f"{protocol}:{variant}"
            result["groups"][key] = {
                "protocol": protocol,
                "variant": variant,
                "seeds": list(SEEDS),
                "rows": rows,
                "summary": _summarize(rows),
            }

    print("| Protocol | Variant | R@1 | R@5 | R@10 | N@10 |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for group in result["groups"].values():
        cells = []
        for metric in METRICS:
            stat = group["summary"][metric]
            cells.append(f"{stat['mean']:.4f} ± {stat['population_std']:.4f}")
        print(
            f"| {group['protocol']} | {group['variant']} | "
            + " | ".join(cells)
            + " |"
        )

    if result["missing"]:
        print("\nIncomplete groups; missing files:")
        for path in result["missing"]:
            print(f"- {path}")

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, output)
        print(f"saved {output}")


if __name__ == "__main__":
    main()
