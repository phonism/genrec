"""
Parse OneRec SFT training logs, pick the best-valid epoch, and emit the matching
test metrics. Used to populate README rows before save_run_results was wired in.

Usage:
    python scripts/extract_sft_metrics.py path/to/train_*.log
    python scripts/extract_sft_metrics.py 'out/onerec/amazon/*/sft_k256_v2_*/train_*.log' --json

Log format expected (one section per evaluation epoch):

    Epoch N - seqrec ...           # valid block precedes
    Epoch N - seqrec TopK: Recall@1=X, NDCG@1=X, Recall@5=X, NDCG@5=X, Recall@10=X, NDCG@10=X
    --- Test ---
    Epoch N - seqrec ...
    Epoch N - seqrec TopK: Recall@1=X, ..., NDCG@10=X
    ...
    New best Recall@10: X  (or "No improvement.")
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional

EPOCH_TOPK_RE = re.compile(
    r"Epoch\s+(\d+)\s*-\s*seqrec TopK:\s*(.+?)$",
    re.MULTILINE,
)
SECTION_RE = re.compile(r"---\s*(Valid|Test)(?:\s*\([^)]*\))?\s*---")
BEST_RE = re.compile(r"New best Recall@10:\s*([0-9.]+)")
METRIC_RE = re.compile(r"(Recall@\d+|NDCG@\d+)=([0-9.]+)")


def parse_log(path: str) -> Dict:
    """Walk the log line by line, pairing each Valid/Test block with its epoch."""
    epochs: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(lambda: {"valid": {}, "test": {}})
    section: Optional[str] = None
    best_valid_epoch: Optional[int] = None
    best_valid_r10 = -1.0

    with open(path) as f:
        for line in f:
            sm = SECTION_RE.search(line)
            if sm:
                section = sm.group(1).lower()
                continue

            tm = EPOCH_TOPK_RE.search(line)
            if tm and section:
                epoch = int(tm.group(1))
                metrics = {k: float(v) for k, v in METRIC_RE.findall(tm.group(2))}
                epochs[epoch][section] = metrics
                # New section can only follow another "--- X ---" line; clear so we
                # don't reuse the same section for a later epoch on the same rank.
                # (Multi-rank logs interleave; we deliberately keep the last value
                # written for each (epoch, section).)
                continue

            bm = BEST_RE.search(line)
            if bm:
                r10 = float(bm.group(1))
                # The "New best Recall@10" line follows the latest Valid block.
                # Find the most-recent epoch that has a valid Recall@10 == r10.
                for ep, blocks in epochs.items():
                    v = blocks.get("valid", {}).get("Recall@10")
                    if v is not None and abs(v - r10) < 1e-6 and r10 > best_valid_r10:
                        best_valid_r10 = r10
                        best_valid_epoch = ep

    return {
        "path": path,
        "epochs": dict(epochs),
        "best_valid_epoch": best_valid_epoch,
        "best_valid_recall@10": best_valid_r10 if best_valid_epoch is not None else None,
        "best_test_metrics": (
            epochs[best_valid_epoch]["test"] if best_valid_epoch is not None else {}
        ),
    }


def render_md(reports: List[Dict]) -> str:
    metrics_order = ["Recall@1", "Recall@5", "Recall@10", "NDCG@5", "NDCG@10"]
    header = ["log", "best_epoch", "valid_R@10"] + [f"test_{m}" for m in metrics_order]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for r in reports:
        if r["best_valid_epoch"] is None:
            row = [r["path"], "-", "-"] + ["-"] * len(metrics_order)
        else:
            tm = r["best_test_metrics"]
            row = [
                r["path"],
                str(r["best_valid_epoch"]),
                f"{r['best_valid_recall@10']:.4f}",
            ] + [(f"{tm[m]:.4f}" if m in tm else "-") for m in metrics_order]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="Log files (globs allowed)")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")
    args = ap.parse_args()

    paths: List[str] = []
    for pat in args.logs:
        matches = sorted(glob.glob(pat))
        paths.extend(matches or [pat])

    reports = [parse_log(p) for p in paths]
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        print(render_md(reports))


if __name__ == "__main__":
    main()
