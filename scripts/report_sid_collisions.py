"""
Report Semantic-ID (SID) collision metrics for one or more SID sources.

Why this exists
---------------
Generative recommenders map every item to a tuple of codes (the SID). When two
distinct items receive the *same* SID, trie/beam decoding cannot tell them apart,
so a single beam hit is credited to several items and Recall@K is *inflated*. This
is a generative-recommendation-specific source of metric inflation that is almost
never reported. This tool makes the collision rate a first-class, training-free
report item across ALL SID models (TIGER / LC-Rec / COBRA / OneRec / RPG), so the
honest "how much of the Recall is real" question can be answered.

It is intentionally standalone (pure NumPy, no genrec import) and read-only: it
consumes a precomputed ``[N, n_layers]`` SID array and never touches model or
trainer code. To produce that array from a tokenizer checkpoint, see
``scripts/verify_codebook_health.py`` (which already dumps per-checkpoint stats);
this tool adds per-prefix collision accumulation and cross-source comparison.

What it adds over verify_codebook_health.collision_stats
--------------------------------------------------------
- per-prefix collision: collision_rate after the first d layers (d = 1..L), which
  shows *where* collisions resolve (e.g. TIGER's 4th disambiguation layer).
- capacity utilisation: unique_sids / codebook_size**n_layers.
- collision group-size histogram (benign redundancy vs harmful collision; QuaSID).
- aggregate mode: many labelled sources -> one Markdown comparison table.

Input formats (auto-detected by extension)
------------------------------------------
- ``.npy``  : an int array of shape [N, n_layers].
- ``.json`` : a list of lists ``[[c0,c1,c2], ...]`` or a dict ``{item_id: [...]}``.
- ``.tsv`` / ``.txt`` / ``.csv`` : one item per line; either whitespace/comma
  separated ints, OR token strings like ``<C0_5><C1_6><C2_2>`` (parsed with
  ``--token-regex``, default ``<C\\d+_(\\d+)>``).

Usage
-----
    # single source, human-readable
    python scripts/report_sid_collisions.py --sem-ids out/tiger/beauty/sem_ids.npy

    # compare several models, Markdown table for the docs / main results table
    python scripts/report_sid_collisions.py --markdown \\
        --sem-ids "TIGER-3code=out/tiger/beauty/sid_3.npy" \\
        --sem-ids "TIGER-4code=out/tiger/beauty/sid_4.npy" \\
        --sem-ids "OneRec-K256=out/onerec/beauty/sid.json"

    # JSON for aggregate_results.py
    python scripts/report_sid_collisions.py --json --sem-ids beauty=sid.npy

    # validate the metric math without any data
    python scripts/report_sid_collisions.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_TOKEN_REGEX = r"<C\d+_(\d+)>"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _rows_to_array(rows: List[List[int]]) -> np.ndarray:
    if not rows:
        raise ValueError("no SID rows parsed")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"inconsistent SID length across items: {sorted(widths)}")
    return np.asarray(rows, dtype=np.int64)


def load_sem_ids(path: str, token_regex: str = DEFAULT_TOKEN_REGEX) -> np.ndarray:
    """Load an [N, n_layers] int SID array from .npy / .json / text."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr.astype(np.int64)

    if ext == ".json":
        with open(path) as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            # {item_id: [codes]} — order by integer key when possible.
            try:
                items = sorted(obj.items(), key=lambda kv: int(kv[0]))
            except (TypeError, ValueError):
                items = list(obj.items())
            rows = [list(map(int, v)) for _, v in items]
        else:
            rows = [list(map(int, v)) for v in obj]
        return _rows_to_array(rows)

    # text (.tsv/.txt/.csv or anything else): one item per line
    pat = re.compile(token_regex) if token_regex else None
    rows: List[List[int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            codes: List[int] = []
            if pat is not None:
                codes = [int(m) for m in pat.findall(line)]
            if not codes:
                codes = [int(tok) for tok in re.split(r"[\s,\t]+", line) if tok]
            rows.append(codes)
    return _rows_to_array(rows)


# --------------------------------------------------------------------------- #
# Metrics (pure, testable)
# --------------------------------------------------------------------------- #
def prefix_collision_stats(sem_ids: np.ndarray) -> List[Dict[str, float]]:
    """For each depth d = 1..n_layers, collision after using only the first d codes."""
    n, n_layers = sem_ids.shape
    out = []
    for d in range(1, n_layers + 1):
        keys = {tuple(row) for row in sem_ids[:, :d].tolist()}
        unique = len(keys)
        out.append({
            "depth": d,
            "unique_prefixes": unique,
            "collision_rate": (n - unique) / n if n else 0.0,
        })
    return out


def collision_report(sem_ids: np.ndarray,
                     codebook_size: Optional[int] = None) -> Dict:
    """Full collision report for an [N, n_layers] SID array."""
    sem_ids = np.asarray(sem_ids, dtype=np.int64)
    if sem_ids.ndim == 1:
        sem_ids = sem_ids.reshape(-1, 1)
    n, n_layers = sem_ids.shape

    keys = [tuple(row) for row in sem_ids.tolist()]
    counter = Counter(keys)
    unique = len(counter)
    # group-size histogram: {group_size: how many SIDs have that many items}
    size_hist = Counter(counter.values())
    items_in_collisions = sum(sz for sz in counter.values() if sz > 1)

    if codebook_size is None:
        codebook_size = int(sem_ids.max()) + 1 if sem_ids.size else 0
    capacity = float(codebook_size) ** n_layers if codebook_size else 0.0
    capacity_util = unique / capacity if capacity else 0.0

    return {
        "n_items": n,
        "n_layers": n_layers,
        "codebook_size": codebook_size,
        "unique_sids": unique,
        "collision_rate": (n - unique) / n if n else 0.0,
        "max_collision_group": max(counter.values()) if counter else 0,
        "items_in_collisions": items_in_collisions,
        "frac_items_in_collisions": items_in_collisions / n if n else 0.0,
        "group_size_histogram": dict(sorted(size_hist.items())),
        "capacity_utilization": capacity_util,
        "per_prefix": prefix_collision_stats(sem_ids),
    }


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def format_report(label: str, r: Dict) -> str:
    lines = [f"=== SID collisions: {label} ==="]
    lines.append(f"n_items={r['n_items']}  n_layers={r['n_layers']}  "
                 f"codebook_size={r['codebook_size']}")
    lines.append(f"unique_sids={r['unique_sids']}  "
                 f"collision_rate={r['collision_rate']:.4%}  "
                 f"max_group={r['max_collision_group']}")
    lines.append(f"items_in_collisions={r['items_in_collisions']} "
                 f"({r['frac_items_in_collisions']:.4%})  "
                 f"capacity_util={r['capacity_utilization']:.3e}")
    lines.append("per-prefix collision (where collisions resolve):")
    for p in r["per_prefix"]:
        lines.append(f"  depth {p['depth']}: collision_rate={p['collision_rate']:.4%} "
                     f"(unique={p['unique_prefixes']})")
    hist = r["group_size_histogram"]
    extra = {k: v for k, v in hist.items() if k > 1}
    if extra:
        lines.append("collision group-size histogram (size: #SIDs): "
                     + ", ".join(f"{k}:{v}" for k, v in extra.items()))
    return "\n".join(lines)


def comparison_table(named_reports: List[Tuple[str, Dict]]) -> str:
    """Markdown comparison table, one row per source."""
    head = ("| source | n_items | n_layers | unique_sids | collision_rate | "
            "max_group | frac_items_collided |")
    sep = "|---|---|---|---|---|---|---|"
    rows = [head, sep]
    for label, r in named_reports:
        rows.append(
            f"| {label} | {r['n_items']} | {r['n_layers']} | {r['unique_sids']} | "
            f"{r['collision_rate']:.4%} | {r['max_collision_group']} | "
            f"{r['frac_items_in_collisions']:.4%} |"
        )
    # second table: per-prefix accumulation (only if depths are comparable)
    max_depth = max((r["n_layers"] for _, r in named_reports), default=0)
    if max_depth > 1:
        rows.append("")
        head2 = "| source | " + " | ".join(f"@{d}" for d in range(1, max_depth + 1)) + " |"
        sep2 = "|---|" + "|".join(["---"] * max_depth) + "|"
        rows.append("per-prefix collision_rate (after first d codes):")
        rows.append(head2)
        rows.append(sep2)
        for label, r in named_reports:
            cells = []
            by_depth = {p["depth"]: p["collision_rate"] for p in r["per_prefix"]}
            for d in range(1, max_depth + 1):
                cells.append(f"{by_depth[d]:.2%}" if d in by_depth else "—")
            rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Self-test (no data needed)
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # 10 items, 8 unique full SIDs -> 2 collisions (matches verify_codebook_health).
    sem = np.array([[0, 0], [0, 1], [0, 1], [1, 0], [1, 1],
                    [2, 0], [2, 1], [3, 0], [3, 1], [3, 1]])
    r = collision_report(sem, codebook_size=4)
    assert r["unique_sids"] == 8, r
    assert abs(r["collision_rate"] - 0.2) < 1e-9, r
    assert r["max_collision_group"] == 2, r
    assert r["items_in_collisions"] == 4, r           # two groups of size 2
    # prefix @1: prefixes {0,1,2,3} -> 4 unique of 10 -> 0.6 collision
    assert abs(r["per_prefix"][0]["collision_rate"] - 0.6) < 1e-9, r
    assert abs(r["per_prefix"][1]["collision_rate"] - 0.2) < 1e-9, r
    assert abs(r["capacity_utilization"] - 8 / 16) < 1e-9, r
    print("selftest OK")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_source(spec: str) -> Tuple[str, str]:
    """'label=path' -> (label, path); bare 'path' -> (basename, path)."""
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), path.strip()
    return os.path.splitext(os.path.basename(spec))[0], spec


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sem-ids", action="append", default=[], metavar="[LABEL=]PATH",
                    help="SID source; repeat for cross-source comparison.")
    ap.add_argument("--codebook-size", type=int, default=None,
                    help="Codebook size per layer (for capacity util). Default: max+1.")
    ap.add_argument("--token-regex", default=DEFAULT_TOKEN_REGEX,
                    help=f"Regex to parse token-string lines. Default: {DEFAULT_TOKEN_REGEX!r}")
    ap.add_argument("--markdown", action="store_true", help="Emit Markdown comparison table.")
    ap.add_argument("--json", action="store_true", help="Emit JSON.")
    ap.add_argument("--selftest", action="store_true", help="Validate metric math and exit.")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.sem_ids:
        ap.error("provide at least one --sem-ids [LABEL=]PATH (or --selftest)")

    named_reports: List[Tuple[str, Dict]] = []
    for spec in args.sem_ids:
        label, path = _parse_source(spec)
        if not os.path.isfile(path):
            sys.exit(f"SID file not found: {path}")
        sem_ids = load_sem_ids(path, token_regex=args.token_regex)
        named_reports.append((label, collision_report(sem_ids, args.codebook_size)))

    if args.json:
        print(json.dumps({label: r for label, r in named_reports}, indent=2))
    elif args.markdown:
        print(comparison_table(named_reports))
    else:
        print("\n\n".join(format_report(label, r) for label, r in named_reports))


if __name__ == "__main__":
    main()
