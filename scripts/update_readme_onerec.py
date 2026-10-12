"""
Update the four OneRec-SFT rows in README.md from train logs.

Usage:
    python scripts/update_readme_onerec.py \\
        --beauty path/to/beauty/train_*.log \\
        --sports path/to/sports/train_*.log \\
        --toys   path/to/toys/train_*.log \\
        --home   path/to/home/train_*.log \\
        [--label "OneRec-SFT (1.7B, v2)"]

For each split, picks the log with the highest best-valid R@10 (handles
multi-rank logs that share a save dir but only one has the section markers),
then rewrites the bold "OneRec-SFT (1.7B)" row in the matching README table.
Other models' rows are left untouched. Use --dry-run to preview the diff.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from extract_sft_metrics import parse_log  # noqa: E402

README_PATH = REPO_ROOT / "README.md"
ROW_REGEX = re.compile(
    r"^\| \*\*OneRec-SFT \(1\.7B\)\*\* \|.*\|$",
    re.MULTILINE,
)
# Split section headers in the README. Order matters: tables appear in this order.
SECTIONS = [
    ("beauty", "### Amazon 2014 Beauty"),
    ("sports", "### Amazon 2014 Sports"),
    ("toys",   "### Amazon 2014 Toys"),
    ("home",   "### Amazon 2014 Home"),
]


def best_log(paths: List[str]) -> Optional[Dict]:
    candidates = []
    for p in sorted(paths):
        r = parse_log(p)
        if r["best_valid_epoch"] is not None:
            candidates.append(r)
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["best_valid_recall@10"])


def format_row(label: str, metrics: Dict[str, float]) -> str:
    def fmt(k):
        v = metrics.get(k)
        return f"**{v:.4f}**" if isinstance(v, (int, float)) else "—"
    return (
        f"| **{label}** "
        f"| {fmt('Recall@5')} "
        f"| {fmt('Recall@10')} "
        f"| {fmt('NDCG@5')} "
        f"| {fmt('NDCG@10')} |"
    )


def replace_row_in_section(text: str, section_header: str, new_row: str) -> Tuple[str, bool]:
    """Replace the OneRec-SFT row in the table under section_header."""
    start = text.find(section_header)
    if start == -1:
        return text, False
    # The next section_header (or end of file) bounds this section.
    next_idx = len(text)
    for _, hdr in SECTIONS:
        idx = text.find(hdr, start + len(section_header))
        if idx != -1 and idx < next_idx:
            next_idx = idx
    chunk = text[start:next_idx]
    new_chunk, n = ROW_REGEX.subn(new_row, chunk, count=1)
    if n == 0:
        return text, False
    return text[:start] + new_chunk + text[next_idx:], True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for split, _ in SECTIONS:
        ap.add_argument(f"--{split}", default=None,
                        help=f"Glob for {split} train logs")
    ap.add_argument("--label", default="OneRec-SFT (1.7B)",
                    help="Row label in the README (kept bold)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the new rows + which sections would change, don't write")
    args = ap.parse_args()

    readme = README_PATH.read_text()
    changes = []
    for split, header in SECTIONS:
        glob_pat = getattr(args, split)
        if not glob_pat:
            continue
        paths = sorted(glob.glob(glob_pat))
        if not paths:
            print(f"[{split}] no logs match {glob_pat!r}", file=sys.stderr)
            continue
        best = best_log(paths)
        if best is None:
            print(f"[{split}] none of {len(paths)} logs has a best-valid epoch yet",
                  file=sys.stderr)
            continue
        row = format_row(args.label, best["best_test_metrics"])
        print(f"[{split}] best_ep={best['best_valid_epoch']} "
              f"valid_R@10={best['best_valid_recall@10']:.4f}  ->  {row}")
        new_readme, ok = replace_row_in_section(readme, header, row)
        if not ok:
            print(f"  WARN: did not find row to replace under {header!r}",
                  file=sys.stderr)
            continue
        readme = new_readme
        changes.append(split)

    if not changes:
        print("\nNo changes applied.", file=sys.stderr)
        return

    if args.dry_run:
        print(f"\n[dry-run] would rewrite {README_PATH} for splits: {changes}")
        return

    README_PATH.write_text(readme)
    print(f"\nWrote {README_PATH} (updated splits: {changes})")


if __name__ == "__main__":
    main()
