"""
Run the reproducibility matrix: model x split x seed.

Each cell shells out to the matching trainer with `--split`, seed, and the
selected evaluation protocol. Skips cells that already have a results.json
(unless --force). Default runs serially because each cell holds the GPU.

Example:
    python scripts/run_matrix.py \
        --models sasrec hstu \
        --splits beauty sports toys \
        --seeds 42 43 44 \
        --dry-run

Custom save_dir layout (include ``{protocol}`` to keep LOO and temporal runs
separate):
    python scripts/run_matrix.py --models sasrec --splits beauty --seeds 42 \
        --save-dir-root "out/{protocol}/{model}/amazon/{split}/seed{seed}"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

# (model_key, trainer_module, default_config_path)
# trainer_module is what `python -m <module>` invokes.
# config path supports {split} substitution at runtime (consumed by parse_config).
REGISTRY = {
    "sasrec":      ("genrec.trainers.sasrec_trainer",      "config/sasrec/amazon.gin"),
    "hstu":        ("genrec.trainers.hstu_trainer",        "config/hstu/amazon.gin"),
    # Tokenizer stages are deliberately absent: TIGER/LCRec/COBRA use
    # incompatible RQ-VAE geometries, and OneRec uses RQ-KMeans. Prepare those
    # artifacts with their explicit pipeline commands before running the
    # recommender matrix so save-path overrides cannot disconnect dependencies.
    "tiger":       ("genrec.trainers.tiger_trainer",       "config/tiger/amazon/tiger.gin"),
    "lcrec":       ("genrec.trainers.lcrec_trainer",       "config/lcrec/amazon/lcrec.gin"),
    "cobra":       ("genrec.trainers.cobra_trainer",       "config/cobra/amazon/cobra.gin"),
    "rpg":         ("genrec.trainers.rpg_trainer",         "config/rpg/amazon.gin"),
    "notellm":     ("genrec.trainers.notellm_trainer",     "config/notellm/amazon.gin"),
    "onerec_sft":  ("genrec.trainers.onerec_sft_trainer",  "config/onerec/amazon/sft.gin"),
    "onerec_grpo": ("genrec.trainers.onerec_grpo_trainer", "config/onerec/amazon/grpo.gin"),
    "onerec_clm":  ("genrec.trainers.onerec_clm_trainer",  "config/onerec/amazon/clm.gin"),
}

# RPG publishes dataset-specific beam widths and optimization settings.  Keep
# this routing explicit so an automated Sports/Toys matrix cannot silently run
# with the Beauty recipe from ``amazon.gin``.
RPG_SPLIT_CONFIGS = {
    "beauty": "config/rpg/amazon.gin",
    "sports": "config/rpg/amazon_sports.gin",
    "toys": "config/rpg/amazon_toys.gin",
}

DATASET_CONFIGURABLES = {
    "sasrec": ["AmazonSASRecDataset"],
    "hstu": ["AmazonHSTUDataset"],
    "tiger": ["AmazonSeqDataset"],
    "lcrec": ["AmazonLCRecDataset"],
    "cobra": ["AmazonCobraDataset"],
    "rpg": ["AmazonRPGDataset"],
    "notellm": ["AmazonNoteLLMDataset"],
    "onerec_sft": ["AmazonOneRecSFTDataset"],
    "onerec_grpo": ["AmazonOneRecRLDataset", "AmazonOneRecSFTDataset"],
    "onerec_clm": ["AmazonOneRecCLMDataset"],
}

# Trainers whose top-level gin function is not `train`.
TRAIN_FN = {
    "onerec_sft":  "train_sft",
    "onerec_grpo": "train_grpo",
    "onerec_clm":  "train_clm",
    "rpg":         "train_rpg",
}


@dataclass
class Cell:
    model: str
    split: str
    seed: int
    config_path: str
    save_dir: str
    evaluation_protocol: str = "leave_one_out"
    sft_checkpoint_path: Optional[str] = None

    def trainer_module(self) -> str:
        return REGISTRY[self.model][0]

    def gin_overrides(self) -> List[str]:
        fn = TRAIN_FN.get(self.model, "train")
        overrides = [
            f"{fn}.seed={self.seed}",
            f'{fn}.save_dir_root="{self.save_dir}"',
        ]
        overrides.extend(
            f'{dataset_name}.evaluation_protocol="{self.evaluation_protocol}"'
            for dataset_name in DATASET_CONFIGURABLES.get(self.model, [])
        )
        if self.model == "onerec_grpo":
            if not self.sft_checkpoint_path:
                raise ValueError(
                    "OneRec GRPO matrix cells require the matching SFT checkpoint"
                )
            overrides.append(
                f'train_grpo.sft_checkpoint_path="{self.sft_checkpoint_path}"'
            )
        return overrides

    def missing_dependencies(self) -> List[str]:
        """Return required artifacts that are absent for this cell."""
        if self.model == "onerec_grpo" and self.sft_checkpoint_path:
            if not os.path.isdir(self.sft_checkpoint_path):
                return [self.sft_checkpoint_path]
        return []

    def cmd(self) -> List[str]:
        argv = [sys.executable, "-m", self.trainer_module(), self.config_path,
                "--split", self.split]
        for g in self.gin_overrides():
            argv += ["--gin", g]
        return argv


def expand_save_dir(template: str, model: str, split: str, seed: int,
                    evaluation_protocol: str) -> str:
    return (template
            .replace("{model}", model)
            .replace("{split}", split)
            .replace("{seed}", str(seed))
            .replace("{protocol}", evaluation_protocol))


def cell_done(save_dir: str) -> bool:
    return os.path.isfile(os.path.join(save_dir, "results.json"))


def build_cells(models: Iterable[str], splits: Iterable[str], seeds: Iterable[int],
                save_dir_template: str, config_overrides: dict,
                evaluation_protocol: str = "leave_one_out") -> List[Cell]:
    cells = []
    for model in models:
        if model not in REGISTRY:
            raise SystemExit(f"unknown model: {model!r} (known: {sorted(REGISTRY)})")
        for split in splits:
            config_path = config_overrides.get(model)
            if config_path is None and model == "rpg":
                config_path = RPG_SPLIT_CONFIGS.get(split, REGISTRY[model][1])
            if config_path is None:
                config_path = REGISTRY[model][1]
            for seed in seeds:
                save_dir = expand_save_dir(
                    save_dir_template, model, split, seed, evaluation_protocol
                )
                sft_checkpoint_path = None
                if model == "onerec_grpo":
                    sft_save_dir = expand_save_dir(
                        save_dir_template,
                        "onerec_sft",
                        split,
                        seed,
                        evaluation_protocol,
                    )
                    sft_checkpoint_path = os.path.join(
                        sft_save_dir, "checkpoint_best"
                    )
                cells.append(Cell(model=model, split=split, seed=seed,
                                  config_path=config_path, save_dir=save_dir,
                                  evaluation_protocol=evaluation_protocol,
                                  sft_checkpoint_path=sft_checkpoint_path))
    return cells


def run_cell(cell: Cell, dry_run: bool, log_file: Optional[str]) -> int:
    print(f"\n[run_matrix] {cell.model} split={cell.split} seed={cell.seed}")
    print("           ", " ".join(cell.cmd()))
    if dry_run:
        return 0
    missing = cell.missing_dependencies()
    if missing:
        print(
            "[run_matrix] missing dependency artifact(s): "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 2
    os.makedirs(cell.save_dir, exist_ok=True)
    stdout = open(log_file, "ab") if log_file else None
    try:
        return subprocess.call(cell.cmd(), stdout=stdout, stderr=subprocess.STDOUT)
    finally:
        if stdout:
            stdout.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help=f"Model keys, one or more from: {sorted(REGISTRY)}")
    ap.add_argument("--splits", nargs="+", required=True,
                    help="Dataset splits (beauty, sports, toys, clothing, ...)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                    help="Seeds to run for each (model, split)")
    ap.add_argument("--save-dir-root",
                    default="out/matrix/{protocol}/{model}/{split}/seed{seed}",
                    help="Save dir template. Supports {protocol}/{model}/{split}/{seed}.")
    ap.add_argument(
        "--evaluation-protocol",
        choices=["leave_one_out", "global_temporal"],
        default="leave_one_out",
        help="Sequence split protocol for recommender models.",
    )
    ap.add_argument("--config", action="append", default=[],
                    help='Per-model config override, e.g. --config "onerec_sft=config/.../sft_k256_v2.gin"')
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")
    ap.add_argument("--force", action="store_true",
                    help="Re-run cells even if results.json exists")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Continue with remaining cells after a failure")
    ap.add_argument("--log-dir", default=None,
                    help="If set, write each cell's stdout/stderr to this dir")
    args = ap.parse_args()

    config_overrides = {}
    for entry in args.config:
        if "=" not in entry:
            raise SystemExit(f"--config expects KEY=PATH, got: {entry!r}")
        key, path = entry.split("=", 1)
        config_overrides[key.strip()] = path.strip()

    cells = build_cells(args.models, args.splits, args.seeds,
                        args.save_dir_root, config_overrides,
                        args.evaluation_protocol)
    print(f"[run_matrix] planning {len(cells)} cells "
          f"({len(args.models)} models × {len(args.splits)} splits × {len(args.seeds)} seeds)")

    failures, skipped = [], 0
    for cell in cells:
        if not args.force and cell_done(cell.save_dir):
            print(f"[run_matrix] SKIP (results.json exists): {cell.save_dir}")
            skipped += 1
            continue
        log_file = None
        if args.log_dir:
            os.makedirs(args.log_dir, exist_ok=True)
            log_file = os.path.join(
                args.log_dir,
                f"{cell.model}__{cell.split}__seed{cell.seed}.log")
        t0 = time.time()
        rc = run_cell(cell, args.dry_run, log_file)
        dt = time.time() - t0
        if rc != 0:
            failures.append((cell, rc))
            print(f"[run_matrix] FAIL rc={rc} ({dt/60:.1f}min): "
                  f"{cell.model} {cell.split} seed={cell.seed}")
            if not args.continue_on_error:
                break
        else:
            print(f"[run_matrix] OK ({dt/60:.1f}min): "
                  f"{cell.model} {cell.split} seed={cell.seed}")

    print(f"\n[run_matrix] {len(cells) - skipped - len(failures)}/{len(cells)} "
          f"completed, {skipped} skipped, {len(failures)} failed.")
    if failures:
        for cell, rc in failures:
            print(f"  FAIL rc={rc}: {cell.model}/{cell.split}/seed={cell.seed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
