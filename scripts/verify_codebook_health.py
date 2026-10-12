"""
Diagnose codebook usage/collapse for a saved RQ-KMeans or RQ-VAE checkpoint.

The K=8192 collapse story (CLAUDE.md Finding 1) lives or dies on these numbers:
    - usage:        fraction of codes that any item ever uses (lower = collapse)
    - max_freq:     largest fraction of items assigned to one code (higher = collapse)
    - dead_clusters: number of codes with zero items

Usage:
    # RQ-KMeans (default — runs on its source item embeddings)
    python scripts/verify_codebook_health.py \\
        --kind rqkmeans \\
        --checkpoint out/onerec/amazon/beauty/rqkmeans/rqkmeans.pt \\
        --dataset-folder dataset/amazon \\
        --split beauty \\
        --encoder ./models_hub/Qwen3-Embedding-8B

    # JSON output (for run_matrix integrations)
    python scripts/verify_codebook_health.py ... --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np


def per_layer_stats(sem_ids: np.ndarray, codebook_size: int) -> List[Dict[str, float]]:
    n_layers = sem_ids.shape[1]
    n_items = sem_ids.shape[0]
    stats = []
    for layer in range(n_layers):
        counts = np.bincount(sem_ids[:, layer], minlength=codebook_size)
        used = int((counts > 0).sum())
        dead = int(codebook_size - used)
        max_freq = float(counts.max() / n_items) if n_items else 0.0
        top_code = int(counts.argmax())
        # Concentration: smallest set of codes holding 50% of items.
        sorted_counts = np.sort(counts)[::-1]
        cum = np.cumsum(sorted_counts)
        half = int(np.searchsorted(cum, n_items * 0.5) + 1) if n_items else 0
        stats.append({
            "layer": layer,
            "usage": used / codebook_size,
            "used_codes": used,
            "dead_codes": dead,
            "max_freq": max_freq,
            "top_code": top_code,
            "codes_for_50pct_items": half,
        })
    return stats


def collision_stats(sem_ids: np.ndarray) -> Dict[str, float]:
    n = len(sem_ids)
    keys = [tuple(row) for row in sem_ids.tolist()]
    unique = len(set(keys))
    from collections import Counter
    counter = Counter(keys)
    return {
        "n_items": n,
        "unique_sids": unique,
        "collision_rate": (n - unique) / n if n else 0.0,
        "max_collision_group": max(counter.values()) if counter else 0,
    }


def load_rqkmeans(checkpoint: str):
    from genrec.models.rqkmeans import RqKmeans
    return RqKmeans.load(checkpoint)


def load_embeddings_for_rqkmeans(dataset_folder: str, split: str, encoder: str) -> np.ndarray:
    from genrec.data.amazon import AmazonItemDataset
    ds = AmazonItemDataset(
        root=dataset_folder,
        split=split,
        train_test_split="all",
        encoder_model_name=encoder,
    )
    return ds.embeddings.astype(np.float32)


def diagnose_rqkmeans(checkpoint: str, dataset_folder: str, split: str,
                      encoder: str) -> Dict:
    model = load_rqkmeans(checkpoint)
    embeddings = load_embeddings_for_rqkmeans(dataset_folder, split, encoder)
    sem_ids = model.assign(embeddings).sem_ids  # [N, n_layers]
    return {
        "kind": "rqkmeans",
        "checkpoint": os.path.abspath(checkpoint),
        "codebook_size": model.codebook_size,
        "n_layers": model.n_layers,
        "per_layer": per_layer_stats(sem_ids, model.codebook_size),
        "collisions": collision_stats(sem_ids),
    }


def diagnose_rqvae(checkpoint: str, dataset_folder: str, split: str,
                   encoder: str, item_text_format: str = "legacy") -> Dict:
    # RQ-VAE pathway: load model and re-encode item embeddings.
    import torch
    from genrec.models.rqvae import RqVae
    from genrec.data.amazon import AmazonItemDataset

    model = RqVae.from_pretrained(checkpoint, map_location="cpu")
    model.eval()

    # Use the same item universe and embedding cache as rqvae_trainer.  The
    # legacy P5 dataset has different preprocessing and item IDs, so using it
    # here can report health for a tokenizer input that was never trained.
    ds = AmazonItemDataset(
        root=dataset_folder,
        split=split,
        train_test_split="all",
        encoder_model_name=encoder,
        item_text_format=item_text_format,
    )
    embeddings = torch.as_tensor(ds.embeddings, dtype=torch.float32)
    with torch.no_grad():
        out = model.get_semantic_ids(embeddings)
    sem_ids = (out.sem_ids if hasattr(out, "sem_ids") else out).cpu().numpy()
    if sem_ids.ndim == 1:
        sem_ids = sem_ids.reshape(-1, 1)
    codebook_size = int(getattr(model, "codebook_size", sem_ids.max() + 1))
    return {
        "kind": "rqvae",
        "checkpoint": os.path.abspath(checkpoint),
        "codebook_size": codebook_size,
        "n_layers": sem_ids.shape[1],
        "per_layer": per_layer_stats(sem_ids, codebook_size),
        "collisions": collision_stats(sem_ids),
    }


def print_report(report: Dict) -> None:
    print(f"=== Codebook Health ({report['kind']}) ===")
    print(f"checkpoint:    {report['checkpoint']}")
    print(f"codebook_size: {report['codebook_size']}")
    print(f"n_layers:      {report['n_layers']}")
    c = report["collisions"]
    print(f"n_items:       {c['n_items']}")
    print(f"unique_sids:   {c['unique_sids']}  "
          f"(collision_rate={c['collision_rate']:.4%}, "
          f"max_group={c['max_collision_group']})")
    print()
    header = f"{'layer':>5} {'usage':>8} {'dead':>6} {'max_freq':>10} {'top_code':>10} {'50%_codes':>10}"
    print(header)
    print("-" * len(header))
    for s in report["per_layer"]:
        print(f"{s['layer']:>5} "
              f"{s['usage']:>8.2%} "
              f"{s['dead_codes']:>6} "
              f"{s['max_freq']:>10.4%} "
              f"{s['top_code']:>10} "
              f"{s['codes_for_50pct_items']:>10}")

    # Highlight known collapse signature.
    bad = [s for s in report["per_layer"]
           if s["usage"] < 0.8 or s["max_freq"] > 0.1]
    if bad:
        print()
        print("WARNING: layer(s) show collapse signature "
              "(usage<80% or max_freq>10%). See CLAUDE.md Finding 1.")
        for s in bad:
            print(f"  layer {s['layer']}: usage={s['usage']:.2%}, "
                  f"max_freq={s['max_freq']:.2%}, dead={s['dead_codes']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["rqkmeans", "rqvae"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-folder", default="dataset/amazon")
    ap.add_argument("--split", default="beauty")
    ap.add_argument("--encoder", default=None,
                    help="Encoder model dir. Default: Qwen3-Embedding-8B for rqkmeans, "
                         "sentence-t5-xl for rqvae.")
    ap.add_argument(
        "--item-text-format",
        choices=("legacy", "tiger_paper"),
        default="legacy",
        help="RQ-VAE embedding cache/template to inspect.",
    )
    ap.add_argument("--json", action="store_true", help="JSON output instead of human-readable")
    args = ap.parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    encoder = args.encoder or (
        "./models_hub/Qwen3-Embedding-8B" if args.kind == "rqkmeans"
        else "./models_hub/sentence-t5-xl"
    )

    if args.kind == "rqkmeans":
        report = diagnose_rqkmeans(args.checkpoint, args.dataset_folder, args.split, encoder)
    else:
        report = diagnose_rqvae(
            args.checkpoint,
            args.dataset_folder,
            args.split,
            encoder,
            item_text_format=args.item_text_format,
        )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
