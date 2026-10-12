"""
Dump an item -> Semantic-ID array to .npy from a trained tokenizer.

Reads item embeddings from the cached parquet (no text encoder needed, so it
works even after models_hub/ is wiped) and applies a trained RQ-KMeans or RQ-VAE
tokenizer to produce an ``[N, n_layers]`` SID array. The array feeds
``scripts/report_sid_collisions.py`` for collision reporting.

Read-only w.r.t. existing code: this is a thin loader that reuses
``genrec.models.{rqkmeans,rqvae}`` exactly as ``verify_codebook_health.py`` does.

Usage
-----
    # RQ-KMeans (OneRec tokenizer)
    python scripts/dump_sem_ids.py --kind rqkmeans \\
        --checkpoint out/onerec/amazon/beauty/rqkmeans_k256/rqkmeans.pt \\
        --parquet dataset/amazon/processed/beauty/item_emb_Qwen3-Embedding-8B.parquet \\
        --out out/_sid_dumps/onerec_beauty_k256.npy

    # RQ-VAE (TIGER tokenizer)
    python scripts/dump_sem_ids.py --kind rqvae \\
        --checkpoint out/rqvae/amazon/beauty/checkpoint.pt \\
        --parquet dataset/amazon/processed/beauty/item_emb_sentence-t5-xl.parquet \\
        --out out/_sid_dumps/tiger_beauty.npy
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def load_embeddings(parquet_path: str) -> np.ndarray:
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    emb = np.stack(df["embedding"].values, axis=0).astype(np.float32)
    return emb


def sem_ids_rqkmeans(checkpoint: str, embeddings: np.ndarray) -> np.ndarray:
    from genrec.models.rqkmeans import RqKmeans
    model = RqKmeans.load(checkpoint)
    return np.asarray(model.assign(embeddings).sem_ids, dtype=np.int64)


def sem_ids_rqvae(checkpoint: str, embeddings: np.ndarray) -> np.ndarray:
    import torch
    from genrec.models.rqvae import RqVae
    model = RqVae.from_pretrained(checkpoint, map_location="cpu")
    model.eval()
    with torch.no_grad():
        out = model.get_semantic_ids(torch.as_tensor(embeddings, dtype=torch.float32))
    sem = (out.sem_ids if hasattr(out, "sem_ids") else out).cpu().numpy()
    if sem.ndim == 1:
        sem = sem.reshape(-1, 1)
    return sem.astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["rqkmeans", "rqvae"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--parquet", required=True, help="Cached item_emb parquet path.")
    ap.add_argument("--out", required=True, help="Output .npy path.")
    args = ap.parse_args()

    for p in (args.checkpoint, args.parquet):
        if not os.path.isfile(p):
            sys.exit(f"not found: {p}")

    emb = load_embeddings(args.parquet)
    if args.kind == "rqkmeans":
        sem = sem_ids_rqkmeans(args.checkpoint, emb)
    else:
        sem = sem_ids_rqvae(args.checkpoint, emb)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, sem)
    print(f"wrote {sem.shape} SID array -> {args.out}  "
          f"(n_items={sem.shape[0]}, n_layers={sem.shape[1]})")


if __name__ == "__main__":
    main()
