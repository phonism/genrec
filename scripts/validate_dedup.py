"""Validate LC-Rec-style last-layer Sinkhorn iterative dedup on the existing
3-code TIGER RQ-VAE checkpoint (no retraining). Reports collision rate and
bucket distribution before (argmin) vs after (dedup). CPU-only.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from genrec.models.rqvae import RqVae

SPLIT = "beauty"
PARQUET = f"dataset/amazon/processed/{SPLIT}/item_emb_sentence-t5-xl.parquet"
RQVAE_CKPT = f"out/tiger/amazon/{SPLIT}/rqvae/checkpoint_epoch_4999.pt"


def stats(codes):
    bucket = defaultdict(int)
    for c in codes:
        bucket[tuple(c)] += 1
    sizes = list(bucket.values())
    dist = defaultdict(int)
    for s in sizes:
        dist[s] += 1
    n = len(codes)
    uniq = len(bucket)
    collided = sum(s for s in sizes if s > 1)
    return {
        "items": n, "unique_sid": uniq, "unique_rate": uniq / n,
        "collided_items": collided, "collision_rate": (n - uniq) / n,
        "max_bucket": max(sizes), "size_dist": dict(sorted(dist.items())),
    }


def show(tag, s):
    print(f"\n[{tag}]")
    print(f"  items={s['items']} unique_sid={s['unique_sid']} "
          f"unique_rate={s['unique_rate']:.3%}")
    print(f"  collision_rate={s['collision_rate']:.3%} "
          f"collided_items={s['collided_items']} max_bucket={s['max_bucket']}")
    print(f"  bucket size dist: {s['size_dist']}")


df = pd.read_parquet(PARQUET)
emb = torch.tensor(np.stack(df["embedding"].values), dtype=torch.float32)

rqvae = RqVae(input_dim=768, embed_dim=32, hidden_dims=[512, 256, 128, 64],
              codebook_kmeans_init=False, codebook_normalize=False,
              codebook_sim_vq=False, codebook_size=256, n_layers=3,
              n_cat_features=0, commitment_weight=0.25)
rqvae.load_pretrained(RQVAE_CKPT)
rqvae.eval()

with torch.no_grad():
    base = rqvae.get_semantic_ids(emb).sem_ids.tolist()
    dedup = rqvae.get_semantic_ids_dedup(emb, max_iter=20,
                                         sk_epsilon=0.003, sk_iters=100).tolist()

show("BEFORE  (argmin, current)", stats(base))
show("AFTER   (3-code + Sinkhorn dedup)", stats(dedup))

# how many items had their last code changed
changed = sum(1 for a, b in zip(base, dedup) if a != b)
print(f"\nitems whose SID changed by dedup: {changed} ({changed/len(base):.2%})")
