"""Verify K=256 RQ-KMeans codebook health for Beauty.

Pass criteria (relative to prior K=8192 baseline with layer-2 max_freq=23%):
  - All layers usage >= 95%
  - All layers max_freq <= 10% (prior was 23% on layer 2 — anything <10% is a win)
  - Collision rate <= 25% (K=256 / 12k items theoretical floor)
"""
import sys
import torch
import pandas as pd
from collections import Counter

EMB_PATH = "dataset/amazon/processed/beauty/item_emb_Qwen3-Embedding-8B.parquet"
RQK_PATH = "out/onerec/amazon/beauty/rqkmeans_k256/rqkmeans.pt"


def main() -> int:
    df = pd.read_parquet(EMB_PATH)
    emb_col = next(c for c in df.columns if hasattr(df[c].iloc[0], "__len__")
                   and len(df[c].iloc[0]) > 100)
    embs = torch.tensor([list(x) for x in df[emb_col].tolist()], dtype=torch.float32)
    n = len(embs)

    ck = torch.load(RQK_PATH, map_location="cpu", weights_only=False)
    n_layers, K = ck["n_layers"], ck["codebook_size"]
    centroids = [torch.tensor(c) for c in ck["centroids"]]

    sids = []
    r = embs.clone()
    for cent in centroids:
        d = torch.cdist(r, cent)
        ids = d.argmin(dim=-1)
        sids.append(ids)
        r = r - cent[ids]
    sids = torch.stack(sids, dim=-1)  # [N, L]

    print(f"\n=== K={K} RQ-KMeans health (Beauty, N={n}) ===")
    passed = True
    for l in range(n_layers):
        cnt = Counter(sids[:, l].tolist())
        used = len(cnt)
        top = max(cnt.values())
        use_pct = 100 * used / K
        top_pct = 100 * top / n
        ok_use = use_pct >= 95
        ok_top = top_pct <= 10
        marks = ("✓" if ok_use else "✗", "✓" if ok_top else "✗")
        print(f"  Layer {l}: usage {used}/{K} ({use_pct:.1f}%) {marks[0]}, "
              f"max_freq={top} ({top_pct:.1f}% of items) {marks[1]}")
        if not (ok_use and ok_top):
            passed = False

    keys = [tuple(s.tolist()) for s in sids]
    uniq = len(set(keys))
    coll_rate = 100 * (1 - uniq / n)
    ok_coll = coll_rate <= 25.0
    print(f"\n  Unique SIDs: {uniq}/{n} (collision {coll_rate:.2f}%) "
          f"{'✓' if ok_coll else '✗'}")
    if not ok_coll:
        passed = False

    print(f"\n=== {'PASS ✓' if passed else 'FAIL ✗ — codebook still unhealthy'} ===")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
