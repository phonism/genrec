"""
Item-level (honest) re-evaluation of a trained COBRA checkpoint on Amazon Beauty.

The COBRA trainer evaluates by matching the *semantic ID tuple* of the target
against the sem-ids of the top-K retrieved items. With a 32x32x32 codebook and
no uniquifying code, many items collide on the same sem-id, so a "hit" is scored
whenever ANY retrieved item shares the target's bucket -- even a different item.
This both inflates Recall and makes BeamFusion's dense disambiguation invisible
(the whole point of COBRA).

This script reproduces the repo's sem-id metric AND computes the true item-id
metric in a single pass, so we can quantify the inflation factor.

Usage (on gpu-remote, inside venv):
    python scripts/eval_cobra_itemlevel.py \
        --ckpt out/cobra/amazon/beauty/checkpoint_final.pt \
        --split beauty [--max-batches N]
"""
import argparse
import math
from collections import defaultdict

import torch
from torch.nn.functional import normalize

from genrec.data.amazon_cobra import AmazonCobraDataset
from genrec.models.cobra import Cobra

# ----- fixed hyper-params, must match config/cobra/amazon/cobra.gin -----
ID_VOCAB = 32
N_CODEBOOKS = 3
D_MODEL = 768
DEC_LAYERS = 2
DEC_HEADS = 6
MAX_LEN = 1024
TEMPERATURE = 0.07
ENC_BASE = "/root/workspace/models_hub/sentence-t5-base"
ENC_XL = "/root/workspace/models_hub/sentence-t5-xl"
RQVAE = "./out/cobra/amazon/{split}/rqvae/checkpoint_epoch_4999.pt"


def build_batches(ds, indices, pad_id):
    """Eval collate that ALSO returns the ground-truth item id."""
    n_items = ds.item_token_ids.shape[0]
    max_text = ds.item_token_ids.shape[-1]
    rows = [ds[i] for i in indices]
    max_seq = max(len(r["input_ids"]) // N_CODEBOOKS for r in rows)
    B = len(rows)
    input_ids = torch.full((B, max_seq * N_CODEBOOKS), pad_id, dtype=torch.long)
    enc_ids = torch.zeros((B, max_seq, max_text), dtype=torch.long)
    target_sem = torch.zeros((B, N_CODEBOOKS), dtype=torch.long)
    target_item = torch.zeros(B, dtype=torch.long)
    for b, (i, r) in enumerate(zip(indices, rows)):
        sl = len(r["input_ids"])
        ni = sl // N_CODEBOOKS
        input_ids[b, :sl] = torch.tensor(r["input_ids"])
        enc_ids[b, :ni] = r["encoder_input_ids"]
        target_sem[b] = torch.tensor(r["target_sem_ids"])
        target_item[b] = ds.samples[i]["target"]
    return input_ids, enc_ids, target_sem, target_item


class Metric:
    def __init__(self, ks=(1, 5, 10)):
        self.ks = ks
        self.recall = defaultdict(float)
        self.ndcg = defaultdict(float)
        self.total = 0

    def add(self, hit_rank):
        # hit_rank: (B,) 0-indexed rank of the hit, or large (=K) if miss
        for k in self.ks:
            hit = (hit_rank < k)
            self.recall[k] += hit.float().sum().item()
            dcg = torch.where(hit, 1.0 / torch.log2(hit_rank.float() + 2),
                              torch.zeros_like(hit_rank, dtype=torch.float))
            self.ndcg[k] += dcg.sum().item()
        self.total += hit_rank.numel()

    def report(self, tag):
        out = {}
        for k in self.ks:
            out[f"R@{k}"] = round(self.recall[k] / max(self.total, 1), 5)
            out[f"N@{k}"] = round(self.ndcg[k] / max(self.total, 1), 5)
        print(f"[{tag}] (N={self.total}) " +
              "  ".join(f"{m}={v}" for m, v in out.items()))
        return out


def rank_of(target, topk):
    """target:(B,) or (B,D); topk:(B,K) or (B,K,D) -> (B,) first-match rank else K."""
    K = topk.size(1)
    if target.dim() == 1:
        matches = topk == target.unsqueeze(1)            # (B,K)
    else:
        matches = (topk == target.unsqueeze(1)).all(-1)  # (B,K)
    found = matches.any(1)
    rank = matches.float().argmax(1)
    return torch.where(found, rank, torch.tensor(K, device=topk.device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="beauty")
    ap.add_argument("--eval-split", default="test", choices=["valid", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-beam", type=int, default=20)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--psi", type=float, default=16.0)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    dev = "cuda"

    ds = AmazonCobraDataset(
        root="dataset/amazon", split=args.split, train_test_split=args.eval_split,
        max_seq_len=100, pretrained_rqvae_path=RQVAE.format(split=args.split),
        encoder_model_name=ENC_XL, max_text_len=128,
        rqvae_input_dim=768, rqvae_embed_dim=32,
        rqvae_hidden_dims=[512, 256, 128, 64],
        rqvae_codebook_size=ID_VOCAB, rqvae_n_layers=N_CODEBOOKS,
    )
    pad_id = ID_VOCAB * N_CODEBOOKS

    model = Cobra(
        id_vocab_size=ID_VOCAB, n_codebooks=N_CODEBOOKS, d_model=D_MODEL,
        max_len=MAX_LEN, temperature=TEMPERATURE,
        decoder_n_layers=DEC_LAYERS, decoder_num_heads=DEC_HEADS,
        encoder_type="pretrained", encoder_model_name=ENC_BASE,
    ).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {args.ckpt}  missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    # item tables (sem_ids_list / item_texts are identical across splits)
    item_sem_ids = torch.tensor(ds.sem_ids_list, dtype=torch.long, device=dev)
    n_items = len(ds.sem_ids_list)
    sem_id_to_items = defaultdict(list)
    for idx, s in enumerate(ds.sem_ids_list):
        sem_id_to_items[tuple(s)].append(idx)
    sem_id_to_items = dict(sem_id_to_items)
    uniq = len(sem_id_to_items)
    print(f"items={n_items}  unique_sem_ids={uniq}  "
          f"collision_rate={1 - uniq / n_items:.3f}  "
          f"avg_items_per_bucket={n_items / uniq:.2f}")

    # precompute item dense vecs via the model encoder (batched)
    tok = ds.item_token_ids.to(dev)  # (N, L)
    vecs = []
    with torch.no_grad():
        for i in range(0, n_items, 256):
            v = model.encoder(tok[i:i + 256].unsqueeze(1)).squeeze(1)
            vecs.append(normalize(v, p=2, dim=-1))
    item_dense_vecs = torch.cat(vecs, 0)

    m_item = Metric()
    m_semid = Metric()
    m_sparse_item = Metric()   # sparse-only generate, item-level (best-effort: 1 item per gen sem-id)
    order = list(range(len(ds.samples)))
    nb = 0
    with torch.no_grad():
        for st in range(0, len(order), args.batch_size):
            idxs = order[st:st + args.batch_size]
            input_ids, enc_ids, tgt_sem, tgt_item = build_batches(ds, idxs, pad_id)
            input_ids, enc_ids = input_ids.to(dev), enc_ids.to(dev)
            tgt_sem, tgt_item = tgt_sem.to(dev), tgt_item.to(dev)

            bf = model.beam_fusion(
                input_ids=input_ids, encoder_input_ids=enc_ids,
                item_dense_vecs=item_dense_vecs, item_sem_ids=item_sem_ids,
                sem_id_to_items=sem_id_to_items, n_candidates=10,
                n_beam=args.n_beam, tau=args.tau, psi=args.psi,
            )
            # item-level: ground-truth ITEM id vs retrieved item ids
            m_item.add(rank_of(tgt_item, bf.item_ids))
            # sem-id level: reproduce the repo's inflated metric
            m_semid.add(rank_of(tgt_sem, bf.sem_ids))

            nb += 1
            if nb % 10 == 0:
                print(f"  batch {nb}/{(len(order)+args.batch_size-1)//args.batch_size}")
            if args.max_batches and nb >= args.max_batches:
                break

    print(f"\n==== COBRA {args.split} honest re-eval ({args.eval_split}) ====")
    r_item = m_item.report("BeamFusion ITEM-level (honest)")
    r_semid = m_semid.report("BeamFusion SEM-ID level (repo metric)")
    if r_semid["R@10"] > 0:
        print(f"\ninflation R@10 (semid/item) = {r_semid['R@10'] / max(r_item['R@10'],1e-9):.2f}x")
    print("paper COBRA Beauty: R@5=0.0537 R@10=0.0725 N@5=0.0395 N@10=0.0456")


if __name__ == "__main__":
    main()
