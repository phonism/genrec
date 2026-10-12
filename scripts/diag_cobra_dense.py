"""
Decisive COBRA diagnostics: is the dense branch healthy, and are the RQ residual
codes learnable at all? Three probes on one checkpoint, forward-only:

  A. Dense-branch health (GPU):
     - oracle-context dense retrieval: teacher-force the TRUE target codes, take
       the model's dense prediction (hidden at the target's last sparse position),
       retrieve over ALL items -> R@1/5/10. Paper's dense-only ablation ~0.0681.
     - in-bucket disambiguation: among items sharing the target's sem-id, does
       the dense vector rank the right one first? (what BeamFusion relies on)
     - encoder collapse check: mean pairwise cos of item dense vectors.
     - loss balance + vec_cos_sim + TF per-codebook/joint acc on BOTH train and
       valid batches (the June diagnosis only ever measured train).

  B. Code learnability probe (CPU-ish):
     - semantic coherence of RQ buckets: mean pairwise cos of xl embeddings for
       item pairs sharing c0 / (c0,c1) / all 3, vs random pairs. If (c0,c1) adds
       nothing over c0, residual codes carry no semantics -> unlearnable.
     - bigram floor: predict valid target c1 from P(c1|c0) train counts (and
       majority-class c2). If the model's valid acc ~= this floor, it learned
       nothing about residuals.

Usage (remote, venv):
    python scripts/diag_cobra_dense.py --ckpt out/cobra/amazon/beauty/checkpoint_final.pt \
        --max-batches 200
"""
import argparse
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

from genrec.data.amazon_cobra import AmazonCobraDataset
from genrec.models.cobra import Cobra
from genrec.trainers.cobra_trainer import cobra_collate_fn

ID_VOCAB, N_CB, D_MODEL = 32, 3, 768
ENC_BASE = "/root/workspace/models_hub/sentence-t5-base"
ENC_XL = "/root/workspace/models_hub/sentence-t5-xl"
RQVAE = "./out/cobra/amazon/beauty/rqvae/checkpoint_epoch_4999.pt"


def make_dataset(split, args):
    return AmazonCobraDataset(
        root="dataset/amazon", split=args.split, train_test_split=split,
        max_seq_len=args.max_seq_len, pretrained_rqvae_path=args.rqvae,
        encoder_model_name=args.rqvae_encoder, max_text_len=128,
        rqvae_input_dim=768, rqvae_embed_dim=32, rqvae_hidden_dims=[512, 256, 128, 64],
        rqvae_codebook_size=ID_VOCAB, rqvae_n_layers=N_CB,
        item_text_format=args.item_text_format,
        rqvae_item_text_format=args.rqvae_item_text_format,
        min_raw_codebook_usage=args.min_raw_codebook_usage,
        evaluation_protocol="leave_one_out",
    )


@torch.no_grad()
def encode_texts(model, token_ids, dev, chunk=256):
    """token_ids: (N, L) -> (N, D) via model.encoder, one text per row."""
    outs = []
    for i in range(0, token_ids.shape[0], chunk):
        t = token_ids[i:i + chunk].to(dev)
        outs.append(model.encoder(t.unsqueeze(1)).squeeze(1))
    return torch.cat(outs, 0)


@torch.no_grad()
def forward_hidden(model, input_ids, encoder_input_ids, dev):
    """Replicate the model forward, return (h, vecs, seq_mask_interleaved)."""
    m = model
    enc = encoder_input_ids.to(dev)
    B_enc, T_enc, L_enc = enc.shape
    flat = enc.view(B_enc * T_enc, L_enc)
    vecs = []
    for i in range(0, flat.shape[0], 256):
        vecs.append(m.encoder(flat[i:i + 256].unsqueeze(1)).squeeze(1))
    vecs = torch.cat(vecs, 0).view(B_enc, T_enc, -1)
    ids = input_ids.to(dev)
    seq_mask = (ids != m.pad_id)
    seq_mask_i = m.interleave_seq_mask(seq_mask, m.C)
    emb = m.cobra_emb(ids, vecs, seq_mask_i)
    h = m.decoder(emb, tgt_key_padding_mask=~seq_mask_i)
    return h, vecs, seq_mask_i


def tf_accs_at_target(model, h, input_ids, dev):
    """TF acc for the LAST item's codes only (the eval target), given true prefix.
    Returns per-codebook correct masks (B,) each, using per-sample lengths."""
    m = model
    ids = input_ids.to(dev)
    n_items = (ids != m.pad_id).sum(dim=1) // m.C  # (B,)
    last = n_items - 1  # target item index
    correct = []
    for c in range(m.C):
        if c == 0:
            pos = (last - 1) * (m.C + 1) + m.C  # dense position of previous item
        else:
            pos = last * (m.C + 1) + (c - 1)    # previous code of target item
        hh = h[torch.arange(h.shape[0], device=dev), pos]
        logits = m.sparse_head[c](hh)
        tgt = ids[torch.arange(ids.shape[0], device=dev), last * m.C + c]
        correct.append(logits.argmax(-1) == tgt)
    return correct, n_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/cobra/amazon/beauty/checkpoint_final.pt")
    ap.add_argument("--rqvae", default=RQVAE)
    ap.add_argument("--split", default="beauty")
    ap.add_argument("--rqvae-encoder", default=ENC_XL)
    ap.add_argument("--model-encoder", default=ENC_BASE)
    ap.add_argument("--encoder-type", choices=("light", "pretrained"), default="pretrained")
    ap.add_argument("--encoder-pooling", choices=("mean", "cls"), default="mean")
    ap.add_argument("--item-text-format", choices=("title_brand", "flattened_attributes", "paper_attributes"), default="title_brand")
    ap.add_argument("--rqvae-item-text-format", default="legacy")
    ap.add_argument("--min-raw-codebook-usage", type=float, default=0.0)
    ap.add_argument("--max-seq-len", type=int, default=50)
    ap.add_argument("--max-batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    dev = "cuda"

    valid_ds = make_dataset("valid", args)
    train_ds = make_dataset("train", args)
    pad_id = ID_VOCAB * N_CB

    model = Cobra(id_vocab_size=ID_VOCAB, n_codebooks=N_CB, d_model=D_MODEL, max_len=1024,
                  temperature=0.07, decoder_n_layers=2, decoder_num_heads=6,
                  encoder_type=args.encoder_type, encoder_model_name=args.model_encoder,
                  encoder_n_layers=1, encoder_hidden_dim=D_MODEL,
                  encoder_num_heads=6, encoder_pooling=args.encoder_pooling,
                  dense_negative_scope="all_batch").to(dev)
    sd = torch.load(args.ckpt, map_location=dev)["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"loaded {args.ckpt} missing={len(missing)} unexpected={len(unexpected)} "
        f"missing_keys={missing[:5]} unexpected_keys={unexpected[:5]}"
    )
    model.eval()

    # ---------- item tables ----------
    sem_ids = torch.tensor(valid_ds.sem_ids_list)  # (N_items, C)
    n_items_total = sem_ids.shape[0]
    print(f"items={n_items_total}")

    item_dense = encode_texts(model, valid_ds.item_token_ids, dev)  # (N, D)
    item_dense = F.normalize(item_dense, dim=-1)

    # encoder collapse check
    idx = torch.randperm(n_items_total)[:2000]
    sub = item_dense[idx]
    cos_mat = sub @ sub.T
    off = cos_mat[~torch.eye(len(idx), dtype=torch.bool, device=dev)]
    print(f"[collapse-check] item dense vecs: mean pairwise cos={off.mean().item():.4f} "
          f"p5={off.quantile(0.05).item():.4f} p95={off.quantile(0.95).item():.4f}")

    # sem-id -> bucket
    buckets = defaultdict(list)
    for i, s in enumerate(valid_ds.sem_ids_list):
        buckets[tuple(s)].append(i)

    # ---------- A. oracle dense retrieval + valid TF ----------
    collate = lambda b: cobra_collate_fn(b, pad_id=pad_id, n_codebooks=N_CB, is_train=True)
    loader_cfg = dict(batch_size=args.batch_size, shuffle=False)

    hits = {1: 0, 5: 0, 10: 0}
    bucket_top1, bucket_n = 0, 0
    tf_c = [0, 0, 0]
    tf_joint = 0
    n_seen = 0
    cos_pred_gt = 0.0

    for st in range(0, min(len(valid_ds), args.max_batches * args.batch_size), args.batch_size):
        batch = [valid_ds[i] for i in range(st, min(st + args.batch_size, len(valid_ds)))]
        data = collate(batch)
        h, vecs, _ = forward_hidden(model, data["input_ids"], data["encoder_input_ids"], dev)
        ids = data["input_ids"].to(dev)
        tgt_items = data["target_item_ids"].to(dev)
        B = ids.shape[0]

        correct, n_it = tf_accs_at_target(model, h, ids, dev)
        for c in range(N_CB):
            tf_c[c] += correct[c].sum().item()
        tf_joint += (correct[0] & correct[1] & correct[2]).sum().item()

        # oracle dense prediction position: last sparse position of target item
        last = n_it - 1
        pos = last * (N_CB + 1) + (N_CB - 1)
        q = h[torch.arange(B, device=dev), pos]
        q = F.normalize(q, dim=-1)
        gt_vec = item_dense[tgt_items]
        cos_pred_gt += (q * gt_vec).sum(-1).sum().item()

        sims = q @ item_dense.T  # (B, N_items)
        top10 = sims.topk(10, dim=-1).indices
        for k in (1, 5, 10):
            hits[k] += (top10[:, :k] == tgt_items.unsqueeze(1)).any(-1).sum().item()

        # in-bucket disambiguation
        for b in range(B):
            t = tgt_items[b].item()
            bucket = buckets[tuple(valid_ds.sem_ids_list[t])]
            if len(bucket) < 2:
                continue
            bidx = torch.tensor(bucket, device=dev)
            s = item_dense[bidx] @ q[b]
            bucket_top1 += int(bidx[s.argmax()].item() == t)
            bucket_n += 1

        n_seen += B

    print(f"\n==== A. oracle-context dense retrieval (valid, N={n_seen}) ====")
    print(f"TF acc at target (VALID): c0={tf_c[0]/n_seen:.4f} c1={tf_c[1]/n_seen:.4f} "
          f"c2={tf_c[2]/n_seen:.4f}  joint={tf_joint/n_seen:.4f}  (chance=1/32=0.031)")
    print(f"dense pred-vs-gt cos (valid): {cos_pred_gt/n_seen:.4f}")
    for k in (1, 5, 10):
        print(f"dense-only full-corpus R@{k} = {hits[k]/n_seen:.4f}   (paper dense-only ablation R@10=0.0681)")
    print(f"in-bucket disambiguation top-1 = {bucket_top1/max(bucket_n,1):.4f} over {bucket_n} multi-item buckets "
          f"(random within avg bucket ~0.57 given avg 1.75 items)")

    # train-batch comparison (memorization reference)
    tf_c_tr = [0, 0, 0]
    tf_joint_tr = 0
    cos_tr = 0.0
    n_tr = 0
    n_train_batches = min(50, args.max_batches)
    for st in range(0, n_train_batches * args.batch_size, args.batch_size):
        batch = [train_ds[i] for i in range(st, min(st + args.batch_size, len(train_ds)))]
        data = collate(batch)
        h, vecs, _ = forward_hidden(model, data["input_ids"], data["encoder_input_ids"], dev)
        ids = data["input_ids"].to(dev)
        correct, n_it = tf_accs_at_target(model, h, ids, dev)
        for c in range(N_CB):
            tf_c_tr[c] += correct[c].sum().item()
        tf_joint_tr += (correct[0] & correct[1] & correct[2]).sum().item()
        last = n_it - 1
        pos = last * (N_CB + 1) + (N_CB - 1)
        q = F.normalize(h[torch.arange(ids.shape[0], device=dev), pos], dim=-1)
        tgt_items = data["target_item_ids"].to(dev)
        cos_tr += (q * item_dense[tgt_items]).sum(-1).sum().item()
        n_tr += ids.shape[0]
    print(f"\nTF acc at target (TRAIN, same protocol): c0={tf_c_tr[0]/n_tr:.4f} "
          f"c1={tf_c_tr[1]/n_tr:.4f} c2={tf_c_tr[2]/n_tr:.4f}  joint={tf_joint_tr/n_tr:.4f}")
    print(f"dense pred-vs-gt cos (train): {cos_tr/n_tr:.4f}")

    # ---------- B. code learnability probe ----------
    print(f"\n==== B. RQ code learnability probe ====")
    xl_emb = F.normalize(valid_ds.item_embeddings, dim=-1)  # (N, 768) xl space

    def mean_pair_cos(pairs):
        if not pairs:
            return float("nan"), 0
        a = xl_emb[[p[0] for p in pairs]]
        b = xl_emb[[p[1] for p in pairs]]
        return (a * b).sum(-1).mean().item(), len(pairs)

    import random
    random.seed(0)
    rand_pairs = [(random.randrange(n_items_total), random.randrange(n_items_total))
                  for _ in range(20000)]
    rand_pairs = [(a, b) for a, b in rand_pairs if a != b]

    by_c0 = defaultdict(list)
    by_c01 = defaultdict(list)
    for i, s in enumerate(valid_ds.sem_ids_list):
        by_c0[s[0]].append(i)
        by_c01[(s[0], s[1])].append(i)

    def sample_within(groups, n=20000):
        pairs = []
        keys = [k for k, v in groups.items() if len(v) >= 2]
        while len(pairs) < n and keys:
            k = random.choice(keys)
            a, b = random.sample(groups[k], 2)
            pairs.append((a, b))
        return pairs

    c0_pairs = sample_within(by_c0)
    c01_pairs = sample_within(by_c01)
    c012_pairs = sample_within(buckets)

    for name, pairs in [("random", rand_pairs), ("same c0", c0_pairs),
                        ("same (c0,c1)", c01_pairs), ("same (c0,c1,c2)", c012_pairs)]:
        m, n = mean_pair_cos(pairs)
        print(f"xl-embedding mean pairwise cos | {name:>15}: {m:.4f}  (n={n})")

    # bigram floor for c1 / majority floor for c2, train -> valid targets
    c1_given_c0 = defaultdict(Counter)
    c2_counter = Counter()
    c2_given_c01 = defaultdict(Counter)
    for s in train_ds.samples:
        t = s["target"]
        if t >= n_items_total:
            continue
        c0, c1, c2 = valid_ds.sem_ids_list[t]
        c1_given_c0[c0][c1] += 1
        c2_counter[c2] += 1
        c2_given_c01[(c0, c1)][c2] += 1

    ok1 = ok2 = ok2b = tot = 0
    for s in valid_ds.samples:
        t = s["target"]
        if t >= n_items_total:
            continue
        c0, c1, c2 = valid_ds.sem_ids_list[t]
        tot += 1
        if c1_given_c0[c0] and c1_given_c0[c0].most_common(1)[0][0] == c1:
            ok1 += 1
        if c2_counter.most_common(1)[0][0] == c2:
            ok2 += 1
        if c2_given_c01[(c0, c1)] and c2_given_c01[(c0, c1)].most_common(1)[0][0] == c2:
            ok2b += 1
    print(f"\nprior floors on VALID targets (no history info):")
    print(f"  c1 via train argmax P(c1|c0)        = {ok1/tot:.4f}   (model valid c1 ~0.05)")
    print(f"  c2 via train majority class          = {ok2/tot:.4f}   (model valid c2 ~0.04)")
    print(f"  c2 via train argmax P(c2|c0,c1)      = {ok2b/tot:.4f}")


if __name__ == "__main__":
    main()
