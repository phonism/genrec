"""
Mechanistic information-flow probe for COBRA's sparse branch.

Question: does the trained model actually USE the history, or does it predict
from priors + the target's own preceding codes only? (valid c1 ~= bigram
P(c1|c0) and c2 ~= P(c2|c0,c1) suggest the latter.)

Method: TF accuracy at the target position on a sparse-only checkpoint under
  A) intact history
  B) history item ORDER shuffled per sample (bag-of-items kept)
  C) history codes replaced by uniform random codes (all info destroyed)
If A ~= C, the model provably ignores history -> mechanical info-flow defect.

Usage:
    python scripts/diag_cobra_infoflow.py --ckpt out/cobra/amazon/beauty_sparse_only/checkpoint_epoch_19.pt \
        --max-batches 100
"""
import argparse

import torch

from genrec.data.amazon_cobra import AmazonCobraDataset
from genrec.models.cobra import Cobra
from genrec.trainers.cobra_trainer import cobra_collate_fn

ID_VOCAB, N_CB, D_MODEL = 32, 3, 768
ENC_XL = "/root/workspace/models_hub/sentence-t5-xl"
RQVAE = "./out/cobra/amazon/beauty/rqvae/checkpoint_epoch_4999.pt"


@torch.no_grad()
def forward_hidden(model, input_ids, encoder_input_ids, dev):
    m = model
    ids = input_ids.to(dev)
    B = ids.shape[0]
    T = encoder_input_ids.shape[1]
    vecs = torch.zeros(B, T, m.d_model, device=dev)  # sparse-only ckpt: dense inputs are zeros
    seq_mask = (ids != m.pad_id)
    seq_mask_i = m.interleave_seq_mask(seq_mask, m.C)
    emb = m.cobra_emb(ids, vecs, seq_mask_i)
    return m.decoder(emb, tgt_key_padding_mask=~seq_mask_i)


@torch.no_grad()
def tf_correct(model, h, ids, dev):
    m = model
    n_items = (ids != m.pad_id).sum(dim=1) // m.C
    last = n_items - 1
    outs = []
    for c in range(m.C):
        if c == 0:
            pos = (last - 1) * (m.C + 1) + m.C
        else:
            pos = last * (m.C + 1) + (c - 1)
        logits = m.sparse_head[c](h[torch.arange(h.shape[0], device=dev), pos])
        tgt = ids[torch.arange(ids.shape[0], device=dev), last * m.C + c]
        outs.append((logits.argmax(-1) == tgt, logits))
    return outs, n_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-batches", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)

    ds = AmazonCobraDataset(
        root="dataset/amazon", split="beauty", train_test_split="valid",
        max_seq_len=100, pretrained_rqvae_path=RQVAE,
        encoder_model_name=ENC_XL, max_text_len=128,
        rqvae_input_dim=768, rqvae_embed_dim=32, rqvae_hidden_dims=[512, 256, 128, 64],
        rqvae_codebook_size=ID_VOCAB, rqvae_n_layers=N_CB,
    )
    pad_id = ID_VOCAB * N_CB

    model = Cobra(id_vocab_size=ID_VOCAB, n_codebooks=N_CB, d_model=D_MODEL, max_len=1024,
                  temperature=0.07, decoder_n_layers=2, decoder_num_heads=6,
                  encoder_type="light", encoder_n_layers=1, encoder_hidden_dim=D_MODEL,
                  encoder_num_heads=6, use_dense=False).to(dev)
    sd = torch.load(args.ckpt, map_location=dev)["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {args.ckpt} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    collate = lambda b: cobra_collate_fn(b, pad_id=pad_id, n_codebooks=N_CB, is_train=True)

    accs = {k: [0, 0, 0] for k in "ABC"}
    joint = {k: 0 for k in "ABC"}
    dlogit_B = dlogit_C = 0.0
    n_seen = 0

    for st in range(0, min(len(ds), args.max_batches * args.batch_size), args.batch_size):
        batch = [ds[i] for i in range(st, min(st + args.batch_size, len(ds)))]
        data = collate(batch)
        ids_a = data["input_ids"].clone()
        B = ids_a.shape[0]
        n_items = (ids_a != pad_id).sum(dim=1) // N_CB
        last = n_items - 1

        # B) shuffle history item order per sample
        ids_b = ids_a.clone()
        for i in range(B):
            nh = last[i].item()  # history item count
            if nh >= 2:
                perm = torch.randperm(nh)
                hist = ids_a[i, :nh * N_CB].view(nh, N_CB)
                ids_b[i, :nh * N_CB] = hist[perm].reshape(-1)

        # C) random history codes
        ids_c = ids_a.clone()
        for i in range(B):
            nh = last[i].item()
            if nh >= 1:
                ids_c[i, :nh * N_CB] = torch.randint(0, ID_VOCAB, (nh * N_CB,))

        enc = data["encoder_input_ids"]
        logits0 = None
        for key, ids in (("A", ids_a), ("B", ids_b), ("C", ids_c)):
            h = forward_hidden(model, ids, enc, dev)
            outs, _ = tf_correct(model, h, ids.to(dev), dev)
            for c in range(N_CB):
                accs[key][c] += outs[c][0].sum().item()
            joint[key] += (outs[0][0] & outs[1][0] & outs[2][0]).sum().item()
            if key == "A":
                logits0 = outs[0][1]
            elif key == "B":
                dlogit_B += (outs[0][1] - logits0).abs().mean().item() * B
            else:
                dlogit_C += (outs[0][1] - logits0).abs().mean().item() * B
        n_seen += B

    print(f"\n==== info-flow probe (valid, N={n_seen}) ====")
    for key, name in (("A", "intact history"), ("B", "shuffled history order"),
                      ("C", "RANDOM history codes")):
        a = accs[key]
        print(f"{name:>24}: c0={a[0]/n_seen:.4f} c1={a[1]/n_seen:.4f} "
              f"c2={a[2]/n_seen:.4f} joint={joint[key]/n_seen:.4f}")
    print(f"mean |dlogit(c0)| B vs A = {dlogit_B/n_seen:.4f}")
    print(f"mean |dlogit(c0)| C vs A = {dlogit_C/n_seen:.4f}")
    print("read: A~=C on c0 -> model ignores history entirely (mechanical bug);")
    print("      A>>C but A~=B -> uses bag of history, no order; A>>B -> uses order.")


if __name__ == "__main__":
    main()
