"""
Forward-only diagnostic for COBRA: is the sparse branch being starved by the
dense InfoNCE loss? Prints loss_sparse / loss_dense / vec_cos_sim and the
per-codebook top-1 accuracy over a handful of TRAIN batches, using a given ckpt
(or a fresh model). No backward pass.

Usage (remote, venv):
    python scripts/diag_cobra_loss.py [--ckpt out/.../checkpoint_final.pt] [--n 15]
"""
import argparse
import torch
import torch.nn.functional as F

from genrec.data.amazon_cobra import AmazonCobraDataset
from genrec.models.cobra import Cobra
from genrec.trainers.cobra_trainer import cobra_collate_fn

ID_VOCAB, N_CB, D_MODEL = 32, 3, 768
ENC_BASE = "/root/workspace/models_hub/sentence-t5-base"
ENC_XL = "/root/workspace/models_hub/sentence-t5-xl"
RQVAE = "./out/cobra/amazon/beauty/rqvae/checkpoint_epoch_4999.pt"


def per_codebook_acc(model, input_ids, encoder_input_ids):
    """Replicate the forward up to per-codebook top-1 accuracy (no loss)."""
    m = model
    enc = encoder_input_ids
    B_enc, T_enc, L_enc = enc.shape
    flat = enc.view(B_enc * T_enc, L_enc)
    vecs = []
    for i in range(0, flat.shape[0], 256):
        vecs.append(m.encoder(flat[i:i + 256].unsqueeze(1)).squeeze(1))
    vecs = torch.cat(vecs, 0).view(B_enc, T_enc, -1)
    B, Tf = input_ids.shape
    seq_mask = (input_ids != m.pad_id)
    seq_mask = m.interleave_seq_mask(seq_mask, m.C)
    emb = m.cobra_emb(input_ids, vecs, seq_mask)
    h = m.decoder(emb, tgt_key_padding_mask=~seq_mask)
    T = Tf // m.C
    accs = []
    for c in range(m.C):
        if c == 0:
            pos = torch.arange(0, T - 1, device=h.device) * (m.C + 1) + m.C
            tgt = input_ids[:, torch.arange(1, T, device=h.device) * m.C]
        else:
            pos = torch.arange(1, T, device=h.device) * (m.C + 1) + (c - 1)
            tgt = input_ids[:, torch.arange(1, T, device=h.device) * m.C + c]
        logits = m.sparse_head[c](h[:, pos, :])
        valid = tgt != m.pad_id
        correct = ((logits.argmax(-1) == tgt) & valid).sum().item()
        accs.append(correct / max(valid.sum().item(), 1))
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()
    dev = "cuda"

    ds = AmazonCobraDataset(
        root="dataset/amazon", split="beauty", train_test_split="train",
        max_seq_len=100, pretrained_rqvae_path=RQVAE,
        encoder_model_name=ENC_XL, max_text_len=128,
        rqvae_input_dim=768, rqvae_embed_dim=32, rqvae_hidden_dims=[512, 256, 128, 64],
        rqvae_codebook_size=ID_VOCAB, rqvae_n_layers=N_CB,
    )
    pad_id = ID_VOCAB * N_CB
    model = Cobra(id_vocab_size=ID_VOCAB, n_codebooks=N_CB, d_model=D_MODEL, max_len=1024,
                  temperature=0.07, decoder_n_layers=2, decoder_num_heads=6,
                  encoder_type="pretrained", encoder_model_name=ENC_BASE).to(dev)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"], strict=False)
        print(f"loaded {args.ckpt}")
    else:
        print("fresh model (no ckpt)")
    model.eval()

    collate = lambda b: cobra_collate_fn(b, pad_id=pad_id, n_codebooks=N_CB, is_train=True)
    import random
    random.seed(0)
    idxs = random.sample(range(len(ds)), min(len(ds), args.n * 32))

    ls_sum = ld_sum = cos_sum = 0.0
    cb = [0.0, 0.0, 0.0]
    nb = 0
    with torch.no_grad():
        for st in range(0, len(idxs), 32):
            batch = [ds[i] for i in idxs[st:st + 32]]
            if len(batch) < 32:
                break
            data = collate(batch)
            out = model(input_ids=data["input_ids"].to(dev),
                        encoder_input_ids=data["encoder_input_ids"].to(dev))
            ls_sum += out.loss_sparse.item()
            ld_sum += out.loss_dense.item()
            cos_sum += out.vec_cos_sim.item()
            accs = per_codebook_acc(model, data["input_ids"].to(dev),
                                    data["encoder_input_ids"].to(dev))
            for c in range(N_CB):
                cb[c] += accs[c]
            nb += 1
            if nb >= args.n:
                break

    print(f"\n==== COBRA loss-balance diagnostic ({nb} train batches) ====")
    print(f"loss_sparse = {ls_sum/nb:.4f}")
    print(f"loss_dense  = {ld_sum/nb:.4f}")
    print(f"ratio dense/sparse = {ld_sum/max(ls_sum,1e-9):.3f}")
    print(f"vec_cos_sim = {cos_sum/nb:.4f}  (dense branch quality; ->1 is good)")
    print(f"per-codebook top1 acc: c0={cb[0]/nb:.4f}  c1={cb[1]/nb:.4f}  c2={cb[2]/nb:.4f}  (chance=1/32=0.031)")


if __name__ == "__main__":
    main()
