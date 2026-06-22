"""
Localize COBRA's failure: teacher-forced sparse prediction vs autoregressive
generation, on the SAME test batches. If teacher-forced joint top-1 >> generation
R@20, the bottleneck is generation (exposure bias / a generate() bug), not the
sparse branch.

Usage (remote, venv):
    python scripts/diag_cobra_gen.py --ckpt out/.../checkpoint_final.pt --nb 5
"""
import argparse
import torch

from genrec.data.amazon_cobra import AmazonCobraDataset
from genrec.models.cobra import Cobra

ID_VOCAB, N_CB, D_MODEL = 32, 3, 768
ENC_BASE = "/root/workspace/models_hub/sentence-t5-base"
ENC_XL = "/root/workspace/models_hub/sentence-t5-xl"
RQVAE = "./out/cobra/amazon/beauty/rqvae/checkpoint_epoch_4999.pt"


def build_eval_batch(ds, idxs, pad_id):
    n_items = ds.item_token_ids.shape[0]
    max_text = ds.item_token_ids.shape[-1]
    rows = [ds[i] for i in idxs]
    max_seq = max(len(r["input_ids"]) // N_CB for r in rows)
    B = len(rows)
    input_ids = torch.full((B, max_seq * N_CB), pad_id, dtype=torch.long)
    enc_ids = torch.zeros((B, max_seq, max_text), dtype=torch.long)
    tgt_sem = torch.zeros((B, N_CB), dtype=torch.long)
    for b, r in enumerate(rows):
        sl = len(r["input_ids"])
        ni = sl // N_CB
        input_ids[b, :sl] = torch.tensor(r["input_ids"])
        enc_ids[b, :ni] = r["encoder_input_ids"]
        tgt_sem[b] = torch.tensor(r["target_sem_ids"])
    return input_ids, enc_ids, tgt_sem


@torch.no_grad()
def teacher_forced_joint(model, hist_input_ids, hist_enc, tgt_sem, dev):
    """Append the TRUE target codes and measure joint (all-3) top-1 with teacher forcing."""
    m = model
    B = hist_input_ids.size(0)
    # build full sequence = history + target (like train collate is_train=True)
    full = torch.cat([hist_input_ids, tgt_sem], dim=1)  # (B, (T+1)*C)
    # encoder inputs need T+1 items; reuse last history enc as a stand-in for target text
    # (we only need the decoder hidden states at the target's sparse positions, which
    #  depend on history dense vecs + the true target codes — target's own dense vec
    #  is NOT used to predict its sparse codes, so the stand-in text is irrelevant here)
    pad_enc = hist_enc[:, -1:, :]
    enc = torch.cat([hist_enc, pad_enc], dim=1)
    # encode
    Be, Te, Le = enc.shape
    flat = enc.reshape(Be * Te, Le)
    vecs = []
    for i in range(0, flat.shape[0], 256):
        vecs.append(m.encoder(flat[i:i + 256].unsqueeze(1)).squeeze(1))
    vecs = torch.cat(vecs, 0).view(Be, Te, -1)
    seq_mask = (full != m.pad_id)
    seq_mask = m.interleave_seq_mask(seq_mask, m.C)
    emb = m.cobra_emb(full, vecs, seq_mask)
    h = m.decoder(emb, tgt_key_padding_mask=~seq_mask)
    T = full.size(1) // m.C  # = T_hist + 1
    # predict the LAST item's codes (the target), teacher-forced
    last = T - 1
    preds = []
    for c in range(m.C):
        if c == 0:
            pos = last * (m.C + 1) - 1  # dense pos of previous item (= last*(C+1)+C of item last-1)
        else:
            pos = last * (m.C + 1) + (c - 1)
        logits = m.sparse_head[c](h[:, pos, :])  # (B, V)
        preds.append(logits.argmax(-1))
    preds = torch.stack(preds, dim=1)  # (B, C)
    per_cb = [(preds[:, c] == tgt_sem[:, c]).float().mean().item() for c in range(m.C)]
    joint = (preds == tgt_sem).all(1).float().mean().item()
    return per_cb, joint


def rank_in(target, beams):
    # target (B,C), beams (B,K,C) -> hit@K booleans via first match
    matches = (beams == target.unsqueeze(1)).all(-1)  # (B,K)
    return matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nb", type=int, default=5)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda"

    # Contended cluster: claim GPU memory up front (fails fast if the card is already
    # full) and KEEP it in PyTorch's caching pool through the run. `del` without
    # empty_cache() keeps the block reserved to this process (driver still sees it as
    # used), so external jobs cannot grab it and our later model/activations reuse it.
    _hold = torch.empty(int(34e9 // 2), dtype=torch.float16, device=dev)

    ds = AmazonCobraDataset(
        root="dataset/amazon", split="beauty", train_test_split="test",
        max_seq_len=100, pretrained_rqvae_path=RQVAE, encoder_model_name=ENC_XL,
        max_text_len=128, rqvae_input_dim=768, rqvae_embed_dim=32,
        rqvae_hidden_dims=[512, 256, 128, 64], rqvae_codebook_size=ID_VOCAB, rqvae_n_layers=N_CB,
    )
    pad_id = ID_VOCAB * N_CB
    del _hold  # free into PyTorch's pool (NOT empty_cache) -> stays reserved to us
    model = Cobra(id_vocab_size=ID_VOCAB, n_codebooks=N_CB, d_model=D_MODEL, max_len=1024,
                  temperature=0.07, decoder_n_layers=2, decoder_num_heads=6,
                  encoder_type="pretrained", encoder_model_name=ENC_BASE).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"], strict=False)
    model.eval()
    print(f"loaded {args.ckpt}")

    tf_cb = [0.0, 0.0, 0.0]
    tf_joint = 0.0
    gen_cb = [0.0, 0.0, 0.0]
    hit = {1: 0.0, 5: 0.0, 10: 0.0, args.beam: 0.0}
    n = 0
    bs = args.bs
    for st in range(0, args.nb * bs, bs):
        idxs = list(range(st, min(st + bs, len(ds.samples))))
        if len(idxs) < bs:
            break
        hist, enc, tgt = build_eval_batch(ds, idxs, pad_id)
        hist, enc, tgt = hist.to(dev), enc.to(dev), tgt.to(dev)

        cb, joint = teacher_forced_joint(model, hist, enc, tgt, dev)
        for c in range(N_CB):
            tf_cb[c] += cb[c]
        tf_joint += joint

        gen = model.generate(input_ids=hist, encoder_input_ids=enc, n_candidates=args.beam)
        beams = gen.sem_ids  # (B, beam, C)
        top1 = beams[:, 0, :]
        for c in range(N_CB):
            gen_cb[c] += (top1[:, c] == tgt[:, c]).float().mean().item()
        m = rank_in(tgt, beams)  # (B, beam)
        for k in hit:
            hit[k] += (m[:, :k].any(1)).float().mean().item()
        n += 1

    print(f"\n==== teacher-forced vs generation ({n} test batches, beam={args.beam}) ====")
    print(f"[teacher-forced] per-cb c0={tf_cb[0]/n:.3f} c1={tf_cb[1]/n:.3f} c2={tf_cb[2]/n:.3f} | JOINT top1={tf_joint/n:.4f}")
    print(f"[generation]     per-cb c0={gen_cb[0]/n:.3f} c1={gen_cb[1]/n:.3f} c2={gen_cb[2]/n:.3f} (top-1 beam)")
    print(f"[generation]     sem-id  R@1={hit[1]/n:.4f} R@5={hit[5]/n:.4f} R@10={hit[10]/n:.4f} R@{args.beam}={hit[args.beam]/n:.4f}")
    print("If teacher-forced JOINT top1 >> generation R@beam -> generation is the bottleneck.")


if __name__ == "__main__":
    main()
