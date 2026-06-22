"""
Combined COBRA generation diagnostic + cheap-decoding test, in ONE process (one
GPU claim). Answers, on a test subset:
  (A) item-level R@10 with n_beam=20 vs n_beam=32  -> does widening beams help? (cheap fix)
  (B) generation sem-id R@1/5/10/20/32 vs teacher-forced joint top-1 -> exposure bias?

Designed for a contended cluster: claims GPU memory up front and keeps it in
PyTorch's pool (del WITHOUT empty_cache) so external jobs can't evict the run.

Usage (remote, venv):
    python scripts/diag_cobra_final.py --ckpt out/.../checkpoint_final.pt --nb 80
"""
import argparse
from collections import defaultdict

import torch
from torch.nn.functional import normalize

from genrec.data.amazon_cobra import AmazonCobraDataset
from genrec.models.cobra import Cobra

ID_VOCAB, N_CB, D_MODEL = 32, 3, 768
ENC_BASE = "/root/workspace/models_hub/sentence-t5-base"
ENC_XL = "/root/workspace/models_hub/sentence-t5-xl"
RQVAE = "./out/cobra/amazon/beauty/rqvae/checkpoint_epoch_4999.pt"


def build_batch(ds, idxs, pad_id):
    n_items = ds.item_token_ids.shape[0]
    max_text = ds.item_token_ids.shape[-1]
    rows = [ds[i] for i in idxs]
    max_seq = max(len(r["input_ids"]) // N_CB for r in rows)
    B = len(rows)
    input_ids = torch.full((B, max_seq * N_CB), pad_id, dtype=torch.long)
    enc_ids = torch.zeros((B, max_seq, max_text), dtype=torch.long)
    tgt_sem = torch.zeros((B, N_CB), dtype=torch.long)
    tgt_item = torch.zeros(B, dtype=torch.long)
    for b, (i, r) in enumerate(zip(idxs, rows)):
        sl = len(r["input_ids"]); ni = sl // N_CB
        input_ids[b, :sl] = torch.tensor(r["input_ids"])
        enc_ids[b, :ni] = r["encoder_input_ids"]
        tgt_sem[b] = torch.tensor(r["target_sem_ids"])
        tgt_item[b] = ds.samples[i]["target"]
    return input_ids, enc_ids, tgt_sem, tgt_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nb", type=int, default=80)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda"
    _hold = torch.empty(int(34e9 // 2), dtype=torch.float16, device=dev)  # claim, keep in pool

    ds = AmazonCobraDataset(
        root="dataset/amazon", split="beauty", train_test_split="test", max_seq_len=100,
        pretrained_rqvae_path=RQVAE, encoder_model_name=ENC_XL, max_text_len=128,
        rqvae_input_dim=768, rqvae_embed_dim=32, rqvae_hidden_dims=[512, 256, 128, 64],
        rqvae_codebook_size=ID_VOCAB, rqvae_n_layers=N_CB,
    )
    pad_id = ID_VOCAB * N_CB
    del _hold  # into pool, NOT empty_cache
    model = Cobra(id_vocab_size=ID_VOCAB, n_codebooks=N_CB, d_model=D_MODEL, max_len=1024,
                  temperature=0.07, decoder_n_layers=2, decoder_num_heads=6,
                  encoder_type="pretrained", encoder_model_name=ENC_BASE).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"], strict=False)
    model.eval()
    print(f"loaded {args.ckpt}", flush=True)

    item_sem_ids = torch.tensor(ds.sem_ids_list, dtype=torch.long, device=dev)
    n_items = len(ds.sem_ids_list)
    sem_id_to_items = defaultdict(list)
    for idx, s in enumerate(ds.sem_ids_list):
        sem_id_to_items[tuple(s)].append(idx)
    sem_id_to_items = dict(sem_id_to_items)
    tok = ds.item_token_ids.to(dev)
    vecs = []
    with torch.no_grad():
        for i in range(0, n_items, 256):
            vecs.append(normalize(model.encoder(tok[i:i + 256].unsqueeze(1)).squeeze(1), p=2, dim=-1))
    item_dense_vecs = torch.cat(vecs, 0)

    # accumulators
    hit_item = {20: {5: 0, 10: 0}, 32: {5: 0, 10: 0}}     # (A) item-level by n_beam
    gen_hit = {1: 0, 5: 0, 10: 0, 20: 0, 32: 0}            # (B) generation sem-id recall
    tf_joint = 0
    N = 0

    order = list(range(0, args.nb * args.bs))
    with torch.no_grad():
        for st in range(0, len(order), args.bs):
            idxs = list(range(st, min(st + args.bs, len(ds.samples))))
            if len(idxs) < args.bs:
                break
            inp, enc, tsem, titem = build_batch(ds, idxs, pad_id)
            inp, enc, tsem, titem = inp.to(dev), enc.to(dev), tsem.to(dev), titem.to(dev)

            # (A) item-level via beam_fusion at two beam widths
            for nb in (20, 32):
                bf = model.beam_fusion(input_ids=inp, encoder_input_ids=enc,
                                       item_dense_vecs=item_dense_vecs, item_sem_ids=item_sem_ids,
                                       sem_id_to_items=sem_id_to_items, n_candidates=10,
                                       n_beam=nb, tau=1.0, psi=16.0)
                m = (bf.item_ids == titem.unsqueeze(1))  # (B,10)
                for k in (5, 10):
                    hit_item[nb][k] += m[:, :k].any(1).float().sum().item()

            # (B) raw generation recall + teacher-forced joint top-1
            gen = model.generate(input_ids=inp, encoder_input_ids=enc, n_candidates=32)
            gm = (gen.sem_ids == tsem.unsqueeze(1)).all(-1)  # (B,32)
            for k in (1, 5, 10, 20, 32):
                gen_hit[k] += gm[:, :k].any(1).float().sum().item()
            # teacher-forced joint top-1 at the LAST item (true prefix)
            full = torch.cat([inp, tsem], dim=1)
            penc = torch.cat([enc, enc[:, -1:, :]], dim=1)
            Be, Te, Le = penc.shape
            flat = penc.reshape(Be * Te, Le)
            vv = []
            for i in range(0, flat.shape[0], 256):
                vv.append(model.encoder(flat[i:i + 256].unsqueeze(1)).squeeze(1))
            vv = torch.cat(vv, 0).view(Be, Te, -1)
            sm = (full != model.pad_id)
            sm = model.interleave_seq_mask(sm, model.C)
            h = model.decoder(model.cobra_emb(full, vv, sm), tgt_key_padding_mask=~sm)
            T = full.size(1) // model.C
            last = T - 1
            preds = []
            for c in range(model.C):
                pos = last * (model.C + 1) - 1 if c == 0 else last * (model.C + 1) + (c - 1)
                preds.append(model.sparse_head[c](h[:, pos, :]).argmax(-1))
            preds = torch.stack(preds, 1)
            tf_joint += (preds == tsem).all(1).float().sum().item()

            N += len(idxs)
            if (st // args.bs) % 10 == 0:
                print(f"  batch {st//args.bs}/{args.nb}  N={N}", flush=True)

    print(f"\n==== COBRA final diagnostic (N={N} test) ====", flush=True)
    print("(A) ITEM-level R@K by beam width:")
    for nb in (20, 32):
        print(f"    n_beam={nb}:  R@5={hit_item[nb][5]/N:.4f}  R@10={hit_item[nb][10]/N:.4f}")
    print("(B) GENERATION sem-id recall + teacher-forced ceiling:")
    print("    " + "  ".join(f"R@{k}={gen_hit[k]/N:.4f}" for k in (1, 5, 10, 20, 32)))
    print(f"    teacher-forced JOINT top-1 = {tf_joint/N:.4f}")
    print("READS: (A) n_beam=32 >> 20 -> widen beams (cheap). "
          "(B) gen R@32 << tf top-1 -> exposure bias (retrain).")


if __name__ == "__main__":
    main()
