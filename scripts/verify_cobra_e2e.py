"""
端到端验证：训练时的logits和生成时的logits是否一致
"""
import torch
import sys
sys.path.insert(0, '.')

from genrec.models.cobra import Cobra

def verify_e2e():
    """
    关键测试：训练时最后一个位置的logits和生成时的logits是否一致
    """
    torch.manual_seed(42)

    C = 3
    T = 5  # 历史items数
    B = 1
    D = 768
    vocab_size = 256

    model = Cobra(
        encoder_n_layers=1,
        encoder_hidden_dim=768,
        encoder_num_heads=8,
        encoder_vocab_size=32128,
        id_vocab_size=vocab_size,
        n_codebooks=C,
        d_model=D,
        max_len=1024,
    )
    model.eval()

    pad_id = vocab_size * C

    # 创建输入数据
    # 假设历史有T个items
    input_ids = torch.randint(0, vocab_size, (B, T * C))
    encoder_input_ids = torch.randint(0, 1000, (B, T, 64))

    print(f"历史items数: {T}")
    print(f"input_ids shape: {input_ids.shape}")
    print(f"encoder_input_ids shape: {encoder_input_ids.shape}")

    # ========== 训练时的forward ==========
    print("\n" + "="*60)
    print("训练时的forward (预测最后一个item e_{T-1})")
    print("="*60)

    with torch.no_grad():
        vecs = model.encoder(encoder_input_ids)
        seq_mask = (input_ids != pad_id)
        seq_mask_interleaved = model.interleave_seq_mask(seq_mask, C)
        emb = model.cobra_emb(input_ids, vecs, seq_mask_interleaved)
        h = model.decoder(emb, tgt_key_padding_mask=~seq_mask_interleaved)

        # 训练时预测最后一个item (e_{T-1}) 的logits
        # c=0: 从 v_{T-2} 预测 e_{T-1}^0
        pos_c0 = (T - 2) * (C + 1) + C  # v_{T-2}的位置
        logits_c0_train = model.sparse_head[0](h[:, pos_c0, :])

        # c=1: 从 e_{T-1}^0 预测 e_{T-1}^1
        pos_c1 = (T - 1) * (C + 1) + 0  # e_{T-1}^0的位置
        logits_c1_train = model.sparse_head[1](h[:, pos_c1, :])

        # c=2: 从 e_{T-1}^1 预测 e_{T-1}^2
        pos_c2 = (T - 1) * (C + 1) + 1  # e_{T-1}^1的位置
        logits_c2_train = model.sparse_head[2](h[:, pos_c2, :])

        print(f"预测e_{T-1}时:")
        print(f"  c=0: 从位置{pos_c0}预测, top5={logits_c0_train.topk(5).indices[0].tolist()}")
        print(f"  c=1: 从位置{pos_c1}预测, top5={logits_c1_train.topk(5).indices[0].tolist()}")
        print(f"  c=2: 从位置{pos_c2}预测, top5={logits_c2_train.topk(5).indices[0].tolist()}")

    # ========== 生成时 (用T-1个items预测第T个item) ==========
    print("\n" + "="*60)
    print("生成时 (用T-1个items作为历史，预测e_{T-1})")
    print("="*60)

    # 用前T-1个items作为历史
    input_ids_gen = input_ids[:, :(T-1)*C]  # 去掉最后一个item
    encoder_input_ids_gen = encoder_input_ids[:, :(T-1), :]

    # 但是！我们需要用ground truth的e_{T-1}^0, e_{T-1}^1来验证c=1, c=2
    e_T1_0 = input_ids[:, (T-1)*C + 0:1+(T-1)*C]  # e_{T-1}^0
    e_T1_1 = input_ids[:, (T-1)*C + 1:2+(T-1)*C]  # e_{T-1}^1

    with torch.no_grad():
        vecs_gen = model.encoder(encoder_input_ids_gen)
        T_gen = vecs_gen.size(1)  # T-1

        # ===== c=0: 预测e_{T-1}^0 =====
        n_complete = T_gen
        seq_mask = (input_ids_gen != pad_id)
        seq_mask_interleaved = model.interleave_seq_mask(seq_mask, C, n_complete_items=n_complete)
        emb = model.cobra_emb(input_ids_gen, vecs_gen, seq_mask_interleaved, n_complete_items=n_complete)
        h_gen = model.decoder(emb, tgt_key_padding_mask=~seq_mask_interleaved)

        last_pos = seq_mask_interleaved.sum() - 1
        logits_c0_gen = model.sparse_head[0](h_gen[:, last_pos, :])
        print(f"\nc=0: 从位置{last_pos}预测, top5={logits_c0_gen.topk(5).indices[0].tolist()}")

        # ===== c=1: 追加e_{T-1}^0后预测e_{T-1}^1 =====
        input_ids_c1 = torch.cat([input_ids_gen, e_T1_0], dim=1)
        seq_mask = (input_ids_c1 != pad_id)
        seq_mask_interleaved = model.interleave_seq_mask(seq_mask, C, n_complete_items=n_complete)
        emb = model.cobra_emb(input_ids_c1, vecs_gen, seq_mask_interleaved, n_complete_items=n_complete)
        h_gen = model.decoder(emb, tgt_key_padding_mask=~seq_mask_interleaved)

        last_pos = seq_mask_interleaved.sum() - 1
        logits_c1_gen = model.sparse_head[1](h_gen[:, last_pos, :])
        print(f"c=1: 从位置{last_pos}预测, top5={logits_c1_gen.topk(5).indices[0].tolist()}")

        # ===== c=2: 追加e_{T-1}^0, e_{T-1}^1后预测e_{T-1}^2 =====
        e_T1_01 = input_ids[:, (T-1)*C:(T-1)*C+2]
        input_ids_c2 = torch.cat([input_ids_gen, e_T1_01], dim=1)
        seq_mask = (input_ids_c2 != pad_id)
        seq_mask_interleaved = model.interleave_seq_mask(seq_mask, C, n_complete_items=n_complete)
        emb = model.cobra_emb(input_ids_c2, vecs_gen, seq_mask_interleaved, n_complete_items=n_complete)
        h_gen = model.decoder(emb, tgt_key_padding_mask=~seq_mask_interleaved)

        last_pos = seq_mask_interleaved.sum() - 1
        logits_c2_gen = model.sparse_head[2](h_gen[:, last_pos, :])
        print(f"c=2: 从位置{last_pos}预测, top5={logits_c2_gen.topk(5).indices[0].tolist()}")

    # ========== 比较 ==========
    print("\n" + "="*60)
    print("比较训练和生成的logits")
    print("="*60)

    diff_c0 = (logits_c0_train - logits_c0_gen).abs().max().item()
    diff_c1 = (logits_c1_train - logits_c1_gen).abs().max().item()
    diff_c2 = (logits_c2_train - logits_c2_gen).abs().max().item()

    cos_c0 = torch.nn.functional.cosine_similarity(logits_c0_train, logits_c0_gen).item()
    cos_c1 = torch.nn.functional.cosine_similarity(logits_c1_train, logits_c1_gen).item()
    cos_c2 = torch.nn.functional.cosine_similarity(logits_c2_train, logits_c2_gen).item()

    print(f"\nc=0: max_diff={diff_c0:.6f}, cosine_sim={cos_c0:.6f}")
    print(f"c=1: max_diff={diff_c1:.6f}, cosine_sim={cos_c1:.6f}")
    print(f"c=2: max_diff={diff_c2:.6f}, cosine_sim={cos_c2:.6f}")

    if cos_c0 > 0.99 and cos_c1 > 0.99 and cos_c2 > 0.99:
        print("\n✓ 训练和生成的logits高度一致!")
    else:
        print("\n✗ 训练和生成的logits不一致，存在问题!")

    # ========== Top-k一致性 ==========
    print("\n" + "="*60)
    print("Top-5预测一致性检查")
    print("="*60)

    top5_c0_train = set(logits_c0_train.topk(5).indices[0].tolist())
    top5_c0_gen = set(logits_c0_gen.topk(5).indices[0].tolist())
    overlap_c0 = len(top5_c0_train & top5_c0_gen)

    top5_c1_train = set(logits_c1_train.topk(5).indices[0].tolist())
    top5_c1_gen = set(logits_c1_gen.topk(5).indices[0].tolist())
    overlap_c1 = len(top5_c1_train & top5_c1_gen)

    top5_c2_train = set(logits_c2_train.topk(5).indices[0].tolist())
    top5_c2_gen = set(logits_c2_gen.topk(5).indices[0].tolist())
    overlap_c2 = len(top5_c2_train & top5_c2_gen)

    print(f"c=0 top5重叠: {overlap_c0}/5")
    print(f"c=1 top5重叠: {overlap_c1}/5")
    print(f"c=2 top5重叠: {overlap_c2}/5")


if __name__ == "__main__":
    verify_e2e()
