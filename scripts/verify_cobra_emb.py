"""
验证COBRA embedding的结构是否正确
"""
import torch
import sys
sys.path.insert(0, '.')

from genrec.models.cobra import Cobra

def verify_cobra_emb():
    """
    验证cobra_emb在训练和生成时的输出结构
    """
    C = 3  # codebooks
    T = 5  # items
    B = 1  # batch size
    D = 768  # hidden dim
    vocab_size = 256

    # 创建模型
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

    # 模拟input_ids (完整序列)
    input_ids = torch.randint(0, vocab_size, (B, T * C))
    print(f"input_ids shape: {input_ids.shape}")

    # 模拟vecs
    vecs = torch.randn(B, T, D)
    print(f"vecs shape: {vecs.shape}")

    # ========== 训练时 (完整序列) ==========
    print("\n" + "="*60)
    print("训练时 (完整序列)")
    print("="*60)

    n_complete = T
    seq_mask = (input_ids != pad_id)
    seq_mask_interleaved = model.interleave_seq_mask(seq_mask, C, n_complete_items=n_complete)
    emb = model.cobra_emb(input_ids, vecs, seq_mask_interleaved, n_complete_items=n_complete)

    print(f"seq_mask_interleaved shape: {seq_mask_interleaved.shape}")
    print(f"emb shape: {emb.shape}")
    print(f"期望emb shape: (1, {T * (C + 1)}, {D}) = (1, {T*(C+1)}, {D})")

    # ========== 生成时 c=0 (完整序列) ==========
    print("\n" + "="*60)
    print("生成时 c=0 (完整序列, 和训练一样)")
    print("="*60)

    # 和训练一样
    n_complete = T
    seq_mask = (input_ids != pad_id)
    seq_mask_interleaved = model.interleave_seq_mask(seq_mask, C, n_complete_items=n_complete)
    emb = model.cobra_emb(input_ids, vecs, seq_mask_interleaved, n_complete_items=n_complete)

    print(f"input_ids长度: {input_ids.shape[1]}")
    print(f"seq_mask_interleaved shape: {seq_mask_interleaved.shape}")
    print(f"emb shape: {emb.shape}")
    print(f"last_pos (用于预测): {seq_mask_interleaved.sum() - 1}")

    # ========== 生成时 c=1 (追加1个token) ==========
    print("\n" + "="*60)
    print("生成时 c=1 (追加1个生成的token)")
    print("="*60)

    # 追加一个生成的token
    new_token = torch.randint(0, vocab_size, (B, 1))
    input_ids_c1 = torch.cat([input_ids, new_token], dim=1)

    n_complete = T  # 仍然只有T个完整items
    seq_mask = (input_ids_c1 != pad_id)
    seq_mask_interleaved = model.interleave_seq_mask(seq_mask, C, n_complete_items=n_complete)
    emb = model.cobra_emb(input_ids_c1, vecs, seq_mask_interleaved, n_complete_items=n_complete)

    print(f"input_ids长度: {input_ids_c1.shape[1]}")
    print(f"seq_mask_interleaved shape: {seq_mask_interleaved.shape}")
    print(f"emb shape: {emb.shape}")
    print(f"期望emb shape: (1, {T*(C+1) + 1}, {D})")
    print(f"last_pos (用于预测): {seq_mask_interleaved.sum() - 1}")

    # 检查结构
    # 应该是: [e0^0, e0^1, e0^2, v0, e1^0, ..., e4^2, v4, e5^0(new)]
    expected_len = T * (C + 1) + 1
    actual_len = emb.shape[1]
    print(f"长度检查: 期望={expected_len}, 实际={actual_len}, 匹配={expected_len == actual_len}")

    # ========== 生成时 c=2 (追加2个tokens) ==========
    print("\n" + "="*60)
    print("生成时 c=2 (追加2个生成的tokens)")
    print("="*60)

    # 追加两个生成的tokens
    new_tokens = torch.randint(0, vocab_size, (B, 2))
    input_ids_c2 = torch.cat([input_ids, new_tokens], dim=1)

    n_complete = T
    seq_mask = (input_ids_c2 != pad_id)
    seq_mask_interleaved = model.interleave_seq_mask(seq_mask, C, n_complete_items=n_complete)
    emb = model.cobra_emb(input_ids_c2, vecs, seq_mask_interleaved, n_complete_items=n_complete)

    print(f"input_ids长度: {input_ids_c2.shape[1]}")
    print(f"seq_mask_interleaved shape: {seq_mask_interleaved.shape}")
    print(f"emb shape: {emb.shape}")
    print(f"期望emb shape: (1, {T*(C+1) + 2}, {D})")
    print(f"last_pos (用于预测): {seq_mask_interleaved.sum() - 1}")

    expected_len = T * (C + 1) + 2
    actual_len = emb.shape[1]
    print(f"长度检查: 期望={expected_len}, 实际={actual_len}, 匹配={expected_len == actual_len}")

    # ========== 验证decoder能否看到之前的token ==========
    print("\n" + "="*60)
    print("验证decoder是否能看到之前生成的token")
    print("="*60)

    # 用一个简单的测试：检查emb的最后几个位置是否有值（非零）
    print(f"\nc=1时emb最后3个位置的norm:")
    emb_c1 = model.cobra_emb(input_ids_c1, vecs,
        model.interleave_seq_mask((input_ids_c1 != pad_id), C, n_complete_items=T),
        n_complete_items=T)
    for i in range(-3, 0):
        norm = emb_c1[0, i].norm().item()
        print(f"  位置{emb_c1.shape[1]+i}: norm={norm:.4f}")

    print(f"\nc=2时emb最后4个位置的norm:")
    emb_c2 = model.cobra_emb(input_ids_c2, vecs,
        model.interleave_seq_mask((input_ids_c2 != pad_id), C, n_complete_items=T),
        n_complete_items=T)
    for i in range(-4, 0):
        norm = emb_c2[0, i].norm().item()
        print(f"  位置{emb_c2.shape[1]+i}: norm={norm:.4f}")


if __name__ == "__main__":
    verify_cobra_emb()
