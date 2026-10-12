"""
验证COBRA训练和生成时使用的hidden state是否一致
"""
import torch
import sys
sys.path.insert(0, '.')

from genrec.models.cobra import Cobra

def verify_hidden_states():
    """
    用相同的输入，比较训练时和生成时使用的hidden state位置
    """
    torch.manual_seed(42)

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

    # 创建固定的输入
    # 假设我们有T个完整items的历史
    input_ids = torch.randint(0, vocab_size, (B, T * C))
    encoder_input_ids = torch.randint(0, 1000, (B, T, 64))  # tokenized text

    print(f"input_ids shape: {input_ids.shape}")
    print(f"encoder_input_ids shape: {encoder_input_ids.shape}")

    # ========== 训练时的forward ==========
    print("\n" + "="*60)
    print("训练时的forward")
    print("="*60)

    with torch.no_grad():
        vecs = model.encoder(encoder_input_ids)  # (B, T, D)

        seq_mask = (input_ids != pad_id)
        seq_mask_interleaved = model.interleave_seq_mask(seq_mask, C)
        emb = model.cobra_emb(input_ids, vecs, seq_mask_interleaved)
        h_train = model.decoder(emb, tgt_key_padding_mask=~seq_mask_interleaved)

        print(f"h_train shape: {h_train.shape}")  # (B, T*(C+1), D)

    # 训练时预测最后一个item (e4) 的各个codebook时用的位置
    # 因为训练是预测e1, e2, e3, e4，我们看预测e4时用的位置
    print("\n训练时预测e4的位置:")
    t = T - 1  # 预测第4个item (0-indexed是4，但我们从1开始预测，所以t=4对应预测e4)

    # c=0: 从v3预测e4^0
    pos_c0_train = (t - 1) * (C + 1) + C  # = 3 * 4 + 3 = 15
    print(f"  c=0: 从位置 {pos_c0_train} (v{t-1}) 预测 e{t}^0")

    # c=1: 从e4^0预测e4^1
    pos_c1_train = t * (C + 1) + 0  # = 4 * 4 + 0 = 16
    print(f"  c=1: 从位置 {pos_c1_train} (e{t}^0) 预测 e{t}^1")

    # c=2: 从e4^1预测e4^2
    pos_c2_train = t * (C + 1) + 1  # = 4 * 4 + 1 = 17
    print(f"  c=2: 从位置 {pos_c2_train} (e{t}^1) 预测 e{t}^2")

    h_c0_train = h_train[0, pos_c0_train, :]
    h_c1_train = h_train[0, pos_c1_train, :]
    h_c2_train = h_train[0, pos_c2_train, :]

    # ========== 生成时模拟 ==========
    print("\n" + "="*60)
    print("生成时模拟 (预测新item e5)")
    print("="*60)

    # 但是！生成时我们没有e5的ground truth，所以我们需要:
    # 1. 用历史(e0-e4)预测e5^0
    # 2. 追加e5^0后预测e5^1
    # 3. 追加e5^1后预测e5^2

    # 让我们模拟这个过程，但用e4的ground truth来验证位置是否一致
    # 也就是说，我们假装在生成时预测e4（使用e0-e3作为历史）

    # 历史: e0, e1, e2, e3 (前4个items)
    input_ids_gen = input_ids[:, :((T-1) * C)]  # (B, 12)
    encoder_input_ids_gen = encoder_input_ids[:, :(T-1), :]  # (B, 4, 64)

    print(f"\n生成时的历史: {T-1}个items")
    print(f"input_ids_gen shape: {input_ids_gen.shape}")

    with torch.no_grad():
        vecs_gen = model.encoder(encoder_input_ids_gen)  # (B, T-1, D)
        T_gen = vecs_gen.size(1)  # = T-1 = 4

        # ===== c=0 =====
        n_complete = T_gen
        seq_mask = (input_ids_gen != pad_id)
        seq_mask_interleaved = model.interleave_seq_mask(seq_mask, C, n_complete_items=n_complete)
        emb = model.cobra_emb(input_ids_gen, vecs_gen, seq_mask_interleaved, n_complete_items=n_complete)
        h_gen = model.decoder(emb, tgt_key_padding_mask=~seq_mask_interleaved)

        last_pos_c0 = seq_mask_interleaved.sum() - 1
        print(f"\nc=0: input长度={input_ids_gen.shape[1]}, h长度={h_gen.shape[1]}, last_pos={last_pos_c0}")
        h_c0_gen = h_gen[0, last_pos_c0, :]

        # ===== c=1: 追加e4^0 =====
        e4_0 = input_ids[:, (T-1)*C : (T-1)*C + 1]  # e4^0 from ground truth
        input_ids_c1 = torch.cat([input_ids_gen, e4_0], dim=1)

        seq_mask = (input_ids_c1 != pad_id)
        seq_mask_interleaved = model.interleave_seq_mask(seq_mask, C, n_complete_items=n_complete)
        emb = model.cobra_emb(input_ids_c1, vecs_gen, seq_mask_interleaved, n_complete_items=n_complete)
        h_gen = model.decoder(emb, tgt_key_padding_mask=~seq_mask_interleaved)

        last_pos_c1 = seq_mask_interleaved.sum() - 1
        print(f"c=1: input长度={input_ids_c1.shape[1]}, h长度={h_gen.shape[1]}, last_pos={last_pos_c1}")
        h_c1_gen = h_gen[0, last_pos_c1, :]

        # ===== c=2: 追加e4^0, e4^1 =====
        e4_01 = input_ids[:, (T-1)*C : (T-1)*C + 2]  # e4^0, e4^1 from ground truth
        input_ids_c2 = torch.cat([input_ids_gen, e4_01], dim=1)

        seq_mask = (input_ids_c2 != pad_id)
        seq_mask_interleaved = model.interleave_seq_mask(seq_mask, C, n_complete_items=n_complete)
        emb = model.cobra_emb(input_ids_c2, vecs_gen, seq_mask_interleaved, n_complete_items=n_complete)
        h_gen = model.decoder(emb, tgt_key_padding_mask=~seq_mask_interleaved)

        last_pos_c2 = seq_mask_interleaved.sum() - 1
        print(f"c=2: input长度={input_ids_c2.shape[1]}, h长度={h_gen.shape[1]}, last_pos={last_pos_c2}")
        h_c2_gen = h_gen[0, last_pos_c2, :]

    # ========== 比较hidden states ==========
    print("\n" + "="*60)
    print("比较训练和生成时的hidden states")
    print("="*60)

    # 注意：由于历史长度不同(训练用5个items，生成用4个items)，
    # hidden states不会完全相同。但我们可以检查它们的相关性。

    cos_sim_c0 = torch.nn.functional.cosine_similarity(h_c0_train.unsqueeze(0), h_c0_gen.unsqueeze(0))
    cos_sim_c1 = torch.nn.functional.cosine_similarity(h_c1_train.unsqueeze(0), h_c1_gen.unsqueeze(0))
    cos_sim_c2 = torch.nn.functional.cosine_similarity(h_c2_train.unsqueeze(0), h_c2_gen.unsqueeze(0))

    print(f"\n由于历史长度不同，hidden states不会完全相同")
    print(f"但我们可以验证位置选择是否正确:\n")
    print(f"训练 vs 生成 cosine similarity:")
    print(f"  c=0: {cos_sim_c0.item():.4f}")
    print(f"  c=1: {cos_sim_c1.item():.4f}")
    print(f"  c=2: {cos_sim_c2.item():.4f}")

    # ========== 更重要的测试：相同输入相同输出 ==========
    print("\n" + "="*60)
    print("关键测试：用完全相同的输入，生成时和训练时的logits是否一致")
    print("="*60)

    # 用完整的5个items作为输入，比较：
    # 1. 训练时预测e4各codebook的logits
    # 2. 生成时（用e0-e3作为历史 + ground truth e4^0, e4^1）的logits

    # 但这里有个问题：训练和生成的历史长度不同
    # 让我们用一个不同的测试：用完全相同的序列，比较取的位置是否得到相同的logits

    print("\n使用完整序列测试位置一致性:")

    with torch.no_grad():
        vecs = model.encoder(encoder_input_ids)

        # 训练方式：一次forward
        seq_mask = (input_ids != pad_id)
        seq_mask_interleaved = model.interleave_seq_mask(seq_mask, C)
        emb = model.cobra_emb(input_ids, vecs, seq_mask_interleaved)
        h = model.decoder(emb, tgt_key_padding_mask=~seq_mask_interleaved)

        # 预测e4时的logits (训练方式)
        logits_c0_train = model.sparse_head[0](h[:, pos_c0_train, :])
        logits_c1_train = model.sparse_head[1](h[:, pos_c1_train, :])
        logits_c2_train = model.sparse_head[2](h[:, pos_c2_train, :])

        print(f"训练方式:")
        print(f"  c=0 logits from pos {pos_c0_train}: top5={logits_c0_train.topk(5).indices[0].tolist()}")
        print(f"  c=1 logits from pos {pos_c1_train}: top5={logits_c1_train.topk(5).indices[0].tolist()}")
        print(f"  c=2 logits from pos {pos_c2_train}: top5={logits_c2_train.topk(5).indices[0].tolist()}")

        # 生成方式：用最后位置
        # 对于c=0: 用前4个items + 预测e4^0
        # 这里我们用完整序列但假装只看到前4个items

        # 实际上，让我们直接用last_pos来取
        seq_lens = seq_mask_interleaved.sum(dim=1)
        last_pos = seq_lens - 1  # = 19 (v4的位置)

        # 但训练时c=0用的是pos 15 (v3)，不是19
        # 因为训练时预测e4^0是从v3预测的

        # 这说明训练和生成的语义不同！
        # 训练时：用v_{t-1}预测e_t^0
        # 生成时：用v_{T-1}预测e_T^0 (T是历史长度)

        print(f"\n注意：训练时预测e4^0用的位置是{pos_c0_train}(v3)")
        print(f"      但如果生成时用5个items作为历史，last_pos是{last_pos.item()}(v4)")
        print(f"      这说明生成时应该用T-1个items作为历史来预测第T个item")


if __name__ == "__main__":
    verify_hidden_states()
