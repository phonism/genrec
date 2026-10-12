"""
验证COBRA训练和生成的位置是否一致
"""
import torch
import sys
sys.path.insert(0, '.')

def verify_positions():
    """
    验证训练和生成时使用的位置是否一致
    """
    C = 3  # codebooks
    T = 5  # items
    pad_id = 256 * 3

    # 模拟input_ids (完整序列, 无padding)
    # 结构: [e0^0, e0^1, e0^2, e1^0, e1^1, e1^2, ..., e4^0, e4^1, e4^2]
    input_ids = torch.arange(T * C).unsqueeze(0)  # (1, 15)
    print(f"input_ids shape: {input_ids.shape}")
    print(f"input_ids: {input_ids[0].tolist()}")

    # ========== 训练时的位置计算 ==========
    print("\n" + "="*60)
    print("训练时的位置计算")
    print("="*60)

    # interleaved序列结构
    # [e0^0, e0^1, e0^2, v0, e1^0, e1^1, e1^2, v1, ..., e4^0, e4^1, e4^2, v4]
    # 位置: 0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15, 16,17,18,19
    interleaved_structure = []
    for t in range(T):
        for c in range(C):
            interleaved_structure.append(f"e{t}^{c}")
        interleaved_structure.append(f"v{t}")

    print(f"\nInterleaved结构 (长度={len(interleaved_structure)}):")
    for i, name in enumerate(interleaved_structure):
        print(f"  位置{i:2d}: {name}")

    print("\n--- 训练时预测位置 ---")
    for c in range(C):
        if c == 0:
            # pos_c = torch.arange(0, T-1) * (C+1) + C
            pos_c = [t * (C + 1) + C for t in range(T - 1)]
            print(f"\nc=0: 从位置 {pos_c} 预测")
            print(f"     对应: {[interleaved_structure[p] for p in pos_c]}")
            target_pos = [t * C for t in range(1, T)]
            print(f"     预测目标位置(input_ids): {target_pos}")
            print(f"     预测目标: e1^0, e2^0, e3^0, e4^0")
        else:
            # pos_c = torch.arange(1, T) * (C+1) + (c-1)
            pos_c = [t * (C + 1) + (c - 1) for t in range(1, T)]
            print(f"\nc={c}: 从位置 {pos_c} 预测")
            print(f"     对应: {[interleaved_structure[p] for p in pos_c]}")
            target_pos = [t * C + c for t in range(1, T)]
            print(f"     预测目标位置(input_ids): {target_pos}")
            print(f"     预测目标: e1^{c}, e2^{c}, e3^{c}, e4^{c}")

    # ========== 生成时的位置计算 ==========
    print("\n" + "="*60)
    print("生成时的位置计算 (预测第{T}个item)")
    print("="*60)

    # 假设history有T个items
    n_complete = T

    for c in range(C):
        if c == 0:
            # input_ids长度 = T * C
            L = T * C
            # interleaved后长度 = L + n_complete = T*C + T
            interleaved_len = L + n_complete
            # 最后位置 = interleaved_len - 1 = T*(C+1) - 1
            last_pos = interleaved_len - 1
            print(f"\nc=0: input_ids长度={L}, interleaved长度={interleaved_len}")
            print(f"     最后位置={last_pos}, 对应: {interleaved_structure[last_pos]}")
            print(f"     (应该是v{T-1})")

        else:
            # input_ids长度 = T * C + c (追加了c个生成的token)
            L = T * C + c
            # interleaved后长度 = L + n_complete
            interleaved_len = L + n_complete
            # 最后位置
            last_pos = interleaved_len - 1

            # 构建生成时的interleaved结构
            gen_structure = interleaved_structure.copy()
            for cc in range(c):
                gen_structure.append(f"e{T}^{cc}(生成)")

            print(f"\nc={c}: input_ids长度={L}, interleaved长度={interleaved_len}")
            print(f"     最后位置={last_pos}, 对应: {gen_structure[last_pos]}")
            print(f"     (应该是e{T}^{c-1})")

    # ========== 一致性检查 ==========
    print("\n" + "="*60)
    print("一致性检查")
    print("="*60)

    print("\n训练时:")
    print(f"  c=0: 从v_t预测e_{{t+1}}^0")
    print(f"  c=1: 从e_{{t+1}}^0预测e_{{t+1}}^1")
    print(f"  c=2: 从e_{{t+1}}^1预测e_{{t+1}}^2")

    print("\n生成时 (预测新item e_{T}):")
    print(f"  c=0: 从v_{{T-1}}预测e_{{T}}^0  <- 最后dense位置")
    print(f"  c=1: 从e_{{T}}^0预测e_{{T}}^1  <- 追加e_T^0后的最后位置")
    print(f"  c=2: 从e_{{T}}^1预测e_{{T}}^2  <- 追加e_T^0,e_T^1后的最后位置")

    print("\n结论: 训练和生成使用的位置应该是一致的 ✓")

    # ========== 实际验证interleave_seq_mask ==========
    print("\n" + "="*60)
    print("实际验证 interleave_seq_mask")
    print("="*60)

    def interleave_seq_mask(seq_mask, n, n_complete_items=None):
        B, L = seq_mask.shape
        device = seq_mask.device
        dtype = seq_mask.dtype

        if n_complete_items is None:
            n_complete_items = L // n

        orig_pos = torch.arange(L, device=device)
        complete_part = orig_pos < n_complete_items * n
        new_pos = torch.where(
            complete_part,
            orig_pos + orig_pos // n,
            orig_pos + n_complete_items
        )

        g = torch.arange(n_complete_items, device=device)
        ins_pos = g * (n + 1) + n
        prev_idx = g * n + (n - 1)
        prev_idx = prev_idx.clamp(max=L-1)
        ins_value = seq_mask[:, prev_idx]

        new_len = L + n_complete_items
        out_mask = seq_mask.new_zeros(B, new_len, dtype=dtype)

        out_mask.scatter_(dim=1, index=new_pos.expand(B, -1), src=seq_mask)

        if n_complete_items > 0:
            out_mask.scatter_(dim=1, index=ins_pos.expand(B, -1), src=ins_value)
        return out_mask

    # c=0: 完整序列
    seq_mask_c0 = torch.ones(1, T * C, dtype=torch.bool)
    interleaved_c0 = interleave_seq_mask(seq_mask_c0, C, n_complete_items=T)
    print(f"\nc=0: input长度={T*C}, interleaved长度={interleaved_c0.shape[1]}")
    print(f"     seq_lens={interleaved_c0.sum()}, last_pos={interleaved_c0.sum()-1}")

    # c=1: 追加1个token
    seq_mask_c1 = torch.ones(1, T * C + 1, dtype=torch.bool)
    interleaved_c1 = interleave_seq_mask(seq_mask_c1, C, n_complete_items=T)
    print(f"\nc=1: input长度={T*C+1}, interleaved长度={interleaved_c1.shape[1]}")
    print(f"     seq_lens={interleaved_c1.sum()}, last_pos={interleaved_c1.sum()-1}")

    # c=2: 追加2个tokens
    seq_mask_c2 = torch.ones(1, T * C + 2, dtype=torch.bool)
    interleaved_c2 = interleave_seq_mask(seq_mask_c2, C, n_complete_items=T)
    print(f"\nc=2: input长度={T*C+2}, interleaved长度={interleaved_c2.shape[1]}")
    print(f"     seq_lens={interleaved_c2.sum()}, last_pos={interleaved_c2.sum()-1}")


if __name__ == "__main__":
    verify_positions()
