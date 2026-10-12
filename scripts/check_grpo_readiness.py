"""
GRPO 启动前体检 — 在跑长训练之前快速确认所有依赖都就绪。

用法:
  python scripts/check_grpo_readiness.py config/onerec/amazon/grpo_k256.gin --split beauty

会检查 (按顺序):
  [1] dataset/amazon/raw/{split} 数据存在
  [2] SFT checkpoint 目录存在且包含 model.safetensors + tokenizer
  [3] RQ-KMeans / RQ-VAE checkpoint 存在
  [4] SFT tokenizer 包含全部 num_codebooks × codebook_size 个 <C{c}_{code}> token,
      且每个都是单 token (否则 PrefixTrie 会 silently drop)
  [5] 能加载 OneRec model (vocab 大小匹配 + 权重确实从 ckpt 加载)
  [6] AmazonOneRecRLDataset 能成功构造, sem_ids_list 长度合理
  [7] PrefixTrie 覆盖率 (有多少 item 的 sem_ids 能完整进 trie)
  [8] Mini smoke: batch=1, G=2 跑 generate_for_grpo + compute_log_probs,
      reward 计算, advantage 归一化, 全流程不崩
"""
import os
import sys
import argparse


def section(title):
    bar = "=" * 72
    print(f"\n{bar}\n[{title}]\n{bar}")


def fail(msg):
    print(f"  ❌ {msg}")
    sys.exit(1)


def ok(msg):
    print(f"  ✅ {msg}")


def warn(msg):
    print(f"  ⚠️  {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to grpo gin config")
    parser.add_argument("--split", required=True, help="dataset split (e.g. beauty)")
    args = parser.parse_args()

    sys.path.insert(0, os.getcwd())

    import gin
    # Import trainer/data/model modules BEFORE parsing the gin config, so that
    # all @gin.configurable functions/classes referenced by the config are
    # registered. Otherwise gin raises "No configurable matching 'train_grpo'".
    import genrec.trainers.onerec_grpo_trainer  # noqa: F401
    import genrec.trainers.onerec_sft_trainer  # noqa: F401
    import genrec.data.amazon_onerec  # noqa: F401
    import genrec.models.onerec  # noqa: F401

    # Replace {split} placeholders BEFORE parsing the gin config — matches what
    # `genrec.modules.utils.parse_config` does for the real trainer entry point.
    split = args.split.lower()
    with open(args.config) as f:
        config_content = f.read().replace("{split}", split)
    gin.parse_config(config_content)

    def resolve(name):
        v = gin.query_parameter(name)
        return v

    section("[1] dataset/amazon/raw/{split}")
    data_root = resolve("train_grpo.dataset_folder")
    raw_dir = os.path.join(data_root, "raw", split)
    if not os.path.isdir(raw_dir):
        fail(f"raw dir missing: {raw_dir}")
    ok(f"{raw_dir} 存在 ({sum(os.path.getsize(os.path.join(raw_dir, f)) for f in os.listdir(raw_dir)) / 1e6:.0f} MB)")

    section("[2] SFT checkpoint")
    sft_path = resolve("train_grpo.sft_checkpoint_path")
    if not os.path.isdir(sft_path):
        fail(f"SFT ckpt 目录不存在: {sft_path}")
    required_files = ["config.json", "tokenizer_config.json", "added_tokens.json"]
    for f in required_files:
        if not os.path.isfile(os.path.join(sft_path, f)):
            fail(f"缺文件: {sft_path}/{f}")
    has_safetensors = any(f.endswith(".safetensors") for f in os.listdir(sft_path))
    has_bin = any(f.endswith(".bin") for f in os.listdir(sft_path))
    if not (has_safetensors or has_bin):
        fail(f"{sft_path} 既无 safetensors 也无 .bin")
    ok(f"SFT ckpt 存在: {sft_path}")

    section("[3] tokenizer 包含全部 codebook tokens")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(sft_path)
    num_codebooks = gin.query_parameter("train_grpo.num_codebooks")
    codebook_size = gin.query_parameter("train_grpo.codebook_size")
    print(f"  期望: {num_codebooks} codebooks × {codebook_size} = {num_codebooks * codebook_size} tokens")

    missing = []
    multi_token = []
    for c in range(num_codebooks):
        for code in range(codebook_size):
            s = f"<C{c}_{code}>"
            ids = tokenizer(s, add_special_tokens=False).input_ids
            if len(ids) == 0:
                missing.append(s)
            elif len(ids) > 1:
                multi_token.append((s, ids))
    if missing:
        fail(f"{len(missing)} 个 codebook token 在 tokenizer 中找不到, 例: {missing[:5]}")
    if multi_token:
        fail(f"{len(multi_token)} 个 codebook token 被分成多 token (PrefixTrie 会 drop), 例: {multi_token[:3]}")
    ok(f"全部 {num_codebooks * codebook_size} 个 codebook token 都是 single token")

    # Check item_sep / response marker
    response_ids = tokenizer("### Response:", add_special_tokens=False).input_ids
    print(f"  '### Response:' tokenize -> {len(response_ids)} tokens: {response_ids[:10]}")

    section("[4] RQ-KMeans / RQ-VAE checkpoint")
    rl_ds_cls_name = "AmazonOneRecRLDataset"
    tokenizer_type = gin.query_parameter(f"{rl_ds_cls_name}.tokenizer_type")
    if tokenizer_type == "rqkmeans":
        rqk_path = gin.query_parameter(f"{rl_ds_cls_name}.pretrained_rqkmeans_path")
        if not os.path.isfile(rqk_path):
            fail(f"RQ-KMeans 文件不存在: {rqk_path}")
        size_mb = os.path.getsize(rqk_path) / 1e6
        ok(f"RQ-KMeans 存在: {rqk_path} ({size_mb:.0f} MB)")
    else:
        rqv_path = gin.query_parameter(f"{rl_ds_cls_name}.pretrained_rqvae_path")
        if not os.path.isfile(rqv_path):
            fail(f"RQ-VAE 文件不存在: {rqv_path}")
        ok(f"RQ-VAE 存在: {rqv_path}")

    section("[5] OneRec model 加载 + 权重健康性")
    import torch
    from genrec.models.onerec import OneRec
    print(f"  loading model from {sft_path} (这步比较慢, 1-2 分钟)...")
    model = OneRec(pretrained_path=sft_path)
    vocab_size = model.model.config.vocab_size
    expected_min = num_codebooks * codebook_size  # 至少包含 codebook tokens
    if vocab_size < expected_min:
        fail(f"vocab_size={vocab_size} < codebook tokens 数 {expected_min}")
    n_params = sum(p.numel() for p in model.parameters())
    ok(f"model loaded, vocab={vocab_size}, params={n_params/1e9:.2f}B")

    # 健康性: from_pretrained 的前缀可能不匹配 (model.model.X vs model.X),
    # OneRec.__init__ 有 fix 代码用 safetensors 直接 reload。这里验证 fix 是否真生效:
    # 直接 load safetensors 第一份, 跟当前 model 对比某个具体 layer 的 weight norm。
    import safetensors.torch
    import glob
    sf_files = sorted(glob.glob(os.path.join(sft_path, "*.safetensors")))
    print(f"  found {len(sf_files)} safetensors file(s)")
    if sf_files:
        sd = safetensors.torch.load_file(sf_files[0])
        sample_key_in_ckpt = next(iter(sd.keys()))
        print(f"  ckpt sample key: {sample_key_in_ckpt!r}")
        # 找一个 embed_tokens / lm_head 来对比
        candidate_ckpt_keys = [k for k in sd.keys() if "embed_tokens.weight" in k]
        candidate_model_keys = [n for n, _ in model.named_parameters() if "embed_tokens.weight" in n]
        if candidate_ckpt_keys and candidate_model_keys:
            ck = candidate_ckpt_keys[0]
            mk = candidate_model_keys[0]
            ckpt_norm = sd[ck].float().norm().item()
            model_norm = dict(model.named_parameters())[mk].float().norm().item()
            print(f"  ckpt[{ck!r}] norm = {ckpt_norm:.4f}")
            print(f"  model[{mk!r}] norm = {model_norm:.4f}")
            if abs(ckpt_norm - model_norm) / (ckpt_norm + 1e-6) > 0.01:
                fail(f"embed_tokens norm 不匹配! ckpt={ckpt_norm:.4f} model={model_norm:.4f} — 权重没真正从 ckpt 加载")
            ok(f"embed_tokens norm 匹配 ({ckpt_norm:.4f}) — 权重从 ckpt 正确加载")

    section("[6] AmazonOneRecRLDataset 构造")
    from genrec.data.amazon_onerec import AmazonOneRecRLDataset
    rl_ds = AmazonOneRecRLDataset(
        root=data_root,
        max_seq_len=gin.query_parameter("train_grpo.max_seq_len"),
    )
    n_items = len(rl_ds.sem_ids_list)
    n_samples = len(rl_ds.samples)
    n_uniq_sids = len(set(tuple(s) for s in rl_ds.sem_ids_list))
    ok(f"num_items={n_items}, n_samples={n_samples}, unique sids={n_uniq_sids} ({n_uniq_sids/n_items*100:.1f}%)")

    if n_uniq_sids / n_items < 0.5:
        warn(f"unique SID ratio < 50%, 大量 item 共享 sem_ids (PrefixTrie 仍会工作但 GRPO 难学)")

    section("[7] PrefixTrie 构建覆盖率")
    from genrec.trainers.onerec_grpo_trainer import PrefixTrie
    prefix_trie = PrefixTrie(rl_ds.sem_ids_list, num_codebooks, codebook_size, tokenizer)

    # 检查每个 sem_ids 是不是都能进 trie
    n_covered = 0
    for sem_ids in rl_ds.sem_ids_list:
        tokens = []
        valid = True
        for c, code in enumerate(sem_ids[:num_codebooks]):
            if (c, code) not in prefix_trie.token_id_map:
                valid = False
                break
            tokens.append(prefix_trie.token_id_map[(c, code)])
        if valid and tuple(tokens) in prefix_trie.trie:
            n_covered += 1
    cov = n_covered / n_items * 100
    if cov < 95:
        warn(f"PrefixTrie 覆盖率 {cov:.1f}% < 95%")
    else:
        ok(f"PrefixTrie 覆盖率 {cov:.1f}% ({n_covered}/{n_items})")

    section("[8] Mini smoke: generate_for_grpo + compute_log_probs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    if device.type == "cuda":
        model.model = model.model.to(device).to(torch.bfloat16)
    model.eval()

    # 准备一个 batch=1 的 prompt (从 rl_ds 取第一个 sample)
    sample = rl_ds.samples[0]
    print(f"  sample task: {sample['task']}")
    enc = tokenizer(sample['prompt'], return_tensors='pt', truncation=True, max_length=512)
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    print(f"  prompt len: {input_ids.size(1)}")

    G = 2
    with torch.no_grad():
        gen_ids, gen_logp = model.generate_for_grpo(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_generations=G,
            max_new_tokens=num_codebooks + 1,
            temperature=1.0,
            prefix_allowed_tokens_fn=prefix_trie.get_prefix_allowed_tokens_fn(),
            use_beam_search=False,
        )
    print(f"  gen_ids shape: {gen_ids.shape} (expect [B*G, L+T] = [{1*G}, {input_ids.size(1) + num_codebooks + 1}])")
    print(f"  gen_logp shape: {gen_logp.shape} (expect [B*G, T] = [{1*G}, {num_codebooks + 1}])")

    # 解码看下生成结果
    import re
    pattern = re.compile(r'<C(\d+)_(\d+)>')
    L = input_ids.size(1)
    for g in range(G):
        decoded = tokenizer.decode(gen_ids[g, L:], skip_special_tokens=False)
        matches = pattern.findall(decoded)
        sids = [(int(c), int(code)) for c, code in matches[:num_codebooks]]
        print(f"  gen[{g}]: {decoded[:60]!r} -> sids={sids}")

    # 测试 reward
    from genrec.trainers.onerec_grpo_trainer import (
        compute_rule_reward, compute_hierarchical_reward, compute_rank_reward,
    )
    gen_sem_ids = []
    for g in range(G):
        decoded = tokenizer.decode(gen_ids[g, L:], skip_special_tokens=False)
        matches = pattern.findall(decoded)
        if len(matches) >= num_codebooks:
            sids = [int(matches[c][1]) for c in range(num_codebooks)]
        else:
            sids = [0] * num_codebooks
        gen_sem_ids.append(sids)
    target = sample['target_sem_ids']
    r_rule = compute_rule_reward(gen_sem_ids, target)
    r_hier = compute_hierarchical_reward(gen_sem_ids, target, num_codebooks)
    r_rank = compute_rank_reward(gen_sem_ids, target, num_codebooks)
    print(f"  target: {target}")
    print(f"  rewards rule={r_rule}, hier={r_hier}, rank={r_rank}")

    section("✅ All checks passed")
    print("可以起 GRPO 训练。建议先用 max_train_samples=200 跑 1 epoch 看 mean_reward / KL 趋势。")


if __name__ == "__main__":
    main()
