#!/usr/bin/env python3
"""
Verify LCRec collate function issues and training pipeline.

This script checks:
1. Whether pad tokens are being masked in labels
2. Whether BOS token causes prompt_len calculation errors
3. Whether tokenization boundaries are consistent
4. Simulates forward pass and checks loss
5. Tests constrained decoding
6. Checks embedding initialization for new tokens

Run: python scripts/verify_lcrec_collate.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

# Config - modify as needed
PRETRAINED_PATH = "/root/workspace/models_hub/Qwen3-1.7B"
NUM_CODEBOOKS = 5
CODEBOOK_SIZE = 256


def add_codebook_tokens(tokenizer, num_codebooks, codebook_size):
    """Add codebook tokens to tokenizer."""
    num_added = 0
    for i in range(num_codebooks):
        for j in range(codebook_size):
            num_added += tokenizer.add_special_tokens(
                {"additional_special_tokens": [f"<C{i}_{j}>"]}
            )
    print(f"Added {num_added} codebook tokens")
    return tokenizer


def test_tokenization_issues(tokenizer):
    print("=" * 60)
    print("TEST 1: Tokenizer Info")
    print("=" * 60)

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print()

    # Test prompt and response
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "User interaction history: 1. <C0_10><C1_20><C2_30><C3_40><C4_50>, "
        "2. <C0_11><C1_21><C2_31><C3_41><C4_51>\n"
        "Predict the next item:\n\n"
        "### Response:"
    )
    response = "<C0_100><C1_200><C2_150><C3_80><C4_90>"
    full_text = prompt + response + tokenizer.eos_token

    print("=" * 60)
    print("TEST 2: BOS Token Issue")
    print("=" * 60)

    full_ids_with_special = tokenizer(full_text).input_ids
    full_ids_no_special = tokenizer(full_text, add_special_tokens=False).input_ids
    prompt_ids_no_special = tokenizer(prompt, add_special_tokens=False).input_ids
    prompt_len = len(prompt_ids_no_special)

    print(f"Full text (with special tokens): {len(full_ids_with_special)} tokens")
    print(f"Full text (no special tokens): {len(full_ids_no_special)} tokens")
    print(f"Prompt (no special tokens): {prompt_len} tokens")

    has_bos = len(full_ids_with_special) > 0 and full_ids_with_special[0] == tokenizer.bos_token_id
    print(f"Does tokenizer add BOS by default? {has_bos}")

    if has_bos:
        print("\n[ISSUE] BOS token is added!")
    else:
        print("\n[OK] No BOS token added by default")

    print()
    print("=" * 60)
    print("TEST 3: Pad Token Masking (Batch Simulation)")
    print("=" * 60)

    short_response = "<C0_1><C1_2><C2_3><C3_4><C4_5>"
    long_response = "<C0_1><C1_2><C2_3><C3_4><C4_5>" * 3

    batch_texts = [
        prompt + short_response + tokenizer.eos_token,
        prompt + long_response + tokenizer.eos_token,
    ]

    encoded_batch = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt',
    )

    input_ids_batch = encoded_batch['input_ids']
    labels_batch = input_ids_batch.clone()

    # FIXED: Mask pad tokens first
    labels_batch[labels_batch == tokenizer.pad_token_id] = -100

    # Then mask prompt
    for i in range(len(batch_texts)):
        labels_batch[i, :prompt_len] = -100

    pad_in_labels_0 = (labels_batch[0] == tokenizer.pad_token_id).sum().item()
    pad_in_labels_1 = (labels_batch[1] == tokenizer.pad_token_id).sum().item()

    print(f"Sample 0 (short): {pad_in_labels_0} pad tokens in labels (should be 0)")
    print(f"Sample 1 (long): {pad_in_labels_1} pad tokens in labels (should be 0)")

    if pad_in_labels_0 == 0 and pad_in_labels_1 == 0:
        print("[OK] Pad tokens are correctly masked")
    else:
        print("[ISSUE] Pad tokens still in labels!")

    return prompt, response, prompt_len


def test_forward_pass(tokenizer, model, prompt, response, prompt_len):
    print()
    print("=" * 60)
    print("TEST 4: Forward Pass & Loss Calculation")
    print("=" * 60)

    device = next(model.parameters()).device
    full_text = prompt + response + tokenizer.eos_token

    # Tokenize
    encoded = tokenizer(
        full_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    # Create labels with masking
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    labels[0, :prompt_len] = -100

    print(f"Input shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Non-masked label positions: {(labels != -100).sum().item()}")

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    loss = outputs.loss.item()
    print(f"\nLoss: {loss:.4f}")

    if loss < 0.1:
        print("[WARNING] Loss is very low - model might be overfitting or labels are wrong")
    elif loss > 10:
        print("[WARNING] Loss is very high - model might not be learning correctly")
    else:
        print("[OK] Loss is in reasonable range")

    # Check logits for response tokens
    logits = outputs.logits[0]  # [seq_len, vocab_size]

    print("\n--- Response Token Predictions ---")
    response_ids = tokenizer(response, add_special_tokens=False).input_ids

    for i, target_id in enumerate(response_ids):
        pos = prompt_len + i
        if pos < logits.size(0):
            # Get top-5 predictions at this position
            probs = F.softmax(logits[pos], dim=-1)
            top5_probs, top5_ids = torch.topk(probs, 5)

            target_token = tokenizer.convert_ids_to_tokens([target_id])[0]
            target_prob = probs[target_id].item()
            target_rank = (probs > probs[target_id]).sum().item() + 1

            top5_tokens = tokenizer.convert_ids_to_tokens(top5_ids.tolist())

            print(f"  Position {pos}: target='{target_token}' (prob={target_prob:.4f}, rank={target_rank})")
            print(f"    Top-5: {list(zip(top5_tokens, [f'{p:.4f}' for p in top5_probs.tolist()]))}")


def test_constrained_decoding(tokenizer, model, prompt):
    print()
    print("=" * 60)
    print("TEST 5: Constrained Decoding")
    print("=" * 60)

    device = next(model.parameters()).device

    # Build allowed tokens per position
    allowed_tokens_per_position = {}
    all_codebook_tokens = set()

    for c in range(NUM_CODEBOOKS):
        allowed_tokens_per_position[c] = set()
        for code in range(CODEBOOK_SIZE):
            token_str = f"<C{c}_{code}>"
            token_ids = tokenizer(token_str, add_special_tokens=False).input_ids
            if len(token_ids) == 1:
                allowed_tokens_per_position[c].add(token_ids[0])
                all_codebook_tokens.add(token_ids[0])

    allowed_tokens_per_position[NUM_CODEBOOKS] = {tokenizer.eos_token_id}

    print(f"Codebook 0 has {len(allowed_tokens_per_position[0])} allowed tokens")
    print(f"Codebook {NUM_CODEBOOKS-1} has {len(allowed_tokens_per_position[NUM_CODEBOOKS-1])} allowed tokens")
    print(f"Total codebook tokens: {len(all_codebook_tokens)}")

    # Get response marker
    response_marker_ids = tokenizer("### Response:", add_special_tokens=False).input_ids
    print(f"Response marker tokens: {response_marker_ids}")

    def get_prefix_allowed_tokens_fn(batch_id, input_ids):
        input_list = input_ids.tolist()

        # Find response marker
        response_start = -1
        for i in range(len(input_list) - len(response_marker_ids) + 1):
            if input_list[i:i + len(response_marker_ids)] == response_marker_ids:
                response_start = i + len(response_marker_ids)
                break

        if response_start == -1:
            num_generated = 0
            for tid in reversed(input_list):
                if tid in all_codebook_tokens:
                    num_generated += 1
                else:
                    break
        else:
            num_generated = len(input_list) - response_start

        pos = min(num_generated, NUM_CODEBOOKS)
        return list(allowed_tokens_per_position[pos])

    # Tokenize prompt
    encoded = tokenizer(prompt, return_tensors='pt').to(device)

    print(f"\nInput prompt length: {encoded['input_ids'].size(1)} tokens")
    print("Generating with constrained decoding...")

    try:
        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                max_new_tokens=NUM_CODEBOOKS + 1,
                num_beams=5,
                num_return_sequences=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=get_prefix_allowed_tokens_fn,
                early_stopping=True,
            )

        print(f"\nGenerated {generated.size(0)} sequences")

        for i in range(min(5, generated.size(0))):
            gen_text = tokenizer.decode(generated[i], skip_special_tokens=False)
            # Extract just the response part
            if "### Response:" in gen_text:
                response_part = gen_text.split("### Response:")[-1].strip()
            else:
                response_part = gen_text[-50:]

            print(f"  Beam {i}: {response_part[:100]}")

        print("\n[OK] Constrained decoding works")

    except Exception as e:
        print(f"\n[ERROR] Constrained decoding failed: {e}")
        import traceback
        traceback.print_exc()


def test_embedding_initialization(tokenizer, model):
    print()
    print("=" * 60)
    print("TEST 6: Embedding Initialization for New Tokens")
    print("=" * 60)

    device = next(model.parameters()).device

    # Get embedding layer
    if hasattr(model, 'model'):
        embed_layer = model.model.embed_tokens
    elif hasattr(model, 'transformer'):
        embed_layer = model.transformer.wte
    else:
        embed_layer = model.get_input_embeddings()

    embed_weight = embed_layer.weight

    print(f"Embedding shape: {embed_weight.shape}")

    # Check codebook token embeddings
    codebook_token_ids = []
    for c in range(NUM_CODEBOOKS):
        for code in range(min(10, CODEBOOK_SIZE)):
            token_str = f"<C{c}_{code}>"
            token_ids = tokenizer(token_str, add_special_tokens=False).input_ids
            if len(token_ids) == 1:
                codebook_token_ids.append(token_ids[0])

    if codebook_token_ids:
        codebook_embeds = embed_weight[codebook_token_ids]

        # Statistics
        mean = codebook_embeds.mean().item()
        std = codebook_embeds.std().item()
        norm = codebook_embeds.norm(dim=1).mean().item()

        print(f"\nCodebook token embeddings (first {len(codebook_token_ids)} tokens):")
        print(f"  Mean: {mean:.6f}")
        print(f"  Std: {std:.6f}")
        print(f"  Avg L2 norm: {norm:.4f}")

        # Compare with regular tokens
        regular_token_ids = list(range(1000, 1050))
        regular_embeds = embed_weight[regular_token_ids]

        reg_mean = regular_embeds.mean().item()
        reg_std = regular_embeds.std().item()
        reg_norm = regular_embeds.norm(dim=1).mean().item()

        print(f"\nRegular token embeddings (for comparison):")
        print(f"  Mean: {reg_mean:.6f}")
        print(f"  Std: {reg_std:.6f}")
        print(f"  Avg L2 norm: {reg_norm:.4f}")

        # Check if embeddings look reasonable
        if abs(std) < 1e-6:
            print("\n[ISSUE] Codebook embeddings have zero variance - not initialized properly!")
        elif norm < 0.01:
            print("\n[ISSUE] Codebook embeddings have very small norm!")
        else:
            print("\n[OK] Codebook embeddings look reasonable")

        # Check if embeddings require grad
        print(f"\nEmbedding requires_grad: {embed_weight.requires_grad}")


def test_actual_data_sample(tokenizer):
    print()
    print("=" * 60)
    print("TEST 7: Actual Dataset Sample")
    print("=" * 60)

    try:
        # Try to load actual dataset
        sys.path.insert(0, '/root/workspace/code/GenerativeRecommenders')
        from genrec.data.amazon_lcrec import AmazonLCRecDataset

        print("Loading dataset (this may take a while)...")
        dataset = AmazonLCRecDataset(
            root="dataset/amazon",
            split="beauty",
            train_test_split="train",
            pretrained_rqvae_path="./out/lcrec/amazon/beauty/rqvae/checkpoint_epoch_1999.pt",
            enabled_tasks=["seqrec"],
            encoder_model_name="/root/workspace/models_hub/sentence-t5-xl",
            # RQVAE config - must match the checkpoint!
            rqvae_input_dim=768,
            rqvae_embed_dim=64,  # Official: 64
            rqvae_hidden_dims=[512, 256, 128],  # Official config
            rqvae_codebook_size=256,
            rqvae_n_layers=5,  # Official: 5 codebooks
        )

        print(f"Dataset size: {len(dataset)}")

        # Print a few samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n--- Sample {i} ---")
            print(f"Task: {sample['task']}")
            print(f"Prompt (first 200 chars): {sample['prompt'][:200]}...")
            print(f"Response: {sample['response']}")
            if 'target_sem_ids' in sample:
                print(f"Target sem_ids: {sample['target_sem_ids']}")

            # Tokenize and check
            full_text = sample['prompt'] + sample['response'] + tokenizer.eos_token
            encoded = tokenizer(full_text)
            prompt_ids = tokenizer(sample['prompt'], add_special_tokens=False).input_ids

            print(f"Full text tokens: {len(encoded['input_ids'])}")
            print(f"Prompt tokens: {len(prompt_ids)}")
            print(f"Response tokens: {len(encoded['input_ids']) - len(prompt_ids)}")

    except Exception as e:
        print(f"Could not load dataset: {e}")
        print("Skipping actual data test")


def main():
    print("=" * 60)
    print("LCRec Training Pipeline Verification")
    print("=" * 60)
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
    tokenizer = add_codebook_tokens(tokenizer, NUM_CODEBOOKS, CODEBOOK_SIZE)

    # Run tokenization tests
    prompt, response, prompt_len = test_tokenization_issues(tokenizer)

    print()
    print("Loading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    print(f"Model loaded on device: {next(model.parameters()).device}")

    # Run model tests
    test_embedding_initialization(tokenizer, model)
    test_forward_pass(tokenizer, model, prompt, response, prompt_len)
    test_constrained_decoding(tokenizer, model, prompt)

    # Test actual data
    test_actual_data_sample(tokenizer)

    print()
    print("=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
