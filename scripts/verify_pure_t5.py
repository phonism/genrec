"""
Verify numerical equivalence between our pure PyTorch T5 and HuggingFace T5.

Since our module names match HF exactly, weights can be copied via state_dict.
"""

import sys
sys.path.insert(0, ".")

import torch
from transformers import T5Config as HFT5Config
from transformers import T5ForConditionalGeneration as HFT5

from genrec.modules.t5 import T5Config as OurT5Config
from genrec.modules.t5 import T5ForConditionalGeneration as OurT5

# ============================================================
# Config (matching TIGER setup, dropout=0 for deterministic comparison)
# ============================================================
D_MODEL = 128
D_KV = 64
D_FF = 1024
NUM_HEADS = 6
NUM_LAYERS = 4
VOCAB_SIZE = 769
DROPOUT = 0.0

torch.manual_seed(42)

# ============================================================
# 1. Create HF T5
# ============================================================
hf_config = HFT5Config(
    num_layers=NUM_LAYERS,
    num_decoder_layers=NUM_LAYERS,
    d_model=D_MODEL,
    d_ff=D_FF,
    num_heads=NUM_HEADS,
    d_kv=D_KV,
    dropout_rate=DROPOUT,
    vocab_size=VOCAB_SIZE,
    pad_token_id=0,
    eos_token_id=0,
    decoder_start_token_id=0,
    feed_forward_proj='relu',
)
hf_model = HFT5(hf_config)
hf_model.eval()

# ============================================================
# 2. Create Our T5
# ============================================================
our_config = OurT5Config(
    num_layers=NUM_LAYERS,
    num_decoder_layers=NUM_LAYERS,
    d_model=D_MODEL,
    d_ff=D_FF,
    num_heads=NUM_HEADS,
    d_kv=D_KV,
    dropout_rate=DROPOUT,
    vocab_size=VOCAB_SIZE,
    pad_token_id=0,
    eos_token_id=0,
    decoder_start_token_id=0,
    feed_forward_proj='relu',
)
our_model = OurT5(our_config)
our_model.eval()

# ============================================================
# 3. Copy weights from HF → Our model
# ============================================================
hf_sd = hf_model.state_dict()
our_sd = our_model.state_dict()

# Check which keys match / differ
hf_keys = set(hf_sd.keys())
our_keys = set(our_sd.keys())

# HF has some keys we don't have (and vice versa)
common_keys = hf_keys & our_keys
only_hf = hf_keys - our_keys
only_ours = our_keys - hf_keys

print(f"HF keys: {len(hf_keys)}")
print(f"Our keys: {len(our_keys)}")
print(f"Common keys: {len(common_keys)}")
if only_hf:
    print(f"Only in HF ({len(only_hf)}):")
    for k in sorted(only_hf):
        print(f"  {k}  {hf_sd[k].shape}")
if only_ours:
    print(f"Only in Ours ({len(only_ours)}):")
    for k in sorted(only_ours):
        print(f"  {k}  {our_sd[k].shape}")

# Copy common weights
mismatched_shapes = []
for key in common_keys:
    if hf_sd[key].shape == our_sd[key].shape:
        our_sd[key].copy_(hf_sd[key])
    else:
        mismatched_shapes.append((key, hf_sd[key].shape, our_sd[key].shape))

if mismatched_shapes:
    print(f"\nShape mismatches ({len(mismatched_shapes)}):")
    for k, hf_shape, our_shape in mismatched_shapes:
        print(f"  {k}: HF={hf_shape} vs Ours={our_shape}")

# Also handle HF-only keys that need mapping
# HF uses "encoder.embed_tokens.weight" and "decoder.embed_tokens.weight"
# which are tied to "shared.weight" — copy shared to encoder/decoder embed
for hf_key in only_hf:
    # Try to find corresponding key in our model
    if hf_key in our_sd:
        our_sd[hf_key].copy_(hf_sd[hf_key])

our_model.load_state_dict(our_sd)

# Re-tie lm_head to shared
our_model.lm_head.weight = our_model.shared.weight

print("\nWeight copy complete!")

# ============================================================
# 4. Test forward (training mode — with labels)
# ============================================================
B, SRC_LEN, TGT_LEN = 2, 10, 4
torch.manual_seed(123)
src_ids = torch.randint(1, VOCAB_SIZE, (B, SRC_LEN))
labels = torch.randint(1, VOCAB_SIZE, (B, TGT_LEN))
src_mask = torch.ones(B, SRC_LEN, dtype=torch.long)
src_mask[1, 7:] = 0  # some padding

with torch.no_grad():
    hf_out = hf_model(input_ids=src_ids, attention_mask=src_mask, labels=labels)
    our_out = our_model(input_ids=src_ids, attention_mask=src_mask, labels=labels)

hf_logits = hf_out.logits
our_logits = our_out.logits
hf_loss = hf_out.loss
our_loss = our_out.loss

logits_diff = (hf_logits - our_logits).abs()
loss_diff = (hf_loss - our_loss).abs()

print(f"\n{'='*60}")
print(f"Forward comparison (with labels):")
print(f"  Logits shape: HF={hf_logits.shape}, Ours={our_logits.shape}")
print(f"  Logits max diff: {logits_diff.max().item():.2e}")
print(f"  Logits mean diff: {logits_diff.mean().item():.2e}")
print(f"  Loss: HF={hf_loss.item():.6f}, Ours={our_loss.item():.6f}, diff={loss_diff.item():.2e}")

THRESHOLD = 1e-5
if logits_diff.max().item() < THRESHOLD and loss_diff.item() < THRESHOLD:
    print(f"\n✅ PASS: Forward outputs match (threshold={THRESHOLD})")
else:
    print(f"\n❌ FAIL: Forward outputs differ beyond threshold={THRESHOLD}")

    # Debug: step through encoder
    print(f"\n--- Debug: step-by-step ---")

    # 1. Check embeddings
    with torch.no_grad():
        hf_emb = hf_model.shared(src_ids)
        our_emb = our_model.shared(src_ids)
        print(f"  Embedding diff: {(hf_emb - our_emb).abs().max().item():.2e}")

    # 2. Check encoder output
    with torch.no_grad():
        hf_enc = hf_model.encoder(input_ids=src_ids, attention_mask=src_mask)
        our_enc = our_model.encoder(input_ids=src_ids, attention_mask=src_mask)

        hf_enc_hidden = hf_enc.last_hidden_state if hasattr(hf_enc, 'last_hidden_state') else hf_enc[0]
        our_enc_hidden = our_enc[0]

        print(f"  Encoder output diff: {(hf_enc_hidden - our_enc_hidden).abs().max().item():.2e}")

    # 3. Check decoder output
    with torch.no_grad():
        decoder_input_ids = our_model._shift_right(labels)
        hf_dec = hf_model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hf_enc_hidden,
            encoder_attention_mask=src_mask,
        )
        our_dec = our_model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=our_enc_hidden,
            encoder_attention_mask=src_mask,
        )
        hf_dec_hidden = hf_dec.last_hidden_state if hasattr(hf_dec, 'last_hidden_state') else hf_dec[0]
        our_dec_hidden = our_dec[0]

        print(f"  Decoder output diff: {(hf_dec_hidden - our_dec_hidden).abs().max().item():.2e}")

    # 4. Check lm_head (with scaling)
    with torch.no_grad():
        hf_scaled = hf_dec_hidden * (D_MODEL ** -0.5)
        our_scaled = our_dec_hidden * (D_MODEL ** -0.5)
        hf_logits2 = hf_model.lm_head(hf_scaled)
        our_logits2 = our_model.lm_head(our_scaled)
        print(f"  LM head logits diff: {(hf_logits2 - our_logits2).abs().max().item():.2e}")

# ============================================================
# 5. Test generation (beam search)
# ============================================================
print(f"\n{'='*60}")
print(f"Generation comparison (beam search):")
with torch.no_grad():
    hf_gen = hf_model.generate(
        input_ids=src_ids, attention_mask=src_mask,
        max_length=4, num_beams=5, num_return_sequences=5,
    )
    our_gen = our_model.generate(
        input_ids=src_ids, attention_mask=src_mask,
        max_length=4, num_beams=5, num_return_sequences=5,
    )

print(f"  HF generated shape: {hf_gen.shape}")
print(f"  Our generated shape: {our_gen.shape}")
if hf_gen.shape == our_gen.shape:
    match = (hf_gen == our_gen).all()
    if match:
        print(f"  ✅ Generated sequences match exactly!")
    else:
        diff_count = (hf_gen != our_gen).sum().item()
        total = hf_gen.numel()
        print(f"  ❌ Generated sequences differ: {diff_count}/{total} tokens")
        print(f"  HF top beam: {hf_gen[0].tolist()}")
        print(f"  Our top beam: {our_gen[0].tolist()}")
else:
    print(f"  ❌ Shape mismatch")

print(f"\n{'='*60}")
print("Done!")
