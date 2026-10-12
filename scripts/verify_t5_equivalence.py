"""
Verify numerical equivalence between custom TransformerEncoderDecoder (hf_compat mode)
and HuggingFace T5ForConditionalGeneration.

Steps:
1. Create HF T5 with specific config
2. Create custom T5 with matching hf_compat flags
3. Copy weights from HF to custom
4. Compare forward outputs
"""

import torch
import torch.nn as nn
import math
from transformers import T5Config, T5ForConditionalGeneration

import sys
sys.path.insert(0, ".")
from genrec.modules.transformer import TransformerEncoderDecoder
from genrec.modules.normalize import RootMeanSquareLayerNorm

# Config matching config/tiger/amazon/tiger.gin
D_MODEL = 128
D_KV = 64
D_FF = 1024
NUM_HEADS = 6
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DROPOUT = 0.0  # Disable dropout for deterministic comparison
VOCAB_SIZE = 769  # 256 * 3 + 1
NUM_BUCKETS = 32
MAX_DISTANCE = 128

torch.manual_seed(42)

# ============================================================
# 1. Create HF T5
# ============================================================
hf_config = T5Config(
    num_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
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
    relative_attention_num_buckets=NUM_BUCKETS,
    relative_attention_max_distance=MAX_DISTANCE,
)
hf_model = T5ForConditionalGeneration(hf_config)
hf_model.eval()

# ============================================================
# 2. Create Custom T5 (hf_compat mode)
# ============================================================
custom_model = TransformerEncoderDecoder(
    d_model=D_MODEL,
    nhead=NUM_HEADS,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    dim_feedforward=D_FF,
    dropout=DROPOUT,
    norm_cls=RootMeanSquareLayerNorm,
    d_kv=D_KV,
    share_position_bias=True,
    scale_attn=False,
    add_final_norm=True,
    decoder_bidirectional=False,
)
custom_model.eval()

# ============================================================
# 3. Copy weights from HF to Custom
# ============================================================
def copy_weights(hf_model, custom_model):
    """Copy HF T5 weights to custom TransformerEncoderDecoder."""
    hf_enc = hf_model.encoder
    hf_dec = hf_model.decoder

    # --- Encoder layers ---
    for i in range(NUM_ENCODER_LAYERS):
        hf_block = hf_enc.block[i]
        custom_block = custom_model.encoder.layers[i]

        # Self-attention
        hf_attn = hf_block.layer[0].SelfAttention
        custom_attn = custom_block.self_attn.attn

        # Q
        custom_attn.q.weight.data.copy_(hf_attn.q.weight.data)
        # K, V -> fused KV
        custom_attn.kv.weight.data.copy_(
            torch.cat([hf_attn.k.weight.data, hf_attn.v.weight.data], dim=0)
        )
        # O
        custom_attn.o.weight.data.copy_(hf_attn.o.weight.data)

        # Relative position bias (only first layer when share_position_bias=True)
        if hasattr(hf_attn, 'relative_attention_bias') and custom_attn.rel_bias is not None:
            # HF: (num_buckets, n_heads) -> Custom: (n_heads * num_buckets, 1)
            hf_bias_weight = hf_attn.relative_attention_bias.weight.data  # (num_buckets, n_heads)
            # Rearrange: for head h, bucket b: custom_idx = h * num_buckets + b
            custom_bias = torch.zeros(NUM_HEADS * NUM_BUCKETS, 1)
            for h in range(NUM_HEADS):
                for b in range(NUM_BUCKETS):
                    custom_bias[h * NUM_BUCKETS + b, 0] = hf_bias_weight[b, h]
            custom_attn.rel_bias.weight.data.copy_(custom_bias)

        # Self-attention layer norm
        custom_block.norm1.weight.data.copy_(hf_block.layer[0].layer_norm.weight.data)

        # FFN
        custom_block.ff.wi.weight.data.copy_(hf_block.layer[1].DenseReluDense.wi.weight.data)
        custom_block.ff.wo.weight.data.copy_(hf_block.layer[1].DenseReluDense.wo.weight.data)

        # FFN layer norm
        custom_block.norm2.weight.data.copy_(hf_block.layer[1].layer_norm.weight.data)

    # Encoder final norm
    if custom_model.encoder.final_norm is not None:
        custom_model.encoder.final_norm.weight.data.copy_(hf_enc.final_layer_norm.weight.data)

    # --- Decoder layers ---
    for i in range(NUM_DECODER_LAYERS):
        hf_block = hf_dec.block[i]
        custom_block = custom_model.decoder.layers[i]

        # Self-attention
        hf_attn = hf_block.layer[0].SelfAttention
        custom_attn = custom_block.self_attn.attn

        custom_attn.q.weight.data.copy_(hf_attn.q.weight.data)
        custom_attn.kv.weight.data.copy_(
            torch.cat([hf_attn.k.weight.data, hf_attn.v.weight.data], dim=0)
        )
        custom_attn.o.weight.data.copy_(hf_attn.o.weight.data)

        # Relative position bias (only first layer)
        if hasattr(hf_attn, 'relative_attention_bias') and custom_attn.rel_bias is not None:
            hf_bias_weight = hf_attn.relative_attention_bias.weight.data
            custom_bias = torch.zeros(NUM_HEADS * NUM_BUCKETS, 1)
            for h in range(NUM_HEADS):
                for b in range(NUM_BUCKETS):
                    custom_bias[h * NUM_BUCKETS + b, 0] = hf_bias_weight[b, h]
            custom_attn.rel_bias.weight.data.copy_(custom_bias)

        # Self-attention layer norm
        custom_block.norm1.weight.data.copy_(hf_block.layer[0].layer_norm.weight.data)

        # Cross-attention
        hf_cross = hf_block.layer[1].EncDecAttention
        custom_cross = custom_block.cross_attn.attn

        custom_cross.q.weight.data.copy_(hf_cross.q.weight.data)
        custom_cross.k.weight.data.copy_(hf_cross.k.weight.data)
        custom_cross.v.weight.data.copy_(hf_cross.v.weight.data)
        custom_cross.o.weight.data.copy_(hf_cross.o.weight.data)

        # Cross-attention layer norm
        custom_block.norm_cross.weight.data.copy_(hf_block.layer[1].layer_norm.weight.data)

        # FFN
        custom_block.ff.wi.weight.data.copy_(hf_block.layer[2].DenseReluDense.wi.weight.data)
        custom_block.ff.wo.weight.data.copy_(hf_block.layer[2].DenseReluDense.wo.weight.data)

        # FFN layer norm
        custom_block.norm2.weight.data.copy_(hf_block.layer[2].layer_norm.weight.data)

    # Decoder final norm
    if custom_model.decoder.final_norm is not None:
        custom_model.decoder.final_norm.weight.data.copy_(hf_dec.final_layer_norm.weight.data)

    print("Weight copy complete!")


copy_weights(hf_model, custom_model)

# ============================================================
# 4. Compare forward outputs
# ============================================================

# Create test inputs (pretend we already have embeddings to feed into the backbone)
B, SRC_LEN, TGT_LEN = 2, 10, 4
torch.manual_seed(123)

# For HF T5: use input_ids through embedding
src_ids = torch.randint(1, VOCAB_SIZE, (B, SRC_LEN))
tgt_ids = torch.randint(1, VOCAB_SIZE, (B, TGT_LEN))
src_mask = torch.ones(B, SRC_LEN, dtype=torch.long)
src_mask[1, 7:] = 0  # some padding

# HF forward: we need to get encoder hidden states and decoder hidden states
# To compare backbone only, we bypass the embedding and lm_head
with torch.no_grad():
    # Get HF embeddings
    hf_src_emb = hf_model.encoder.embed_tokens(src_ids)
    hf_tgt_emb = hf_model.decoder.embed_tokens(tgt_ids)

    # HF encoder forward
    hf_enc_out = hf_model.encoder(
        inputs_embeds=hf_src_emb,
        attention_mask=src_mask,
    ).last_hidden_state

    # HF decoder forward (with causal mask automatically applied)
    hf_dec_out = hf_model.decoder(
        inputs_embeds=hf_tgt_emb,
        encoder_hidden_states=hf_enc_out,
        encoder_attention_mask=src_mask,
    ).last_hidden_state

    # Custom forward: feed same embeddings
    src_key_padding_mask = (src_mask == 0)  # True = ignore
    # Custom model needs causal mask for decoder
    causal_mask = torch.triu(
        torch.full((TGT_LEN, TGT_LEN), float("-inf")),
        diagonal=1,
    )

    custom_enc_out = custom_model.encoder(
        hf_src_emb,
        key_padding_mask=src_key_padding_mask,
    )
    custom_dec_out = custom_model.decoder(
        hf_tgt_emb,
        memory=custom_enc_out,
        attn_mask=causal_mask,
        memory_key_padding_mask=src_key_padding_mask,
    )

# Compare
enc_diff = (hf_enc_out - custom_enc_out).abs()
dec_diff = (hf_dec_out - custom_dec_out).abs()

print(f"\n{'='*60}")
print(f"Encoder output comparison:")
print(f"  Max absolute diff: {enc_diff.max().item():.2e}")
print(f"  Mean absolute diff: {enc_diff.mean().item():.2e}")
print(f"  HF range: [{hf_enc_out.min().item():.4f}, {hf_enc_out.max().item():.4f}]")
print(f"  Custom range: [{custom_enc_out.min().item():.4f}, {custom_enc_out.max().item():.4f}]")

print(f"\nDecoder output comparison:")
print(f"  Max absolute diff: {dec_diff.max().item():.2e}")
print(f"  Mean absolute diff: {dec_diff.mean().item():.2e}")
print(f"  HF range: [{hf_dec_out.min().item():.4f}, {hf_dec_out.max().item():.4f}]")
print(f"  Custom range: [{custom_dec_out.min().item():.4f}, {custom_dec_out.max().item():.4f}]")

THRESHOLD = 1e-5
if enc_diff.max().item() < THRESHOLD and dec_diff.max().item() < THRESHOLD:
    print(f"\n✅ PASS: Custom T5 matches HF T5 (threshold={THRESHOLD})")
else:
    print(f"\n❌ FAIL: Outputs differ beyond threshold={THRESHOLD}")

    # Debug: check encoder layer by layer
    print(f"\n--- Layer-by-layer encoder debug ---")
    hf_enc_hidden = hf_src_emb
    custom_enc_hidden = hf_src_emb.clone()
    position_bias = None

    for layer_idx in range(NUM_ENCODER_LAYERS):
        hf_block = hf_model.encoder.block[layer_idx]
        custom_block = custom_model.encoder.layers[layer_idx]

        # HF encoder layer forward
        hf_layer_out = hf_block(
            hf_enc_hidden,
            attention_mask=hf_model.encoder.get_extended_attention_mask(
                src_mask, src_ids.shape, src_ids.device
            ),
        )
        hf_enc_hidden = hf_layer_out[0]

        # Custom encoder layer forward
        custom_enc_hidden, position_bias = custom_block(
            custom_enc_hidden,
            key_padding_mask=src_key_padding_mask,
            position_bias=position_bias,
        )

        diff = (hf_enc_hidden - custom_enc_hidden).abs().max().item()
        print(f"  Encoder layer {layer_idx}: max diff = {diff:.2e}")

    # Check final norm
    if custom_model.encoder.final_norm is not None:
        hf_final = hf_model.encoder.final_layer_norm(hf_enc_hidden)
        custom_final = custom_model.encoder.final_norm(custom_enc_hidden)
        custom_final = custom_model.encoder.final_dropout(custom_final)
        diff = (hf_final - custom_final).abs().max().item()
        print(f"  Final norm: max diff = {diff:.2e}")
