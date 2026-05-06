"""
Pure PyTorch T5ForConditionalGeneration — pixel-perfect match with HuggingFace.

No dependency on `transformers`. Weights can be copied directly from HF T5.
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 1. T5Config
# ──────────────────────────────────────────────
@dataclass
class T5Config:
    vocab_size: int = 32128
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_decoder_layers: Optional[int] = None
    num_heads: int = 8
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "relu"
    is_encoder_decoder: bool = True
    use_cache: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: Optional[int] = None
    is_decoder: bool = False
    tie_word_embeddings: bool = True

    # derived — filled in __post_init__
    dense_act_fn: str = field(init=False)
    is_gated_act: bool = field(init=False)
    scale_decoder_outputs: bool = field(init=False)

    def __post_init__(self):
        if self.num_decoder_layers is None:
            self.num_decoder_layers = self.num_layers
        if self.decoder_start_token_id is None:
            self.decoder_start_token_id = self.pad_token_id

        # parse feed_forward_proj  (e.g. "relu", "gated-gelu", "gated-silu")
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        # HF quirk: "gated-gelu" → dense_act_fn = "gelu_new"
        if self.feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        # HF quirk: scale_decoder_outputs = True when tie_word_embeddings is True
        # (original T5 scales, T5 1.1 does not — HF uses tie_word_embeddings as indicator)
        self.scale_decoder_outputs = self.tie_word_embeddings is True
        # HF forces tie_word_embeddings = True always
        self.tie_word_embeddings = True


# ──────────────────────────────────────────────
# 2. Activation functions
# ──────────────────────────────────────────────
def _gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_new": _gelu_new,
    "silu": F.silu,
    "swish": F.silu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}


# ──────────────────────────────────────────────
# 3. T5LayerNorm  (RMSNorm — no bias, no mean subtraction)
# ──────────────────────────────────────────────
class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # convert to half-precision if weight is half
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


# ──────────────────────────────────────────────
# 4. Feed-forward layers
# ──────────────────────────────────────────────
class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # dtype compat (fp16/int8)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


# ──────────────────────────────────────────────
# 5. T5Attention
# ──────────────────────────────────────────────
class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        """Translate relative position to a bucket number (matches HF exactly)."""
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in [0, inf)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        """Compute binned relative position bias → (1, n_heads, q_len, k_len)."""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # (q_len, k_len)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # (q, k, n_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # (1, n_heads, q, k)
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Self-attention (key_value_states=None) or cross-attention.
        Returns: (attn_output, position_bias[, (new_key, new_value)])
        """
        batch_size, seq_length = hidden_states.shape[:2]
        is_cross_attention = key_value_states is not None

        # Q projection
        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        # K, V projection (with optional KV-cache for generation)
        if is_cross_attention and past_key_value is not None:
            # reuse cached cross-attention K/V
            key_states, value_states = past_key_value
        else:
            current_states = key_value_states if is_cross_attention else hidden_states
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

            if not is_cross_attention and past_key_value is not None:
                # concat self-attention KV cache
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # optionally save cache
        new_key_value = (key_states, value_states) if use_cache else None

        # Attention scores — T5 does NOT scale by 1/sqrt(d_k)!
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        # Position bias
        if position_bias is None:
            key_length = key_states.shape[2]
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length),
                    device=scores.device, dtype=scores.dtype,
                )
            else:
                # for cached decoding, compute bias for full length
                real_query_length = key_length if not is_cross_attention else seq_length
                position_bias = self.compute_bias(real_query_length, key_length, device=scores.device)
                # slice to current query length (for cached decoding)
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                position_bias = position_bias + mask[:, :, :seq_length, :key_states.shape[2]]

        scores += position_bias

        # Softmax in fp32
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output, position_bias)
        if use_cache:
            outputs = outputs + (new_key_value,)
        return outputs


# ──────────────────────────────────────────────
# 6. Attention wrapper layers
# ──────────────────────────────────────────────
class T5LayerSelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        normed = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed, mask=attention_mask, position_bias=position_bias,
            past_key_value=past_key_value, use_cache=use_cache,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # position_bias, maybe cache
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        normed = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed, mask=attention_mask, key_value_states=key_value_states,
            position_bias=position_bias, past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]
        return outputs


# ──────────────────────────────────────────────
# 7. T5Block
# ──────────────────────────────────────────────

def _fp16_clamp(hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.dtype == torch.float16:
        clamp_value = torch.where(
            torch.isinf(hidden_states).any(),
            torch.finfo(hidden_states.dtype).max - 1000,
            torch.finfo(hidden_states.dtype).max,
        )
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    return hidden_states


class T5Block(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        )
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_decoder_position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
    ):
        """
        Returns:
            Without cache: (hidden_states, self_position_bias, [cross_position_bias])
            With cache:    (hidden_states, self_position_bias, [cross_position_bias], layer_cache)
            where layer_cache = (self_kv, cross_kv) for decoder, (self_kv,) for encoder
        """
        # Unpack per-layer cache
        self_attn_past = None
        cross_attn_past = None
        if past_key_value is not None:
            self_attn_past = past_key_value[0]
            if self.is_decoder and len(past_key_value) > 1:
                cross_attn_past = past_key_value[1]

        # --- Self-attention ---
        # output: (hidden_states, position_bias, [kv_cache])
        sa_outputs = self.layer[0](
            hidden_states, attention_mask=attention_mask,
            position_bias=position_bias, past_key_value=self_attn_past,
            use_cache=use_cache,
        )
        hidden_states = sa_outputs[0]
        self_position_bias = sa_outputs[1]
        self_cache = sa_outputs[2] if use_cache else None

        hidden_states = _fp16_clamp(hidden_states)

        # --- Cross-attention (decoder only) ---
        cross_position_bias = None
        cross_cache = None
        do_cross = self.is_decoder and encoder_hidden_states is not None
        if do_cross:
            ca_outputs = self.layer[1](
                hidden_states, key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                past_key_value=cross_attn_past, use_cache=use_cache,
            )
            hidden_states = ca_outputs[0]
            cross_position_bias = ca_outputs[1]
            cross_cache = ca_outputs[2] if use_cache else None
            hidden_states = _fp16_clamp(hidden_states)

        # --- Feed-forward ---
        hidden_states = self.layer[-1](hidden_states)
        hidden_states = _fp16_clamp(hidden_states)

        # Build output tuple — matches HF index convention:
        # [0] = hidden_states
        # [1] = self_position_bias
        # [2] = cross_position_bias  (decoder only)
        # [-1] = layer_cache (if use_cache)
        outputs = (hidden_states, self_position_bias)
        if do_cross:
            outputs += (cross_position_bias,)
        if use_cache:
            layer_cache = (self_cache, cross_cache) if self.is_decoder else (self_cache,)
            outputs += (layer_cache,)

        return outputs


# ──────────────────────────────────────────────
# 8. Mask utilities
# ──────────────────────────────────────────────
def _make_pad_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert (B, L) padding mask → (B, 1, 1, L) additive mask for encoder / cross-attention."""
    # attention_mask: 1 = attend, 0 = pad
    mask = attention_mask[:, None, None, :].to(dtype)
    mask = (1.0 - mask) * torch.finfo(dtype).min
    return mask


def _make_causal_mask(
    input_ids: torch.Tensor, dtype: torch.dtype, past_length: int = 0,
) -> torch.Tensor:
    """Create (B, 1, L, L+past) causal + padding mask for decoder self-attention."""
    batch_size, seq_length = input_ids.shape
    total_length = seq_length + past_length
    # causal: lower-triangular ones
    causal = torch.ones(seq_length, total_length, dtype=dtype, device=input_ids.device)
    causal = torch.tril(causal, diagonal=past_length)
    causal = causal[None, None, :, :]  # (1, 1, L, L+past)
    causal = (1.0 - causal) * torch.finfo(dtype).min
    return causal.expand(batch_size, -1, -1, -1)


# ──────────────────────────────────────────────
# 9. T5Stack
# ──────────────────────────────────────────────
class T5Stack(nn.Module):
    def __init__(self, config: T5Config, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()
        self.config = config
        self.is_decoder = config.is_decoder
        self.embed_tokens = embed_tokens  # shared embedding passed in

        self.block = nn.ModuleList([
            T5Block(config, has_relative_attention_bias=bool(i == 0))
            for i in range(config.num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            (last_hidden_state, [present_key_values])
        """
        if inputs_embeds is None:
            assert self.embed_tokens is not None
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = inputs_embeds.shape[:2]
        dtype = inputs_embeds.dtype

        past_length = 0
        if past_key_values is not None and past_key_values[0] is not None:
            # past_key_values[0] = first layer's cache
            # past_key_values[0][0] = self-attention cache (k, v)
            # k shape: (B, n_heads, past_len, d_kv)
            past_length = past_key_values[0][0][0].shape[2]

        # Build attention masks
        if self.is_decoder:
            # causal mask: (B, 1, L, L+past)
            causal_mask = _make_causal_mask(
                torch.zeros(batch_size, seq_length, dtype=torch.long, device=inputs_embeds.device),
                dtype, past_length=past_length,
            )
            if attention_mask is not None:
                # combine with padding mask: expand pad mask to (B, 1, 1, L+past)
                pad_mask = _make_pad_mask(attention_mask, dtype)
                attention_mask_4d = causal_mask + pad_mask
            else:
                attention_mask_4d = causal_mask
        else:
            # encoder: bidirectional
            if attention_mask is not None:
                attention_mask_4d = _make_pad_mask(attention_mask, dtype)
            else:
                attention_mask_4d = None

        # encoder cross-attention mask (for decoder)
        encoder_attention_mask_4d = None
        if self.is_decoder and encoder_hidden_states is not None and encoder_attention_mask is not None:
            encoder_attention_mask_4d = _make_pad_mask(encoder_attention_mask, dtype)

        # Run through blocks
        hidden_states = self.dropout(inputs_embeds)
        position_bias = None
        encoder_decoder_position_bias = None
        present_key_values = [] if use_cache else None

        for i, layer_module in enumerate(self.block):
            layer_past = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask_4d,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                past_key_value=layer_past,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]

            # cross position bias for decoder
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[2]

            if use_cache:
                present_key_values.append(layer_outputs[-1])

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if use_cache:
            return hidden_states, tuple(present_key_values)
        return (hidden_states,)


# ──────────────────────────────────────────────
# 10. T5ForConditionalGeneration
# ──────────────────────────────────────────────
class T5ForConditionalGeneration(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.model_dim = config.d_model

        # Shared embedding — weight is tied across encoder, decoder, and lm_head
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        self.encoder = T5Stack(encoder_config, embed_tokens=self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, embed_tokens=self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie weights: lm_head.weight = shared.weight
        self.lm_head.weight = self.shared.weight

        # Initialize weights
        self.apply(self._init_weights)

    # ------ weight initialization (matches HF exactly) ------
    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        factor = self.config.initializer_factor
        d_model = self.config.d_model
        d_kv = self.config.d_kv
        d_ff = self.config.d_ff
        n_heads = self.config.num_heads

        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, T5ForConditionalGeneration):
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseActDense):
            module.wi.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.wo.weight.data.normal_(mean=0.0, std=factor * (d_ff ** -0.5))
        elif isinstance(module, T5DenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.wo.weight.data.normal_(mean=0.0, std=factor * (d_ff ** -0.5))
        elif isinstance(module, T5Attention):
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * d_kv) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * d_kv) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(
                    mean=0.0, std=factor * (d_model ** -0.5)
                )

    # ------ helper: shift labels right ------
    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        pad_token_id = self.config.pad_token_id
        decoder_start_token_id = self.config.decoder_start_token_id

        shifted = input_ids.new_zeros(input_ids.shape)
        shifted[..., 1:] = input_ids[..., :-1].clone()
        shifted[..., 0] = decoder_start_token_id
        # replace -100 (ignore_index) with pad_token_id
        shifted.masked_fill_(shifted == -100, pad_token_id)
        return shifted

    # ------ forward (training) ------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple] = None,
    ):
        """
        Returns an object with .loss and .logits attributes (SimpleNamespace).
        """
        # Encode
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hidden_states = encoder_outputs[0]

        # Shift labels → decoder_input_ids
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self._shift_right(labels)

        # Build full decoder attention mask for cached decoding
        if use_cache and past_key_values is not None and decoder_attention_mask is None:
            past_length = past_key_values[0][0][0].shape[2]
            decoder_attention_mask = torch.ones(
                decoder_input_ids.shape[0], decoder_input_ids.shape[1] + past_length,
                dtype=torch.long, device=decoder_input_ids.device,
            )

        # Decode
        decoder_out = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        sequence_output = decoder_out[0]
        present_key_values = decoder_out[1] if use_cache else None

        # HF quirk: scale decoder output when tie_word_embeddings=True
        if self.config.scale_decoder_outputs:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        # Return a simple namespace-like object
        return _Seq2SeqLMOutput(loss=loss, logits=lm_logits, past_key_values=present_key_values)

    # ------ generation (beam search) ------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 20,
        num_beams: int = 1,
        num_return_sequences: int = 1,
    ) -> torch.LongTensor:
        """Simple beam search generation (no sampling)."""
        batch_size = input_ids.shape[0]

        # Encode once
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if num_beams == 1:
            return self._greedy_generate(
                encoder_outputs, attention_mask, batch_size, max_length,
            )
        else:
            return self._beam_search(
                encoder_outputs, attention_mask, batch_size,
                max_length, num_beams, num_return_sequences,
            )

    def _greedy_generate(self, encoder_outputs, attention_mask, batch_size, max_length):
        device = encoder_outputs[0].device
        decoder_input_ids = torch.full(
            (batch_size, 1), self.config.decoder_start_token_id,
            dtype=torch.long, device=device,
        )
        past_key_values = None

        for _ in range(max_length - 1):
            out = self.forward(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids if past_key_values is None else decoder_input_ids[:, -1:],
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
            if (next_token == self.config.eos_token_id).all():
                break

        return decoder_input_ids

    def _beam_search(
        self, encoder_outputs, attention_mask, batch_size,
        max_length, num_beams, num_return_sequences,
    ):
        device = encoder_outputs[0].device
        vocab_size = self.config.vocab_size

        # Expand encoder outputs for beam search: (B, ...) → (B*num_beams, ...)
        enc_hidden = encoder_outputs[0]
        enc_hidden = enc_hidden.unsqueeze(1).expand(-1, num_beams, -1, -1).reshape(
            batch_size * num_beams, enc_hidden.shape[1], enc_hidden.shape[2]
        )
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(-1, num_beams, -1).reshape(
                batch_size * num_beams, -1
            )

        # Initialize beams
        decoder_input_ids = torch.full(
            (batch_size * num_beams, 1), self.config.decoder_start_token_id,
            dtype=torch.long, device=device,
        )
        # beam_scores: (batch_size, num_beams)
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = -1e9  # only first beam is active initially

        past_key_values = None

        for step in range(max_length - 1):
            out = self.forward(
                encoder_outputs=(enc_hidden,),
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids if past_key_values is None else decoder_input_ids[:, -1:],
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values

            # next token logits: (B*num_beams, vocab_size)
            next_token_logits = out.logits[:, -1, :]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            # (B, num_beams * vocab_size)
            next_scores = next_token_scores.view(batch_size, num_beams, -1)
            next_scores = next_scores + beam_scores.unsqueeze(-1)
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)

            # top-k — take num_beams directly (already sorted, no EOS filtering needed)
            top_scores, top_indices = torch.topk(next_scores, num_beams, dim=-1, largest=True, sorted=True)

            beam_indices = top_indices // vocab_size  # (B, num_beams) which beam
            token_indices = top_indices % vocab_size  # (B, num_beams) which token

            # Vectorized beam selection — no Python loops, no .item() calls
            beam_scores = top_scores  # (B, num_beams)

            # Flat beam indices for reordering: batch_idx * num_beams + beam_idx
            batch_offset = torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
            beam_idx = (batch_offset + beam_indices).reshape(-1)  # (B*num_beams,)

            # reorder decoder_input_ids and past_key_values
            decoder_input_ids = torch.cat([
                decoder_input_ids[beam_idx],
                token_indices.reshape(-1, 1),
            ], dim=-1)

            past_key_values = self._reorder_cache(past_key_values, beam_idx)

        # Select top num_return_sequences per batch
        # decoder_input_ids shape: (B*num_beams, seq_len)
        if num_return_sequences == num_beams:
            return decoder_input_ids
        else:
            # Vectorized: select top-scoring beams
            _, indices = beam_scores.topk(num_return_sequences, dim=-1)  # (B, num_return)
            batch_offset = torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
            flat_indices = (batch_offset + indices).reshape(-1)  # (B*num_return,)
            return decoder_input_ids[flat_indices]

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorder KV cache for beam search reordering."""
        reordered = []
        for layer_cache in past_key_values:
            reordered_layer = []
            for kv_pair in layer_cache:
                if kv_pair is None:
                    reordered_layer.append(None)
                else:
                    k, v = kv_pair
                    reordered_layer.append((k[beam_idx], v[beam_idx]))
            reordered.append(tuple(reordered_layer))
        return tuple(reordered)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)


# ──────────────────────────────────────────────
# Output container
# ──────────────────────────────────────────────
class _Seq2SeqLMOutput:
    __slots__ = ('loss', 'logits', 'past_key_values')

    def __init__(self, loss, logits, past_key_values=None):
        self.loss = loss
        self.logits = logits
        self.past_key_values = past_key_values
