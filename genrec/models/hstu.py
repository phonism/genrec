"""
HSTU: Hierarchical Sequential Transduction Unit
From "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"
https://arxiv.org/abs/2402.17152

Key differences from standard Transformer:
1. SiLU activation instead of softmax normalization (captures preference intensity)
2. Update gate U for gating mechanism
3. Relative attention bias with both position and temporal components
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class HSTU(nn.Module):
    """
    HSTU model for sequential recommendation.

    Architecture:
        Input -> Item Embedding + (optional) Temporal Encoding
              -> [HSTU Layer × num_blocks]
              -> Prediction (dot product with item embeddings)
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        dropout: float = 0.2,
        num_position_buckets: int = 32,
        num_time_buckets: int = 64,
        max_position_distance: int = 128,
        use_temporal_bias: bool = True,
    ):
        """
        Args:
            num_items: Total number of items
            max_seq_len: Maximum sequence length
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_blocks: Number of HSTU layers
            dropout: Dropout rate
            num_position_buckets: Number of buckets for position bias
            num_time_buckets: Number of buckets for temporal bias
            max_position_distance: Max distance for position bucketing
            use_temporal_bias: Whether to use temporal attention bias
        """
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.use_temporal_bias = use_temporal_bias

        # Item embedding (0 is padding)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)

        # Embedding dropout
        self.emb_dropout = nn.Dropout(dropout)

        # HSTU layers
        self.layers = nn.ModuleList([
            HSTULayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_position_buckets=num_position_buckets,
                num_time_buckets=num_time_buckets,
                max_position_distance=max_position_distance,
                use_temporal_bias=use_temporal_bias,
            )
            for _ in range(num_blocks)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, L]
        timestamps: Optional[torch.Tensor] = None,  # [B, L] unix timestamps
        targets: Optional[torch.Tensor] = None,  # [B, L] for training
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Item ID sequence [B, L], 0 is padding
            timestamps: Unix timestamps [B, L], optional for temporal bias
            targets: Target item IDs [B, L] for loss computation

        Returns:
            logits: Prediction logits [B, L, num_items+1]
            loss: Cross-entropy loss if targets provided
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

        # Padding mask
        padding_mask = (input_ids == 0)

        # Item embedding
        x = self.item_embedding(input_ids)  # [B, L, D]
        x = self.emb_dropout(x)

        # Apply HSTU layers
        for layer in self.layers:
            x = layer(x, causal_mask, padding_mask, timestamps)

        x = self.final_norm(x)

        # Prediction via dot product with item embeddings
        logits = x @ self.item_embedding.weight.T  # [B, L, V]

        # Compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_items + 1),
                targets.view(-1),
                ignore_index=0
            )

        return logits, loss

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, timestamps: Optional[torch.Tensor] = None, top_k: int = 10) -> torch.Tensor:
        """Predict top-k items for next item."""
        logits, _ = self.forward(input_ids, timestamps)
        last_logits = logits[:, -1, :]
        last_logits[:, 0] = float('-inf')  # Exclude padding
        _, top_k_items = torch.topk(last_logits, top_k, dim=-1)
        return top_k_items


class HSTULayer(nn.Module):
    """
    Single HSTU layer.

    Structure:
        1. Pointwise Projection: X -> SiLU(Linear(X)) -> split to U, V, Q, K
        2. Spatial Aggregation: SiLU(QK^T + RAB) @ V
        3. Pointwise Transformation: Norm(Attention) ⊙ U -> FFN
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        num_position_buckets: int,
        num_time_buckets: int,
        max_position_distance: int,
        use_temporal_bias: bool,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_temporal_bias = use_temporal_bias

        assert embed_dim % num_heads == 0

        # Pointwise projection: projects to 4 * embed_dim (for U, V, Q, K)
        self.projection = nn.Linear(embed_dim, 4 * embed_dim)

        # Relative attention bias (position-based, shared across heads)
        self.position_bias = RelativePositionBias(
            num_buckets=num_position_buckets,
            max_distance=max_position_distance,
            num_heads=num_heads,
        )

        # Temporal bias (optional)
        if use_temporal_bias:
            self.temporal_bias = TemporalBias(
                num_buckets=num_time_buckets,
                num_heads=num_heads,
            )

        # Layer norm for attention output
        self.attn_norm = nn.LayerNorm(embed_dim)

        # FFN (pointwise transformation)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Final layer norm
        self.ffn_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        causal_mask: torch.Tensor,  # [L, L]
        padding_mask: torch.Tensor,  # [B, L]
        timestamps: Optional[torch.Tensor] = None,  # [B, L]
    ) -> torch.Tensor:
        B, L, D = x.shape
        residual = x

        # === Pointwise Projection ===
        # Project and apply SiLU, then split into U, V, Q, K
        projected = F.silu(self.projection(x))  # [B, L, 4D]
        U, V, Q, K = projected.chunk(4, dim=-1)  # Each [B, L, D]

        # Reshape for multi-head attention
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, d]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # === Spatial Aggregation ===
        # Compute attention scores (without softmax!)
        scores = Q @ K.transpose(-2, -1)  # [B, H, L, L]

        # Add relative position bias
        pos_bias = self.position_bias(L, x.device)  # [H, L, L]
        scores = scores + pos_bias.unsqueeze(0)

        # Add temporal bias if enabled and timestamps provided
        if self.use_temporal_bias and timestamps is not None:
            time_bias = self.temporal_bias(timestamps)  # [B, H, L, L]
            scores = scores + time_bias

        # Apply causal mask (set masked positions to large negative)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)

        # Apply padding mask
        scores = scores.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), -1e9)

        # HSTU key: SiLU instead of softmax!
        # This allows capturing preference intensity
        attn_weights = F.silu(scores)

        # Apply attention to values
        attn_output = attn_weights @ V  # [B, H, L, d]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]

        # === Pointwise Transformation ===
        # Normalize and gate with U
        attn_output = self.attn_norm(attn_output)
        attn_output = attn_output * U  # Element-wise gating

        # Residual connection
        x = residual + self.dropout(attn_output)

        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))

        return x


class RelativePositionBias(nn.Module):
    """
    Relative position bias using logarithmic bucketing (T5-style).

    Buckets relative positions into logarithmically spaced bins,
    allowing the model to generalize to longer sequences.
    """

    def __init__(self, num_buckets: int = 32, max_distance: int = 128, num_heads: int = 2):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads

        # Learnable bias for each bucket and head
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """
        Convert relative position to bucket index using logarithmic bucketing.

        For causal attention, we only care about positions where query >= key,
        so relative_position >= 0.
        """
        # We use half buckets for exact positions, half for log-spaced
        num_buckets = self.num_buckets
        max_distance = self.max_distance

        # Clamp to non-negative (causal)
        relative_position = torch.clamp(relative_position, min=0)

        # Half buckets for small distances (exact)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # Log-spaced buckets for larger distances
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()

        relative_position_if_large = torch.clamp(relative_position_if_large, max=num_buckets - 1)

        bucket = torch.where(is_small, relative_position, relative_position_if_large)
        return bucket

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute relative position bias matrix.

        Returns:
            bias: [num_heads, seq_len, seq_len]
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        # relative_position[i, j] = i - j (query_pos - key_pos)
        relative_position = positions.unsqueeze(0) - positions.unsqueeze(1)  # [L, L]

        # Convert to buckets
        buckets = self._relative_position_bucket(relative_position)  # [L, L]

        # Look up bias values
        bias = self.relative_attention_bias(buckets)  # [L, L, H]
        bias = bias.permute(2, 0, 1)  # [H, L, L]

        return bias


class TemporalBias(nn.Module):
    """
    Temporal attention bias using logarithmic bucketing of time differences.

    Quantizes timestamp differences into log-spaced buckets,
    capturing both recent and long-term temporal patterns.
    """

    def __init__(self, num_buckets: int = 64, num_heads: int = 2):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads

        # Learnable bias for each bucket and head
        self.temporal_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _temporal_bucket(self, time_diff: torch.Tensor) -> torch.Tensor:
        """
        Convert time difference to bucket index.

        Uses formula: bucket = floor(log(max(1, |diff|)) / log_base)
        where log_base ≈ 0.301 (log10(2)) as in the paper
        """
        # Take absolute value and ensure minimum of 1
        abs_diff = torch.clamp(torch.abs(time_diff), min=1).float()

        # Log bucketing (using natural log, scaled)
        # Paper uses: floor(log(max(1, |diff|)) / 0.301)
        # We use a similar approach but cap at num_buckets - 1
        buckets = (torch.log(abs_diff) / 0.693).long()  # 0.693 = ln(2)
        buckets = torch.clamp(buckets, min=0, max=self.num_buckets - 1)

        return buckets

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal bias matrix.

        Args:
            timestamps: [B, L] unix timestamps

        Returns:
            bias: [B, num_heads, L, L]
        """
        B, L = timestamps.shape

        # Compute pairwise time differences
        # time_diff[i, j] = timestamps[i] - timestamps[j]
        time_diff = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)  # [B, L, L]

        # Convert to buckets
        buckets = self._temporal_bucket(time_diff)  # [B, L, L]

        # Look up bias values
        bias = self.temporal_attention_bias(buckets)  # [B, L, L, H]
        bias = bias.permute(0, 3, 1, 2)  # [B, H, L, L]

        return bias
