"""
SASRec: Self-Attentive Sequential Recommendation
https://arxiv.org/abs/1808.09781

A self-attention based sequential recommendation model that adaptively
considers consumed items for next-item prediction.

This implementation follows the official TensorFlow implementation:
https://github.com/kang205/SASRec
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SASRec(nn.Module):
    """
    SASRec model implementation following the official TF version.

    Key details from official implementation:
    1. Embedding scaled by sqrt(d)
    2. Key masking based on embedding values (zero = padding)
    3. Query masking after softmax (zero out padding positions)
    4. Residual connection inside attention
    5. Mask applied after each block
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_blocks: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # Embeddings: item 0 is padding (will be zeroed)
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Dropout for embeddings
        self.emb_dropout = nn.Dropout(dropout)

        # Self-attention blocks
        self.blocks = nn.ModuleList([
            SASRecBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_blocks)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-8)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, L]
        targets: Optional[torch.Tensor] = None,  # [B, L] for training
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Item IDs sequence [B, L], 0 is padding
            targets: Target item IDs [B, L] for computing loss (shifted by 1)

        Returns:
            logits: Prediction logits [B, L, num_items+1]
            loss: Cross-entropy loss if targets provided, else None
        """
        B, L = input_ids.shape
        device = input_ids.device

        # Mask for padding positions: [B, L, 1]
        # True for valid positions, False for padding
        mask = (input_ids != 0).unsqueeze(-1).float()

        # Item embedding (scaled by sqrt(d) as in official impl)
        x = self.item_embedding(input_ids) * (self.embed_dim ** 0.5)  # [B, L, D]

        # Position embedding (not scaled)
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = x + self.position_embedding(positions)

        # Dropout and mask
        x = self.emb_dropout(x)
        x = x * mask  # Zero out padding positions

        # Apply self-attention blocks
        for block in self.blocks:
            x = block(x, mask)
            x = x * mask  # Zero out padding after each block (as in official impl)

        x = self.final_norm(x)

        # Prediction: dot product with item embeddings
        logits = x @ self.item_embedding.weight.T  # [B, L, num_items+1]

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, self.num_items + 1)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=0)

        return logits, loss

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, top_k: int = 10) -> torch.Tensor:
        """Predict top-k items for next item."""
        logits, _ = self.forward(input_ids)
        last_logits = logits[:, -1, :]
        last_logits[:, 0] = float('-inf')  # Exclude padding
        _, top_k_items = torch.topk(last_logits, top_k, dim=-1)
        return top_k_items


class SASRecBlock(nn.Module):
    """Single self-attention block for SASRec (following official impl)."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = PointWiseFeedForward(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-8)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-8)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]
            mask: Valid position mask [B, L, 1], True=valid, False=padding
        """
        # Self-attention: normalize queries only (as in official impl)
        # Residual is added inside attention
        x = self.attention(self.norm1(x), x, mask)

        # Feed-forward with residual (added inside ffn)
        x = self.ffn(self.norm2(x), x)

        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention following official SASRec implementation.

    Key differences from standard attention:
    1. Key masking based on embedding values (sum of abs)
    2. Query masking after softmax
    3. Use large negative number instead of -inf
    4. Residual connection inside this module
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,  # [B, L, D] - normalized
        key_value: torch.Tensor,  # [B, L, D] - not normalized (original x)
        mask: torch.Tensor,  # [B, L, 1]
    ) -> torch.Tensor:
        B, L, _ = query.shape

        # Project Q from normalized input, K/V from original input
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)

        # Reshape for multi-head: [B, L, D] -> [B, H, L, D_h]
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, H, L, L]
        scores = (Q @ K.transpose(-2, -1)) * self.scale

        # Key masking: based on whether key positions are padding
        # In official impl: key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))
        # Since padding embeddings are zero, we use the mask directly
        # mask: [B, L, 1] -> key_mask: [B, 1, 1, L]
        key_mask = mask.squeeze(-1).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]

        # Use large negative number instead of -inf (official uses -2^32+1)
        padding_value = -1e9
        scores = scores.masked_fill(key_mask == 0, padding_value)

        # Causal masking (future blinding)
        causal_mask = torch.triu(torch.ones(L, L, device=scores.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), padding_value)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Query masking: zero out attention weights for padding query positions
        # This is done AFTER softmax in official impl
        query_mask = mask.squeeze(-1).unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
        attn_weights = attn_weights * query_mask

        # Dropout
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = attn_weights @ V  # [B, H, L, D_h]
        out = out.transpose(1, 2).contiguous().view(B, L, self.embed_dim)

        # Residual connection (inside attention as in official impl)
        # Note: residual uses the normalized query, not original x
        out = out + query

        return out


class PointWiseFeedForward(nn.Module):
    """Point-wise feed-forward network with residual inside."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Normalized input [B, L, D]
            residual: Original input for residual connection [B, L, D]
        """
        out = self.fc2(self.dropout(F.relu(self.fc1(x))))
        out = self.dropout(out)
        return out + residual  # Residual connection
