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
        score_l2_norm: bool = False,
        score_temperature: float = 1.0,
        dqk: Optional[int] = None,
        dv: Optional[int] = None,
        use_position_bias: bool = True,
        concat_ua: bool = False,
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
            score_l2_norm: L2-normalize query and item embeddings before dot product
                (matches Meta HSTU sampled-softmax retrieval setup). Required so the
                eval path uses the same score function as SS training.
            score_temperature: divisor applied to logits after dot product.
        """
        super().__init__()
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.use_temporal_bias = use_temporal_bias
        self.score_l2_norm = score_l2_norm
        self.score_temperature = score_temperature

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
                max_seq_len=max_seq_len,
                dqk=dqk,
                dv=dv,
                use_position_bias=use_position_bias,
                concat_ua=concat_ua,
            )
            for _ in range(num_blocks)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Score `x` against the full item vocabulary, applying L2 norm + temperature
        when configured (must be used by every train/eval path so they stay aligned).
        eps=1e-6 matches Meta's `l2_norm_eps`."""
        w = self.item_embedding.weight
        if self.score_l2_norm:
            x = F.normalize(x, dim=-1, eps=1e-6)
            w = F.normalize(w, dim=-1, eps=1e-6)
        return (x @ w.T) / self.score_temperature

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
        sampled_softmax: bool = False,
        num_negatives: int = 128,
        sampling_strategy: str = "local",
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            input_ids: Item ID sequence [B, L], 0 is padding
            timestamps: Unix timestamps [B, L], optional for temporal bias
            targets: Target item IDs [B, L] for loss computation
            sampled_softmax: dispatch to forward_sampled_softmax. The trainer must
                route training through forward() (not call the method on the
                unwrapped module) so DDP's gradient-sync hooks fire.
            num_negatives / sampling_strategy: only used when sampled_softmax=True.

        Returns:
            logits: Prediction logits [B, L, num_items+1]
            loss: Cross-entropy loss if targets provided
        """
        if sampled_softmax:
            return self.forward_sampled_softmax(
                input_ids, timestamps, targets,
                num_negatives=num_negatives, sampling_strategy=sampling_strategy,
            )
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

        # Prediction via dot product with item embeddings (optionally L2-normed + temp-scaled)
        logits = self._compute_logits(x)  # [B, L, V]

        # Compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_items + 1),
                targets.view(-1),
                ignore_index=0
            )

        return logits, loss

    def forward_sampled_softmax(
        self,
        input_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        num_negatives: int = 128,
        sampling_strategy: str = "local",  # Meta local-table; also in_batch/global
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward with sampled softmax loss (aligned with Meta's original HSTU).

        Score function (L2 norm + temperature) comes from `self.score_l2_norm` and
        `self.score_temperature` — the single source of truth shared with
        `_compute_logits`, so training and eval can never disagree.

        Negative sampling:
            - "local": for every positive, independently sample ``num_negatives``
              item IDs from the local embedding table.  This matches Meta's
              ``LocalNegativesSampler`` ("local" refers to a non-sharded local
              table, not to in-batch sampling).
            - "in_batch": independently sample from unique valid targets in
              the current batch, matching Meta's separately named sampler.
            - "global": legacy GenRec behavior using one shared, deduplicated
              uniform-negative set for the whole batch.
        """
        B, L = input_ids.shape
        device = input_ids.device

        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        padding_mask = (input_ids == 0)

        x = self.item_embedding(input_ids)
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, causal_mask, padding_mask, timestamps)

        x = self.final_norm(x)  # [B, L, D]

        # Full-vocab logits are only needed on the no-target (eval) path; during
        # training they'd be a [B, L, V] matmul whose result the trainer discards.
        logits = None if targets is not None else self._compute_logits(x)

        use_l2 = self.score_l2_norm
        temp = self.score_temperature

        loss = None
        if targets is not None:
            # Flatten to [B*L, D] and [B*L]
            x_flat = x.view(-1, self.embed_dim)  # [B*L, D]
            targets_flat = targets.view(-1)  # [B*L]

            # Filter out padding positions
            valid_mask = targets_flat != 0
            x_valid = x_flat[valid_mask]  # [N, D]
            targets_valid = targets_flat[valid_mask]  # [N]

            if x_valid.size(0) > 0:
                # Get positive embeddings
                pos_emb = self.item_embedding(targets_valid)  # [N, D]

                # Meta's LocalNegativesSampler draws a separate [R] sample for
                # every positive from all item IDs.  The previous implementation
                # misread "local" as "in-batch" and also shared one small
                # negative set across all positions, so it was not the published
                # sampled-softmax objective.
                if sampling_strategy == "local":
                    neg_ids = torch.randint(
                        1,
                        self.num_items + 1,
                        (x_valid.size(0), num_negatives),
                        device=device,
                    )
                elif sampling_strategy == "in_batch":
                    pool = targets_valid.unique()
                    offsets = torch.randint(
                        0,
                        pool.numel(),
                        (x_valid.size(0), num_negatives),
                        device=device,
                    )
                    neg_ids = pool[offsets]
                elif sampling_strategy == "global":
                    neg_ids = torch.randint(
                        1, self.num_items + 1, (num_negatives,), device=device
                    ).unique()
                else:
                    raise ValueError(
                        "sampling_strategy must be 'local', 'in_batch', or "
                        f"'global', got {sampling_strategy!r}"
                    )
                neg_emb = self.item_embedding(neg_ids)

                if use_l2:
                    # eps=1e-6 to match Meta's `l2_norm_eps`. Default F.normalize eps=1e-12.
                    x_valid = F.normalize(x_valid, dim=-1, eps=1e-6)
                    pos_emb = F.normalize(pos_emb, dim=-1, eps=1e-6)
                    neg_emb = F.normalize(neg_emb, dim=-1, eps=1e-6)

                # Compute logits: [N, 1+K]
                pos_logits = (x_valid * pos_emb).sum(dim=-1, keepdim=True)  # [N, 1]
                if neg_emb.ndim == 3:
                    neg_logits = torch.einsum("nd,nrd->nr", x_valid, neg_emb)
                else:
                    neg_logits = x_valid @ neg_emb.T

                # Collision mask: when a sampled negative is the same item as the
                # positive (frequent in in-batch sampling on small vocabs), exclude
                # it from the softmax. Matches Meta's `torch.where(... == ..., -5e4, ...)`.
                if neg_ids.ndim == 2:
                    collision = targets_valid.unsqueeze(1) == neg_ids
                else:
                    collision = targets_valid.unsqueeze(1) == neg_ids.unsqueeze(0)
                neg_logits = neg_logits.masked_fill(collision, -5e4)

                all_logits = torch.cat([pos_logits, neg_logits], dim=-1) / temp

                # Target is always index 0 (positive)
                loss_targets = torch.zeros(x_valid.size(0), device=device, dtype=torch.long)
                loss = F.cross_entropy(all_logits, loss_targets)

        return logits, loss

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, timestamps: Optional[torch.Tensor] = None, top_k: int = 10) -> torch.Tensor:
        """Predict top-k items for next item (uses same score function as training)."""
        logits, _ = self.forward(input_ids, timestamps)
        last_logits = logits[:, -1, :].clone()
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
        max_seq_len: int = 50,
        dqk: Optional[int] = None,
        dv: Optional[int] = None,
        use_position_bias: bool = True,
        concat_ua: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        # Decoupled head dims: paper uses dqk=dv=25 with embed_dim=50, num_heads=2
        # When unset, fall back to embed_dim/num_heads (legacy behavior).
        if dqk is None or dv is None:
            assert embed_dim % num_heads == 0, \
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}) when dqk/dv are unset"
        self.dqk = dqk if dqk is not None else embed_dim // num_heads
        self.dv = dv if dv is not None else embed_dim // num_heads
        self.qk_dim = num_heads * self.dqk
        self.uv_dim = num_heads * self.dv
        self.use_temporal_bias = use_temporal_bias
        self.use_position_bias = use_position_bias
        self.concat_ua = concat_ua

        # Pointwise projection: outputs [U | V | Q | K] with sizes [uv, uv, qk, qk]
        self.projection = nn.Linear(embed_dim, 2 * (self.uv_dim + self.qk_dim))

        # Relative attention bias (position-based, shared across heads).
        # Paper's ML-1M setup uses only temporal RAB; position_bias kept for legacy.
        if use_position_bias:
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

        # Pre-norm on the layer input (matches Meta's `self._norm_input(x)` before projection)
        self.input_norm = nn.LayerNorm(embed_dim)
        # Layer norm for attention output (operates on V dim)
        self.attn_norm = nn.LayerNorm(self.uv_dim)
        # Output projection back to embed_dim for the residual connection.
        # When concat_ua=True, gating produces 3*uv_dim and out_proj maps that to embed_dim.
        out_in_dim = (3 * self.uv_dim) if concat_ua else self.uv_dim
        self.out_proj = nn.Linear(out_in_dim, embed_dim)

        # FFN block (kept for ML-1M; paper HSTU omits it but empirically it helps SS here)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
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

        # === Pointwise Projection (with pre-norm) ===
        # Pre-norm matches Meta's `normed_x = self._norm_input(x)` before projection.
        # SiLU is applied once on the attention scores below — projection is pure linear.
        normed_x = self.input_norm(x)
        projected = self.projection(normed_x)
        U, V, Q, K = projected.split([self.uv_dim, self.uv_dim, self.qk_dim, self.qk_dim], dim=-1)

        # Reshape for multi-head attention
        Q = Q.view(B, L, self.num_heads, self.dqk).transpose(1, 2)  # [B, H, L, dqk]
        K = K.view(B, L, self.num_heads, self.dqk).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.dv).transpose(1, 2)   # [B, H, L, dv]

        # === Spatial Aggregation ===
        # Compute attention scores (without softmax!)
        scores = Q @ K.transpose(-2, -1)  # [B, H, L, L]

        # Add relative position bias (paper's ML-1M setup uses only temporal RAB)
        if self.use_position_bias:
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

        # HSTU attention activation: SiLU + divide by n (Meta: `F.silu(qk_attn) / n`).
        # n must be the FIXED max_seq_len, not the batch's padded length L: collate
        # pads to the batch max, so /L would make a sequence's representation depend
        # on the longest sequence sharing its batch (metrics would shift with
        # eval_batch_size / data order). Without the division, output magnitude
        # scales with sequence length and destabilizes training.
        attn_weights = F.silu(scores) / float(self.max_seq_len)

        # Apply attention to values
        attn_output = attn_weights @ V  # [B, H, L, dv]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.uv_dim)  # [B, L, uv_dim]

        # === Pointwise Transformation ===
        # Normalize and gate with U. concat_ua=True uses paper's multi-channel
        # gating `cat([u, a, u*a])` for richer interaction; default is just `u*a`.
        a = self.attn_norm(attn_output)
        if self.concat_ua:
            gated = torch.cat([U, a, U * a], dim=-1)  # [B, L, 3*uv_dim]
        else:
            gated = a * U  # [B, L, uv_dim]
        attn_output = self.out_proj(gated)  # -> embed_dim

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
