"""Independent paper-based TokenMixer-Large backbone (arXiv:2602.06563v2).

Figure 1 and sections 3.3.3/A.4 determine the Pre-RMSNorm residual layout;
equations 12/16 instead print Post-Norm. Undisclosed/public-data choices are
listed in config/reproduction/tokenmixer_large_paper_mapping_20261112.json.
No author or third-party TokenMixer-Large implementation is imported/copied.
The existing framework feature embedding adapter retains its MIT attribution
in genrec.models.rankmixer and licenses/RankElastor-MIT.txt.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from genrec.models.rankmixer import _FeatureEmbedding, ranking_feature_specs


PAPER = "https://arxiv.org/abs/2602.06563v2"


class TokenMixerRMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(dtype) * self.weight


def mix_tokens(x, num_heads):
    """(B,T,D) -> (B,H,T*D/H), preserving the original token/head order."""
    b, t, d = x.shape
    if num_heads < 1 or d % num_heads:
        raise ValueError("token_dim must be divisible by num_heads")
    return x.reshape(b, t, num_heads, d // num_heads).transpose(1, 2).reshape(b, num_heads, -1)


def revert_tokens(x, token_num):
    b, h, width = x.shape
    if token_num < 1 or width % token_num:
        raise ValueError("mixed width must be divisible by token_num")
    return x.reshape(b, h, token_num, width // token_num).transpose(1, 2).reshape(b, token_num, -1)


class PerTokenSwiGLU(nn.Module):
    """Parameter-isolated bias-free SwiGLUs, batched over token positions."""
    def __init__(self, slots, width, hidden, down_init_gain=0.01):
        super().__init__()
        self.slots, self.width, self.hidden = slots, width, hidden
        self.up_weight = nn.Parameter(torch.empty(slots, width, hidden))
        self.gate_weight = nn.Parameter(torch.empty(slots, width, hidden))
        self.down_weight = nn.Parameter(torch.empty(slots, hidden, width))
        for i in range(slots):
            nn.init.xavier_uniform_(self.up_weight[i])
            nn.init.xavier_uniform_(self.gate_weight[i])
            nn.init.xavier_uniform_(self.down_weight[i], gain=down_init_gain)

    def forward(self, x):
        up = torch.einsum("btd,tdh->bth", x, self.up_weight)
        gate = torch.einsum("btd,tdh->bth", x, self.gate_weight)
        return torch.einsum("bth,thd->btd", F.silu(gate) * up, self.down_weight)


class SwiGLUExpert(nn.Module):
    def __init__(self, width, hidden, down_init_gain):
        super().__init__()
        self.up = nn.Linear(width, hidden, bias=False)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.down.weight, gain=down_init_gain)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SparsePerTokenMoE(nn.Module):
    """One always-active shared expert plus top-(k-1) routed experts per slot.

    Selected sample rows alone reach each routed FFN. This is a portable
    reference dispatcher, without the paper's grouped kernels/token parallelism.
    'selected' implements v2's sum-to-one selected gates; 'all' is an explicitly
    different full-softmax-before-topk ablation. In the selected/top-1 case the
    selected weight is 1 and the router has zero gate gradient, as the formula
    implies; no unreported straight-through estimator is inserted.
    """
    def __init__(self, slots, width, dense_hidden, num_experts=4, top_k=2,
                 gate_scale=None, router_normalization="selected", down_init_gain=0.01):
        super().__init__()
        if not 2 <= top_k <= num_experts or dense_hidden % num_experts:
            raise ValueError("require 2 <= top_k <= num_experts and hidden divisible by num_experts")
        if router_normalization not in {"selected", "all"}:
            raise ValueError("router_normalization must be selected or all")
        self.slots, self.width = slots, width
        self.num_experts, self.top_k = num_experts, top_k
        self.hidden = dense_hidden // num_experts
        self.gate_scale = num_experts / top_k if gate_scale is None else gate_scale
        self.router_normalization = router_normalization
        self.router_weight = nn.Parameter(torch.empty(slots, width, num_experts - 1))
        for weight in self.router_weight:
            nn.init.xavier_uniform_(weight)
        self.shared = PerTokenSwiGLU(slots, width, self.hidden, down_init_gain)
        self.routed = nn.ModuleList([
            nn.ModuleList([SwiGLUExpert(width, self.hidden, down_init_gain)
                           for _ in range(num_experts - 1)]) for _ in range(slots)
        ])
        self.register_buffer("route_counts", torch.zeros(slots, num_experts - 1, dtype=torch.long), persistent=False)
        self.register_buffer("route_samples", torch.tensor(0, dtype=torch.long), persistent=False)

    def routes(self, x):
        logits = torch.einsum("btd,tde->bte", x, self.router_weight).float()
        values, indices = logits.topk(self.top_k - 1, dim=-1)
        if self.router_normalization == "selected":
            weights = values.softmax(-1)
        else:
            weights = logits.softmax(-1).gather(-1, indices)
        return indices, weights

    def forward(self, x):
        chosen, weights = self.routes(x)
        output = self.shared(x)
        rows = []
        for slot in range(self.slots):
            update = torch.zeros_like(output[:, slot])
            for index, expert in enumerate(self.routed[slot]):
                row, choice = torch.where(chosen[:, slot] == index)
                if row.numel() == 0:
                    continue
                contribution = expert(x[row, slot])
                contribution = contribution * (self.gate_scale * weights[row, slot, choice, None]).to(contribution.dtype)
                update = update.index_add(0, row, contribution.to(update.dtype))
            rows.append(update)
        if self.training:
            with torch.no_grad():
                self.route_counts.add_(F.one_hot(chosen, self.num_experts - 1).sum((0, 2)))
                self.route_samples.add_(len(x))
        return output + torch.stack(rows, dim=1)

    @torch.no_grad()
    def partition_from_dense(self, dense):
        """Split the dense hidden axis; gate weighting is a separate choice."""
        if dense.slots != self.slots or dense.width != self.width or dense.hidden != self.hidden * self.num_experts:
            raise ValueError("dense and sparse bank shapes are incompatible")
        n = self.hidden
        # The final hidden slice is the always-active shared expert.
        self.shared.up_weight.copy_(dense.up_weight[:, :, -n:])
        self.shared.gate_weight.copy_(dense.gate_weight[:, :, -n:])
        self.shared.down_weight.copy_(dense.down_weight[:, -n:, :])
        for t in range(self.slots):
            for e, expert in enumerate(self.routed[t]):
                region = slice(e * n, (e + 1) * n)
                expert.up.weight.copy_(dense.up_weight[t, :, region].T)
                expert.gate.weight.copy_(dense.gate_weight[t, :, region].T)
                expert.down.weight.copy_(dense.down_weight[t, region, :].T)

    def routing_summary(self):
        samples = max(1, self.route_samples.item())
        return {"samples_per_slot": self.route_samples.item(),
                "routed_fractions_per_slot": (self.route_counts.float() / samples).cpu().tolist(),
                "shared_fraction_per_slot": 1.0, "activated_experts_including_shared": self.top_k,
                "router_normalization": self.router_normalization,
                "top1_selected_router_gradient_zero": self.router_normalization == "selected" and self.top_k == 2}


class TokenMixerLargeBlock(nn.Module):
    def __init__(self, token_num, token_dim, num_heads, hidden_multiplier=4,
                 num_experts=0, top_k=2, gate_scale=None, router_normalization="selected",
                 down_init_gain=0.01, rms_eps=1e-6):
        super().__init__()
        if token_dim % num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_num, self.num_heads = token_num, num_heads
        self.norm_mix, self.norm_token = TokenMixerRMSNorm(token_dim, rms_eps), TokenMixerRMSNorm(token_dim, rms_eps)
        mixed_width = token_num * token_dim // num_heads

        def bank(slots, width):
            hidden = int(hidden_multiplier * width)
            if num_experts:
                return SparsePerTokenMoE(slots, width, hidden, num_experts, top_k, gate_scale,
                                         router_normalization, down_init_gain)
            return PerTokenSwiGLU(slots, width, hidden, down_init_gain)

        self.mix_ffn, self.token_ffn = bank(num_heads, mixed_width), bank(token_num, token_dim)

    def forward(self, x):
        # Figure 1: Norm -> Mix -> FFN -> Revert -> residual, then Norm -> FFN -> residual.
        x = x + revert_tokens(self.mix_ffn(mix_tokens(self.norm_mix(x), self.num_heads)), self.token_num)
        return x + self.token_ffn(self.norm_token(x))


class GroupwiseTokenizer(nn.Module):
    def __init__(self, feature_specs, embedding_dim, token_num, token_dim,
                 hidden=128, dropout=0.0, feature_groups=None):
        super().__init__()
        names = list(feature_specs)
        if token_num < 2:
            raise ValueError("include at least one group token and one global token")
        if feature_groups is None:
            # Criteo columns are anonymous: deterministic balanced groups, not semantic claims.
            count = token_num - 1
            if count > len(names):
                raise ValueError("cannot create more nonempty groups than fields")
            feature_groups = [names[i * len(names) // count:(i + 1) * len(names) // count] for i in range(count)]
        flat = [name for group in feature_groups for name in group]
        if len(feature_groups) != token_num - 1 or sorted(flat) != sorted(names) or any(not g for g in feature_groups):
            raise ValueError("feature_groups must partition every field exactly once")
        self.feature_groups = [list(g) for g in feature_groups]
        self.indices = [[names.index(name) for name in group] for group in feature_groups]

        def projection(fields):
            return nn.Sequential(nn.Linear(fields * embedding_dim, hidden, bias=False), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, token_dim, bias=False))

        self.global_projection = projection(len(names))
        self.group_projections = nn.ModuleList([projection(len(g)) for g in self.indices])
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, embedded):
        global_token = self.global_projection(embedded.flatten(1))
        groups = [project(embedded[:, indices].flatten(1))
                  for indices, project in zip(self.indices, self.group_projections)]
        return torch.stack([global_token] + groups, dim=1)


class TokenMixerLarge(nn.Module):
    """Public CTR adaptation with an independently reconstructed paper backbone."""
    def __init__(self, feature_map, embedding_dim=20, token_num=16, token_dim=32,
                 num_blocks=4, num_heads=8, hidden_multiplier=4, tokenizer_hidden=128,
                 tokenizer_dropout=0.0, feature_groups=None, num_experts=0, top_k=2,
                 gate_scale=None, router_normalization="selected", down_init_gain=0.01,
                 inter_residual_gap=2, aux_loss_weight=0.1, embedding_regularizer=1e-5,
                 rms_eps=1e-6):
        super().__init__()
        if min(embedding_dim, token_num, token_dim, num_blocks, num_heads, tokenizer_hidden) < 1:
            raise ValueError("model dimensions must be positive")
        if hidden_multiplier <= 0 or down_init_gain < 0 or aux_loss_weight < 0 or inter_residual_gap < 0:
            raise ValueError("invalid hidden expansion, initialization or auxiliary settings")
        self.embedding_regularizer, self.aux_loss_weight = embedding_regularizer, aux_loss_weight
        self.inter_residual_gap = inter_residual_gap
        self.feature_specs = ranking_feature_specs(feature_map)
        self.embedding_layer = _FeatureEmbedding(self.feature_specs, embedding_dim)
        for module in self.embedding_layer.modules():
            if isinstance(module, nn.Embedding):
                start = 1 if module.padding_idx == 0 else 0
                nn.init.normal_(module.weight.data[start:], std=1e-4)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
        self.tokenizer = GroupwiseTokenizer(self.feature_specs, embedding_dim, token_num, token_dim,
                                            tokenizer_hidden, tokenizer_dropout, feature_groups)
        self.blocks = nn.ModuleList([
            TokenMixerLargeBlock(token_num, token_dim, num_heads, hidden_multiplier, num_experts,
                                 top_k, gate_scale, router_normalization, down_init_gain, rms_eps)
            for _ in range(num_blocks)
        ])
        self.aux_depths = [i for i in range(1, num_blocks) if inter_residual_gap and i % inter_residual_gap == 0]
        self.final_norm = TokenMixerRMSNorm(token_dim, rms_eps)
        self.head = nn.Linear(token_dim, 1, bias=False)
        self.aux_heads = nn.ModuleDict({str(i): nn.Linear(token_dim, 1, bias=False) for i in self.aux_depths})
        nn.init.xavier_uniform_(self.head.weight)
        for layer in self.aux_heads.values():
            nn.init.xavier_uniform_(layer.weight)

    def encode(self, features, return_auxiliary=False):
        x = self.tokenizer(self.embedding_layer(features))
        anchor, intermediate = x, {}
        for depth, block in enumerate(self.blocks, start=1):
            x = block(x)
            if depth in self.aux_depths:
                x = x + anchor
                anchor = x
                if return_auxiliary:
                    intermediate[depth] = self.aux_heads[str(depth)](self.final_norm(x)[:, 0])
        return self.head(self.final_norm(x)[:, 0]), intermediate

    def forward(self, features, labels=None):
        logits, auxiliary = self.encode(features, return_auxiliary=labels is not None and self.aux_loss_weight > 0)
        if labels is None:
            return logits, None
        target = labels.float().reshape_as(logits)
        loss = F.binary_cross_entropy_with_logits(logits.float(), target)
        if auxiliary:
            # Figure 2/text: shallow prediction estimates the residual of deeper logits.
            # Exact tower/weight/stop-gradient details are not specified: cumulative
            # logits, mean auxiliary BCE and no detach are explicit implementation choices.
            combined, losses = logits.float(), []
            for depth in sorted(auxiliary, reverse=True):
                combined = combined + auxiliary[depth].float()
                losses.append(F.binary_cross_entropy_with_logits(combined, target))
            loss = loss + self.aux_loss_weight * torch.stack(losses).mean()
        regularization = [m.weight.float().square().sum() for m in self.modules() if isinstance(m, nn.Embedding)]
        return logits, loss + self.embedding_regularizer * 0.5 * sum(regularization)

    def routing_summary(self):
        return {name: module.routing_summary() for name, module in self.named_modules()
                if isinstance(module, SparsePerTokenMoE)}
