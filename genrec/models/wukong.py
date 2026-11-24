"""Independent Wukong reconstruction (arXiv:2403.02545v4, sections 3.3-3.6).

FMB uses the associative low-rank product X (X^T Y); Y can be static or
input-conditioned. MLP sizes, attentive-Y compression and normalization scope
are explicit public-data choices, not inferred author defaults. The shared
feature adapter retains its existing MIT notice in genrec.models.rankmixer.
"""
import torch
from torch import nn
from torch.nn import functional as F

from genrec.models.rankmixer import _FeatureEmbedding, ranking_feature_specs


def mlp(input_width, hidden, output_width, dropout=0.0):
    layers, width = [], input_width
    for size in hidden:
        layers.extend([nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)])
        width = size
    layers.append(nn.Linear(width, output_width))
    return nn.Sequential(*layers)


class LowRankFactorization(nn.Module):
    def __init__(self, fields, embed_dim, rank=24, projection="attentive",
                 projection_tokens=4, projection_hidden=128):
        super().__init__()
        if projection not in {"static", "attentive"}:
            raise ValueError("projection must be static or attentive")
        self.fields, self.rank, self.projection = fields, rank, projection
        if projection == "static":
            self.y = nn.Parameter(torch.empty(fields, rank))
            nn.init.xavier_normal_(self.y)
        else:
            self.compress = nn.Linear(fields, projection_tokens, bias=False)
            self.project = mlp(projection_tokens * embed_dim, [projection_hidden], fields * rank)

    def projection_matrix(self, x):
        if self.projection == "static":
            return self.y.expand(len(x), -1, -1)
        compressed = self.compress(x.transpose(1, 2)).transpose(1, 2)
        return self.project(compressed.flatten(1)).reshape(len(x), self.fields, self.rank)

    def forward(self, x):
        y = self.projection_matrix(x)
        return x @ (x.transpose(1, 2) @ y)


class WukongInteraction(nn.Module):
    def __init__(self, fields, embed_dim, fmb_tokens=16, lcb_tokens=16, rank=24,
                 hidden=(512, 512), dropout=0.0, projection="attentive",
                 projection_tokens=4, projection_hidden=128):
        super().__init__()
        self.embed_dim, self.fmb_tokens = embed_dim, fmb_tokens
        self.fm = LowRankFactorization(fields, embed_dim, rank, projection,
                                        projection_tokens, projection_hidden)
        self.fm_norm = nn.LayerNorm(fields * rank)
        self.fm_mlp = mlp(fields * rank, hidden, fmb_tokens * embed_dim, dropout)
        self.lcb = nn.Linear(fields, lcb_tokens, bias=False)
        output_fields = fmb_tokens + lcb_tokens
        self.residual = nn.Identity() if fields == output_fields else nn.Linear(fields, output_fields, bias=False)
        self.output_norm = nn.LayerNorm([output_fields, embed_dim])

    def forward(self, x):
        fmb = self.fm_mlp(self.fm_norm(self.fm(x).flatten(1))).reshape(len(x), self.fmb_tokens, self.embed_dim)
        lcb = self.lcb(x.transpose(1, 2)).transpose(1, 2)
        residual = self.residual(x.transpose(1, 2)).transpose(1, 2)
        return self.output_norm(torch.cat([fmb, lcb], dim=1) + residual)


class Wukong(nn.Module):
    def __init__(self, feature_map, embedding_dim=20, num_blocks=3, fmb_tokens=16,
                 lcb_tokens=16, rank=24, fmb_hidden=(512, 512), head_hidden=(512, 512),
                 dropout=0.0, projection="attentive", projection_tokens=4,
                 projection_hidden=128, embedding_regularizer=1e-5):
        super().__init__()
        if min(embedding_dim, num_blocks, fmb_tokens, lcb_tokens, rank) < 1:
            raise ValueError("Wukong dimensions must be positive")
        specs = ranking_feature_specs(feature_map)
        self.embedding_regularizer = embedding_regularizer
        self.embedding_layer = _FeatureEmbedding(specs, embedding_dim)
        fields, blocks = len(specs), []
        for _ in range(num_blocks):
            blocks.append(WukongInteraction(fields, embedding_dim, fmb_tokens, lcb_tokens,
                                            rank, fmb_hidden, dropout, projection,
                                            projection_tokens, projection_hidden))
            fields = fmb_tokens + lcb_tokens
        self.blocks = nn.Sequential(*blocks)
        self.head = mlp(fields * embedding_dim, head_hidden, 1, dropout)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                start = 1 if module.padding_idx == 0 else 0
                nn.init.normal_(module.weight.data[start:], std=1e-4)

    def forward(self, features, labels=None):
        x = self.blocks(self.embedding_layer(features))
        logits = self.head(x.flatten(1))
        if labels is None:
            return logits, None
        loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float().reshape_as(logits))
        penalty = sum(m.weight.float().square().sum() for m in self.modules() if isinstance(m, nn.Embedding))
        return logits, loss + self.embedding_regularizer * 0.5 * penalty
