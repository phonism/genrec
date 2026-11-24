"""GenRec CTR models matching the released RankElastor Criteo architectures.

Reference: vasile-paskardlgm/RankElastor, revision 68e5f29c9147b0c2263a1d28216b500940c057fe.
Adapted components copyright (c) 2026 PaskardLi, MIT license; see
licenses/RankElastor-MIT.txt. Neither FuxiCTR nor an upstream checkout is
required by these models. The industrial RankMixer release is a different scope.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class RankingRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return self.gamma * (x / torch.sqrt(x.square().mean(-1, keepdim=True) + self.eps))


def ranking_feature_specs(feature_map):
    if isinstance(feature_map, (str, Path)):
        feature_map = json.loads(Path(feature_map).read_text())
    features = feature_map.get("features", feature_map)
    if isinstance(features, list):
        features = {name: spec for field in features for name, spec in field.items()}
    if not isinstance(features, Mapping) or not features:
        raise ValueError("feature_map must contain ordered numeric/categorical fields")
    for name, spec in features.items():
        if spec["type"] not in {"numeric", "categorical"}:
            raise ValueError(f"Unsupported ranking feature {name}: {spec['type']}")
    return dict(features)


class _FeatureEmbeddingDict(nn.Module):
    def __init__(self, specs, width):
        super().__init__()
        self.specs = specs
        self.embedding_layers = nn.ModuleDict({
            name: (nn.Linear(1, width, bias=False) if spec["type"] == "numeric" else
                   nn.Embedding(spec["vocab_size"], width, padding_idx=spec.get("padding_idx")))
            for name, spec in specs.items()
        })

    def forward(self, features):
        outputs = []
        for index, (name, spec) in enumerate(self.specs.items()):
            value = features[name] if isinstance(features, Mapping) else features[:, index]
            value = value.float().reshape(-1, 1) if spec["type"] == "numeric" else value.long()
            outputs.append(self.embedding_layers[name](value))
        return torch.stack(outputs, dim=1)


class _FeatureEmbedding(nn.Module):
    def __init__(self, specs, width):
        super().__init__()
        self.embedding_layer = _FeatureEmbeddingDict(specs, width)

    def forward(self, features):
        return self.embedding_layer(features)


class _EmbeddingTokenizer(nn.Module):
    def __init__(self, fields, embedding_dim, tokens, width, hidden, dropout, normalization):
        super().__init__()
        total = fields * embedding_dim
        if total % tokens:
            raise ValueError("matrix tokenization requires fields * embedding_dim divisible by token_num")
        self.tokens = tokens
        self.pre_norm = RankingRMSNorm(total)
        layers = [nn.Linear(total // tokens, hidden)]
        if normalization:
            layers.append(RankingRMSNorm(hidden))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, width))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features):
        x = self.pre_norm(features.flatten(1))
        return self.mlp(x.reshape(x.shape[0], self.tokens, -1))


class _PublishedGLU(nn.Module):
    """Published 'SwiGLU' uses GELU and an extra skip inside the gate."""
    def __init__(self, width, expansion):
        super().__init__()
        lifted = int(width * expansion)
        self.exp_rate = expansion
        self.fc_lift1 = nn.Linear(width, lifted)
        self.fc_lift2 = nn.Linear(width, lifted)
        self.skip1 = nn.Linear(width, width)
        if expansion != 1:
            self.fc_reduce = nn.Linear(lifted, width)
            self.skip2 = nn.Linear(width, lifted)
        self.activation = nn.GELU()

    def forward(self, x):
        gate = self.activation(self.fc_lift1(x))
        value = self.fc_lift2(x)
        if self.exp_rate != 1:
            return self.fc_reduce((self.skip2(x) + gate) * value) + self.skip1(x)
        return (self.skip1(x) + gate) * value + self.skip1(x)


class _MeanOn4(nn.Module):
    def forward(self, x):
        if x.ndim == 4:
            x = x.mean(dim=2)
            return x.squeeze(1) if x.shape[1] == 1 else x
        return x


class _TokenMixing(nn.Module):
    def __init__(self, tokens, width, full_mixing):
        super().__init__()
        self.tokens, self.width, self.full_mixing = tokens, width, full_mixing
        if full_mixing:
            self.cross_mixing = nn.Linear(tokens * width, tokens * width)
            self.pre_norm = RankingRMSNorm(tokens * width)
        else:
            if width % tokens:
                raise ValueError("RankMixer requires token_dim % token_num == 0 (H=T)")
            self.layer_norm = nn.LayerNorm(width)

    def forward(self, x):
        if self.full_mixing:
            return x + self.cross_mixing(self.pre_norm(x.flatten(1))).reshape_as(x)
        mixed = x.reshape(x.shape[0], self.tokens, self.tokens, self.width // self.tokens)
        return self.layer_norm(x + mixed.transpose(1, 2).reshape_as(x))


class _TokenInteraction(nn.Module):
    def __init__(self, tokens, width, hidden, expansion, full_mixing):
        super().__init__()
        ffns = []
        for _ in range(tokens):
            if full_mixing:
                layers = [RankingRMSNorm(width), _PublishedGLU(width, expansion)]
                if hidden != width:
                    layers.append(nn.Linear(width, hidden))
                layers.extend([_MeanOn4(), nn.Linear(hidden, width)])
            else:
                layers = [nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width)]
            ffns.append(nn.Sequential(*layers))
        self.parallel_ffns = nn.ModuleList(ffns)

    def forward(self, x):
        return x + torch.stack([ffn(x[:, i]) for i, ffn in enumerate(self.parallel_ffns)], dim=1)


class _RankingBlock(nn.Module):
    def __init__(self, tokens, width, hidden, expansion, full_mixing):
        super().__init__()
        self.token_mixing = _TokenMixing(tokens, width, full_mixing)
        self.token_interaction = _TokenInteraction(tokens, width, hidden, expansion, full_mixing)
        self.layer_norm1 = RankingRMSNorm(width)

    def forward(self, x):
        return self.token_interaction(self.layer_norm1(self.token_mixing(x)))


class RankMixer(nn.Module):
    """Public CTR baseline with GenRec's (logits, loss) model interface.

    Numeric fields use bias-free linear embeddings, categorical fields use
    their training-fitted vocabularies. State names are compatible with the
    pinned public UniMixer model for explicit cross-implementation checks.
    """
    def __init__(self, feature_map, embedding_dim=20, token_num=15, token_dim=30,
                 num_blocks=2, ffn_hidden=None, tokenizer_hidden=500,
                 tokenizer_dropout=0.1, embedding_regularizer=1e-5,
                 _full_mixing=False, _expansion=1):
        super().__init__()
        self.feature_specs = ranking_feature_specs(feature_map)
        self.embedding_regularizer = embedding_regularizer
        if min(embedding_dim, token_num, token_dim, num_blocks, tokenizer_hidden) < 1:
            raise ValueError("model dimensions must be positive")
        hidden = token_dim if ffn_hidden is None else ffn_hidden
        self.embedding_layer = _FeatureEmbedding(self.feature_specs, embedding_dim)
        self.embedding_tokenization = _EmbeddingTokenizer(
            len(self.feature_specs), embedding_dim, token_num, token_dim,
            tokenizer_hidden, tokenizer_dropout, _full_mixing,
        )
        self.backbone = nn.ModuleList([
            _RankingBlock(token_num, token_dim, hidden, _expansion, _full_mixing)
            for _ in range(num_blocks)
        ])
        size = token_num * token_dim
        self.pred = nn.Sequential(nn.Flatten(), nn.Linear(size, size * 2), nn.ReLU(), nn.Linear(size * 2, 1))
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # FuxiCTR initializes categorical embeddings with N(0, 1e-4).
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                start = 1 if module.padding_idx == 0 else 0
                nn.init.normal_(module.weight.data[start:], mean=0.0, std=1e-4)

    def regularization_loss(self):
        weights = [m.weight.square().sum() for m in self.modules() if isinstance(m, nn.Embedding)]
        return self.embedding_regularizer * 0.5 * sum(weights)

    def forward(self, features, labels=None):
        x = self.embedding_tokenization(self.embedding_layer(features))
        for block in self.backbone:
            x = block(x)
        logits = self.pred(x)
        loss = None
        if labels is not None:
            # The public baseline trains BCE on sigmoid probabilities.
            # BCE on probabilities must execute outside CUDA autocast; existing
            # FP32 recipes retain exactly the same objective and values.
            with torch.autocast(device_type=logits.device.type, enabled=False):
                prediction = logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
                loss = F.binary_cross_entropy(prediction.sigmoid(), labels.float().reshape_as(logits))
            loss = loss + self.regularization_loss()
        return logits, loss


class RankElastor(RankMixer):
    """Released full-mixing / gated-FFN variant, including its documented skip."""
    def __init__(self, *args, expansion_rate=3, **kwargs):
        super().__init__(*args, _full_mixing=True, _expansion=expansion_rate, **kwargs)
