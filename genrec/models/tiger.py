"""
TIGER: Recommender Systems with Generative Retrieval.

Uses HuggingFace T5ForConditionalGeneration directly (matching the reference
implementation). Items are represented as flat offset-encoded codebook tokens:
    token = code + codebook_idx * codebook_size + 1
(0 is reserved for padding.)

Reference:
    TIGER: https://arxiv.org/abs/2305.05065
"""

import torch
import torch.nn as nn
from genrec.modules.t5 import T5ForConditionalGeneration, T5Config
from typing import Dict, Any


class Tiger(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        t5config = T5Config(
            num_layers=config.get('num_layers', 4),
            num_decoder_layers=config.get('num_decoder_layers', 4),
            d_model=config.get('d_model', 128),
            d_ff=config.get('d_ff', 1024),
            num_heads=config.get('num_heads', 6),
            d_kv=config.get('d_kv', 64),
            dropout_rate=config.get('dropout_rate', 0.1),
            vocab_size=config.get('vocab_size', 769),
            pad_token_id=config.get('pad_token_id', 0),
            eos_token_id=config.get('eos_token_id', 0),
            decoder_start_token_id=config.get('pad_token_id', 0),
            feed_forward_proj=config.get('feed_forward_proj', 'relu'),
        )
        self.model = T5ForConditionalGeneration(t5config)
        self.sem_id_dim = config.get('sem_id_dim', 3)

    @property
    def num_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb = sum(p.numel() for p in self.model.get_input_embeddings().parameters() if p.requires_grad)
        return total, emb

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs.loss, outputs.logits

    def generate(self, input_ids, attention_mask=None, num_beams=20):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.sem_id_dim + 1,  # +1 for decoder_start_token
            num_beams=num_beams,
            num_return_sequences=num_beams,
        )
