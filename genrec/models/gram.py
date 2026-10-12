"""
GRAM: Generative Retrieval via Multi-Granularity Late Fusion (ACL 2025)

Based on T5 with Fusion-in-Decoder (FiD) style multi-granularity late fusion.
Each input passage (user prompt + per-item text) is encoded independently,
then concatenated before being fed to the decoder.

Position embeddings distinguish different item positions in the sequence.

Reference: https://github.com/phonism/GRAM
"""
import copy
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import BaseModelOutput


class EncoderWrapper(nn.Module):
    """
    Encoder Wrapper for T5 to obtain a Fusion-in-Decoder model.

    Each passage (user prompt or item text) is encoded independently,
    then their hidden states are concatenated along the sequence dimension
    for the decoder to attend to.

    Optionally adds position embeddings to distinguish different items.
    """

    def __init__(self, encoder, config=None, position_embedding=None):
        super().__init__()
        self.main_input_name = getattr(encoder, 'main_input_name', 'input_ids')
        self.encoder = encoder
        self.config = config
        self.position_embedding = position_embedding
        self.n_passages = 1  # set dynamically during forward

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if input_ids is not None:
            bsz, total_length = input_ids.shape  # B x (N * L)
            passage_length = total_length // self.n_passages  # L
            input_ids = input_ids.view(bsz * self.n_passages, passage_length)
            attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
            outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )
        elif inputs_embeds is not None:
            bsz, total_length, d = inputs_embeds.shape
            passage_length = total_length // self.n_passages
            inputs_embeds = inputs_embeds.view(bsz * self.n_passages, passage_length, d)
            attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
            outputs = self.encoder(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
            )
        else:
            raise ValueError("At least one of input_ids or inputs_embeds should be not None")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        last_hidden_states = outputs[0]  # (B * N) x L x D

        # Add position embeddings to distinguish different items
        if self.position_embedding is not None:
            position_ids = torch.arange(self.n_passages, device=device).expand(
                bsz, self.n_passages
            )
            position_embeddings = self.position_embedding(position_ids)  # B x N x D
            position_embeddings = position_embeddings.view(
                bsz * self.n_passages, 1, -1
            )  # (B * N) x 1 x D
            last_hidden_states = last_hidden_states + position_embeddings

        # Reshape back: (B * N) x L x D -> B x (N * L) x D
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(
            last_hidden_state=last_hidden_states.view(bsz, self.n_passages * passage_length, -1),
        )


class GRAM(T5ForConditionalGeneration):
    """
    GRAM model: T5 with FiD-style multi-granularity late fusion.

    Input is organized as B x N x L (batch x num_passages x passage_length):
    - Passage 0: User prompt with coarse-grained lexical ID history
    - Passage 1..N-1: Fine-grained item text descriptions with similar items

    The encoder processes each passage independently, then concatenates
    hidden states for the decoder. Position embeddings mark item positions.
    """

    def __init__(self, config):
        super().__init__(config)
        self.max_seq_len = getattr(config, 'max_seq_len', 128)
        self.max_item_num = getattr(config, 'max_item_num', 20)
        self.use_position_embedding = getattr(config, 'use_position_embedding', True)

        if self.use_position_embedding:
            pos_emb_size = self.max_item_num + 1  # +1 for user prompt
            self.position_embedding = nn.Embedding(pos_emb_size, config.d_model)
            nn.init.normal_(self.position_embedding.weight, std=0.02)
        else:
            self.position_embedding = None

        self._wrap_encoder()

    def _wrap_encoder(self):
        """Wrap T5 encoder to obtain FiD model."""
        self.encoder = EncoderWrapper(
            encoder=self.encoder,
            config=self.config,
            position_embedding=self.position_embedding,
        )

    def _unwrap_encoder(self):
        """Unwrap FiD encoder to load vanilla T5 weights."""
        self.encoder = self.encoder.encoder

    def load_t5(self, state_dict):
        """Load pretrained T5 weights into GRAM model."""
        self._unwrap_encoder()
        self.load_state_dict(state_dict, strict=False)
        self._wrap_encoder()

    def get_encoder(self):
        return self.encoder

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        Forward pass. Resizes inputs from B x N x L to B x (N*L) for T5,
        while EncoderWrapper handles the per-passage encoding internally.
        """
        if input_ids is not None:
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask is not None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)

        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

    def generate(self, input_ids, attention_mask, max_length, **kwargs):
        """
        Generate with FiD: encode all passages first, then decode.

        Pre-computes encoder outputs to avoid re-encoding during beam search.
        """
        self.encoder.n_passages = input_ids.size(1)

        # Flatten for encoder
        input_ids_flat = input_ids.view(input_ids.size(0), -1)
        attention_mask_flat = attention_mask.view(attention_mask.size(0), -1)

        # Pre-compute encoder outputs
        last_hidden_states = self.encoder(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat,
            return_dict=True,
        )[0]
        encoder_outputs = BaseModelOutput(last_hidden_state=last_hidden_states)

        return super().generate(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat,
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            **kwargs,
        )

    @classmethod
    def from_pretrained_t5(cls, model_name_or_path, max_item_num=20, use_position_embedding=True):
        """
        Create GRAM from a pretrained T5 model.

        Args:
            model_name_or_path: HuggingFace model name or path (e.g., "t5-small")
            max_item_num: Maximum number of history items
            use_position_embedding: Whether to use position embeddings

        Returns:
            GRAM model with pretrained T5 weights
        """
        config = T5Config.from_pretrained(model_name_or_path)
        config.max_seq_len = 128
        config.max_item_num = max_item_num
        config.use_position_embedding = use_position_embedding

        # Load vanilla T5 weights
        t5 = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        t5_state_dict = t5.state_dict()

        # Create GRAM and load T5 weights
        model = cls(config)
        model.load_t5(t5_state_dict)

        return model
