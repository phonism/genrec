"""
encoder module
"""
from torch import nn
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, T5Config
from genrec.modules.normalize import L2Norm
from typing import List
from torch import nn
from torch import Tensor


class LightT5Encoder(nn.Module):
    """
    Lightweight T5-style Encoder with configurable number of layers.
    Randomly initialized, no pretrained weights needed.
    """
    def __init__(
        self,
        n_layers: int = 1,
        hidden_dim: int = 768,
        output_dim: int = 768,
        num_heads: int = 8,
        ff_dim: int = 2048,
        vocab_size: int = 32128,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize a lightweight encoder (randomly initialized).

        Args:
            n_layers: number of encoder layers
            hidden_dim: hidden dimension
            output_dim: output embedding dimension
            num_heads: number of attention heads
            ff_dim: feed-forward dimension
            vocab_size: vocabulary size
            max_seq_len: maximum sequence length
            dropout: dropout rate
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.proj = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        print(f"Initialized LightT5Encoder with {n_layers} layers (random init)")

    def forward(self, batch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            batch_tokens: (B, T, L) or (B, L) - token IDs

        Returns:
            (B, T, D) or (B, D) - embeddings
        """
        if batch_tokens.dim() == 3:
            B, T, L = batch_tokens.shape
            flat_tokens = batch_tokens.view(B * T, L)
        elif batch_tokens.dim() == 2:
            B, L = batch_tokens.shape
            T = 1
            flat_tokens = batch_tokens
        else:
            raise ValueError(f"Expected 2D or 3D input, got {batch_tokens.dim()}D")

        # Embeddings
        positions = torch.arange(L, device=flat_tokens.device).unsqueeze(0)
        x = self.embedding(flat_tokens) + self.pos_embedding(positions)

        # Attention mask (True = ignore)
        attention_mask = (flat_tokens == 0)

        # Encode
        hidden = self.encoder(x, src_key_padding_mask=attention_mask)
        hidden = self.layer_norm(hidden)

        # Mean pooling
        mask = (~attention_mask).unsqueeze(-1).float()
        sum_hidden = (hidden * mask).sum(dim=1)
        pooled = sum_hidden / mask.sum(dim=1).clamp(min=1e-9)

        # Project and normalize
        projected = self.proj(pooled)
        normalized = F.normalize(projected, p=2, dim=-1)

        if T > 1:
            normalized = normalized.view(B, T, -1)

        return normalized

class SentenceT5Encoder(nn.Module):
    """
    Pretrained Sentence T5 Encoder (sentence-t5-base or sentence-t5-xl)
    """
    def __init__(self, model_name="./models_hub/sentence-t5-base", output_dim: int = 768) -> None:
        """
        Initialize the SentenceT5Encoder

        Args:
            model_name: path to the pretrained model
            output_dim: output embedding dimension (768 for base, 768 for xl after dense)
        """
        super().__init__()
        full_model = SentenceTransformer(model_name)

        # get submodules
        self.tokenizer = full_model.tokenizer
        self.encoder_model = full_model._modules["0"].auto_model.encoder
        self.pooling = full_model._modules["1"]

        # Check if dense layer exists (xl has it, base doesn't)
        if "2" in full_model._modules:
            self.dense = full_model._modules["2"]
            self.has_dense = True
        else:
            self.has_dense = False

        # Get encoder output dim
        encoder_dim = self.encoder_model.config.d_model  # 768 for base, 1024 for xl

        # Projection if needed
        if self.has_dense:
            self.proj = None  # dense layer handles projection
        else:
            # For base model, add projection if output_dim differs
            if encoder_dim != output_dim:
                self.proj = nn.Linear(encoder_dim, output_dim)
            else:
                self.proj = None

        self.output_dim = output_dim
        self.encoder_model.gradient_checkpointing_enable()
        print(f"Loaded pretrained SentenceT5Encoder from {model_name}, output_dim={output_dim}")

    def forward(self, batch_tokens):
        """
        Forward Pass

        Args:
            batch_tokens: (B, T, L) or (B, L) - token IDs
        Returns:
            (B, T, D) or (B, D) - embeddings
        """
        if batch_tokens.dim() == 3:
            B, T, L = batch_tokens.shape
            flat_tokens = batch_tokens.view(B * T, L)
        elif batch_tokens.dim() == 2:
            B, L = batch_tokens.shape
            T = 1
            flat_tokens = batch_tokens
        else:
            raise ValueError(f"Expect 2-D or 3-D, got {batch_tokens.dim()}-D")

        attention_mask = (flat_tokens != 0).long()

        # Encode
        encoder_outputs = self.encoder_model(
            input_ids=flat_tokens,
            attention_mask=attention_mask,
            return_dict=True,
        )
        token_embeddings = encoder_outputs.last_hidden_state

        # Pooling
        features = {
            "token_embeddings": token_embeddings,
            "attention_mask": attention_mask
        }
        pooled = self.pooling(features)["sentence_embedding"]

        # Dense projection (for xl model)
        if self.has_dense:
            pooled = self.dense({"sentence_embedding": pooled})["sentence_embedding"]
        elif self.proj is not None:
            pooled = self.proj(pooled)

        # Normalize
        normalized = F.normalize(pooled, p=2, dim=-1)

        if T > 1:
            normalized = normalized.view(B, T, -1)
        return normalized


class ErnieEncoder(nn.Module):
    """
    Ernie Encoder
    """
    def __init__(self, model_name="./models_hub/ernie-3.0-medium-zh", out_dim=768) -> None:
        """
        Initialize the ErnieEncoder

        Args:
            model_name: path to the model
            out_dim: output dimension
        """
        super().__init__()
        self.out_dim = out_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.pooler = None
        #self.backbone.pooler.dense.bias.requires_grad_(False)
        #self.backbone.pooler.dense.weight.requires_grad_(False)
        #self.backbone.gradient_checkpointing_enable()
    
    def encode(self, texts) -> torch.Tensor:
        """
        encode texts to embeddings

        Args:
            texts: (B, T)
        Returns:
            torch.Tensor
        """
        batch_tokens = self.tokenize(texts)
        return self.forward(batch_tokens["input_ids"]).view(-1, self.out_dim)
    
    def tokenize(self, texts) -> torch.Tensor:
        """
        Tokenize texts

        Args:
            texts: (B, T)
        Returns:
            torch.Tensor
        """
        if isinstance(texts, str):
            texts = [[texts]]                      # (1,1)
        elif isinstance(texts[0], str):
            texts = [[t] for t in texts]           # (B,1)

        B, T = len(texts), len(texts[0])           # (B,T)
        flat_texts = [t for seq in texts for t in seq]     # length B*T

        batch_tokens = self.tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=102400,
            return_tensors="pt"
        ).to(self.backbone.device)

        return batch_tokens

    def forward(self, batch_tokens) -> torch.Tensor:
        """
        Forward Pass

        Args:
            batch_tokens: (B, T, C)
        Returns:
            torch.Tensor
        """
        if batch_tokens.dim() == 3:
            B, T, C = batch_tokens.shape
            flat_tokens = batch_tokens.view(B * T, C)       # (B*T, C)
        elif batch_tokens.dim() == 2:
            B, C = batch_tokens.shape
            T = 1
            flat_tokens = batch_tokens                      # (B, C)
        else:
            raise ValueError(f"Expect 2-D or 3-D, got {batch_tokens.dim()}-D")
        attention_mask = (flat_tokens != 0).long()          # (B*T, C) or (B, C)

        # -------- backbone forward --------
        outputs = self.backbone(
            input_ids=flat_tokens,
            attention_mask=attention_mask,
            return_dict=True,
        )
        h_cls = outputs.last_hidden_state[:, 0]             # (B*T, H)

        z = h_cls

        if T > 1:                                           # if it is 3-D
            z = z.view(B, T, -1)                            # (B, T, D)
        return z                                            # (B, T, D) or (B, D)


class BgeEncoder(nn.Module):
    """
    BGE Encoder
    """
    def __init__(self, model_name="./models_hub/bge-base-zh"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.pooler = None
        self.backbone.gradient_checkpointing_enable()
        self.out_dim = self.backbone.config.hidden_size

    def tokenize(self, texts) -> torch.Tensor:
        """
        Tokenize texts

        Args:
            texts: (B, T)
        Returns:
            torch.Tensor
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt").to(self.backbone.device)
    
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids
    ) -> torch.Tensor:
        """
        Forward Pass

        Args:
            input_ids: (B, T, C)
            attention_mask: (B, T, C)
            token_type_ids: (B, T, C)
        Returns:
            torch.Tensor
        """
        if input_ids.dim() == 3:
            B, T, C = input_ids.shape
            flat_tokens = input_ids.view(B * T, C)
            flat_attention_mask = attention_mask.view(B * T, C)
            flat_token_type_ids = token_type_ids.view(B * T, C)
        elif input_ids.dim() == 2:
            B, C = input_ids.shape
            T = 1
            flat_tokens = input_ids
            flat_attention_mask = attention_mask.view(B * T, C)
            flat_token_type_ids = token_type_ids.view(B * T, C)
        else:
            raise ValueError(f"Expect 2-D or 3-D, got {input_ids.dim()}-D")
        outputs = self.backbone(
            input_ids=flat_tokens,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids
        )
        hidden_state = outputs.last_hidden_state[:, 0]
        hidden_state = torch.nn.functional.normalize(hidden_state, p=2, dim=-1)
        if T > 1:
            hidden_state = hidden_state.view(B, T, -1)
        return hidden_state

    @torch.no_grad()
    def encode(self, texts, device="cuda") -> torch.Tensor:
        """
        Encode texts to embeddings

        Args:
            texts: (B, T)
            device: device
        Returns:
            torch.Tensor
        """
        batch = self.tokenize(texts)
        return self.forward(**batch)


class MLP(nn.Module):
    """
    MLP Layer
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        out_dim: int,
        dropout: float = 0.0,
        normalize: bool = False
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.dropout = dropout

        dims = [self.input_dim] + self.hidden_dims + [self.out_dim]
        
        self.mlp = nn.Sequential()
        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:])):
            self.mlp.append(nn.Linear(in_d, out_d, bias=False))
            if i != len(dims) - 2:
                self.mlp.append(nn.SiLU())
                if dropout != 0:
                    self.mlp.append(nn.Dropout(dropout))
        self.mlp.append(L2Norm() if normalize else nn.Identity())

    def forward(self, x: Tensor) -> torch.Tensor:
        """
        Forward Pass

        Args:
            x: (B, D)
        Returns:
            torch.Tensor
        """
        assert x.shape[-1] == self.input_dim, f"Invalid input dim: Expected {self.input_dim}, found {x.shape[-1]}"
        return self.mlp(x)

if __name__ == "__main__":
    bge = SentenceTransformer("./models_hub/bge-base-zh")
    bge_new = BgeEncoder("./models_hub/bge-base-zh")
    print(bge.encode(["hello world"])[0][:10])
    print(bge_new.encode(["hello world"])[0][:10])