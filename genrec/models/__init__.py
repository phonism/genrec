"""
Model implementations for GenRec.

This module provides implementations of various recommendation models:

Baseline Models:
    - SASRec: Self-attention with softmax for sequential recommendation
    - HSTU: SiLU attention + update gate + temporal bias

Generative Models:
    - RQVAE: Vector quantized VAE for semantic ID generation
    - TIGER: Generative retrieval with trie-based constrained decoding
    - LCRec: LLM with collaborative semantics via codebook tokens
    - COBRA: Sparse/Dense hybrid with cascaded representations
    - NoteLLM: Qwen2-based LLM for note recommendation
"""

from importlib import import_module

# A small CTR/sequence model must not require every LLM/tokenizer dependency.
_EXPORTS = {
    "RqVae": "rqvae", "QuantizeForwardMode": "rqvae", "Tiger": "tiger",
    "SASRec": "sasrec", "HSTU": "hstu", "LCRec": "lcrec", "Cobra": "cobra",
    "GRAM": "gram", "FuXiAlpha": "fuxi_alpha", "RankMixer": "rankmixer",
    "RankElastor": "rankmixer",
    "TokenMixerLarge": "tokenmixer_large",
    "FuXiBeta": "fuxi_beta", "FuXiGamma": "fuxi_gamma", "Wukong": "wukong",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
