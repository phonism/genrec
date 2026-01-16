"""
GenRec: A Model Zoo for Generative Recommendation

This package provides implementations of various generative recommendation models:

Baseline Models:
    - SASRec: Self-Attentive Sequential Recommendation
    - HSTU: Hierarchical Sequential Transduction Unit

Generative Models:
    - RQVAE: Residual Quantized VAE for semantic ID generation
    - TIGER: Generative Retrieval with semantic IDs
    - LCRec: LLM-based recommendation with collaborative semantics
    - COBRA: Cascaded sparse-dense representations
    - NoteLLM: Note recommendation with LLMs
"""

__version__ = "0.1.0"
__author__ = "Qi Lu"

from genrec import models
from genrec import modules
from genrec import data
from genrec import trainers

__all__ = ["models", "modules", "data", "trainers", "__version__"]
