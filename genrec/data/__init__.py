"""
Data loading and preprocessing utilities for GenRec.

This module provides dataset implementations for various recommendation scenarios:

Datasets:
    - AmazonItemDataset: Item-level data for RQVAE training
    - AmazonSeqDataset: Sequential data for TIGER training
    - AmazonLCRecDataset: SFT format data for LCRec training
    - AmazonCobraDataset: Interleaved sparse-dense data for COBRA
    - AmazonSASRecDataset: Leave-one-out data for SASRec
    - AmazonHSTUDataset: Temporal data for HSTU

Data Schemas:
    - SeqData: Named tuple for sequence data
    - SeqBatch: Batched sequence data
    - TokenizedSeqBatch: Tokenized batch for transformer models
"""

from genrec.data.schemas import SeqData, SeqBatch, TokenizedSeqBatch
from genrec.data.utils import cycle

__all__ = [
    "SeqData",
    "SeqBatch",
    "TokenizedSeqBatch",
    "cycle",
]
