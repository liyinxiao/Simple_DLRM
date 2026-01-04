# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a simplified implementation of Facebook's Deep Learning Recommendation Model (DLRM) for personalization and recommendation systems. Based on the paper at https://arxiv.org/abs/1906.00091.

## Commands

### Setup and Run
```bash
python3 -m venv .env
source .env/bin/activate
pip install torch torchvision torchaudio
```

### Run Training
```bash
# Basic DLRM with BCE loss
python3 simple_dlrm.py

# DLRM with LambdaMART for NDCG optimization (ranking)
python3 lambdamart_dlrm.py
```

## Architecture

### Model Implementations

**`simple_dlrm.py`** - Basic DLRM with BCE loss:
- `DLRM` class implementing the standard architecture
- Bottom MLP processes dense features
- Embedding layers for sparse/categorical features
- Interaction layer (dot product or concatenation between dense and sparse embeddings)
- Top MLP produces final prediction with sigmoid output

**`lambdamart_dlrm.py`** - DLRM with LambdaRank for ranking:
- `SimpleDLRM` class using TorchRec-compatible components
- Session-based batching via `SessionBatchSampler` for listwise ranking
- `LambdaRankDataset` with dense feature normalization
- Mean pooling for multi-hot categorical features (via `KeyedJaggedTensor`)
- Optimizes NDCG directly using LambdaRank loss

### Supporting Modules

**`ndcg_loss.py`** - LambdaRank NDCG loss implementation:
- `LambdaRankNdcgLoss` module for listwise ranking optimization
- Requires samples from same session to be contiguous in batch
- Supports exponential gain and IDCG normalization

**`torchrec_compat.py`** - Pure PyTorch implementations of TorchRec components for macOS compatibility:
- `KeyedJaggedTensor` for variable-length sparse features
- `EmbeddingBagCollection` with configurable pooling (SUM/MEAN/MAX)
- `MLP` helper class
- Avoids fbgemm_gpu dependency

## Key Patterns

- Dense features are normalized using min-max scaling in `LambdaRankDataset`
- Sparse features use `KeyedJaggedTensor` format for variable-length category lists
- Session-based training requires sorted data by session_id for LambdaRank loss
- Interaction layer combines dense and sparse embeddings via dot product
