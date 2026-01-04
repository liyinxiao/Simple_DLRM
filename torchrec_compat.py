# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchRec Compatibility Layer for macOS

This module provides pure PyTorch implementations of torchrec components
that work on macOS without requiring fbgemm_gpu.

Implements:
- PoolingType: Enum for embedding pooling strategies
- EmbeddingBagConfig: Configuration for embedding tables
- EmbeddingBagCollection: Collection of embedding bag tables
- MLP: Multi-layer perceptron
- KeyedJaggedTensor: Jagged tensor with named keys
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch
from torch import nn


class PoolingType(Enum):
    """Pooling type for embedding bags."""
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"


@dataclass
class EmbeddingBagConfig:
    """
    Configuration for an embedding bag table.

    Args:
        name: Name of the embedding table
        embedding_dim: Dimension of the embeddings
        num_embeddings: Number of unique embeddings (vocabulary size)
        feature_names: List of feature names that use this table
        pooling: Pooling type (SUM, MEAN, or MAX)
    """
    name: str
    embedding_dim: int
    num_embeddings: int
    feature_names: List[str] = field(default_factory=list)
    pooling: PoolingType = PoolingType.MEAN


class KeyedJaggedTensor:
    """
    A jagged tensor with named keys.

    This is a simplified implementation that stores:
    - keys: List of feature names
    - values: 1D tensor of all values concatenated
    - lengths: 1D tensor indicating the length of each row
    - offsets: Computed offsets for indexing

    Example:
        For 3 samples with category_ids [1,2], [3], [4,5,6]:
        - keys = ["category_ids"]
        - values = tensor([1, 2, 3, 4, 5, 6])
        - lengths = tensor([2, 1, 3])
    """

    def __init__(
        self,
        keys: List[str],
        values: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        self._keys = keys
        self._values = values
        self._lengths = lengths
        # Compute offsets from lengths
        self._offsets = torch.zeros(len(lengths) + 1, dtype=torch.long, device=lengths.device)
        self._offsets[1:] = torch.cumsum(lengths.long(), dim=0)

    @staticmethod
    def from_lengths_sync(
        keys: List[str],
        values: torch.Tensor,
        lengths: torch.Tensor,
    ) -> "KeyedJaggedTensor":
        """Create a KeyedJaggedTensor from keys, values, and lengths."""
        return KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)

    def keys(self) -> List[str]:
        """Return the list of keys."""
        return self._keys

    def values(self) -> torch.Tensor:
        """Return the values tensor."""
        return self._values

    def lengths(self) -> torch.Tensor:
        """Return the lengths tensor."""
        return self._lengths

    def offsets(self) -> torch.Tensor:
        """Return the offsets tensor."""
        return self._offsets

    def __getitem__(self, key: str) -> torch.Tensor:
        """Get values for a specific key (for compatibility)."""
        if key not in self._keys:
            raise KeyError(f"Key '{key}' not found in KeyedJaggedTensor")
        return self._values

    def length_per_key(self) -> List[int]:
        """Return the total length for each key."""
        return [int(self._lengths.sum().item())]

    def to(self, device: torch.device) -> "KeyedJaggedTensor":
        """Move the tensor to the specified device."""
        return KeyedJaggedTensor(
            keys=self._keys,
            values=self._values.to(device),
            lengths=self._lengths.to(device),
        )


class KeyedTensor:
    """
    A keyed tensor that maps feature names to their embeddings.

    Used as the output of EmbeddingBagCollection.
    """

    def __init__(
        self,
        keys: List[str],
        length_per_key: List[int],
        values: torch.Tensor,
    ) -> None:
        self._keys = keys
        self._length_per_key = length_per_key
        self._values = values

        # Build key to column offset mapping
        self._key_to_offset: Dict[str, int] = {}
        offset = 0
        for key, length in zip(keys, length_per_key):
            self._key_to_offset[key] = offset
            offset += length

    def __getitem__(self, key: str) -> torch.Tensor:
        """Get the embedding tensor for a specific key."""
        if key not in self._key_to_offset:
            raise KeyError(f"Key '{key}' not found in KeyedTensor")

        offset = self._key_to_offset[key]
        idx = self._keys.index(key)
        length = self._length_per_key[idx]

        return self._values[:, offset:offset + length]

    def keys(self) -> List[str]:
        """Return the list of keys."""
        return self._keys

    def values(self) -> torch.Tensor:
        """Return the values tensor."""
        return self._values


class EmbeddingBagCollection(nn.Module):
    """
    Collection of embedding bag tables.

    This is a simplified implementation that wraps PyTorch's nn.EmbeddingBag
    for each configured table.

    Args:
        tables: List of EmbeddingBagConfig for each table
        is_weighted: Whether to use weighted embeddings (not implemented)
        device: Device to place embeddings on
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.tables = tables
        self.is_weighted = is_weighted
        self._device = device

        # Create embedding bags for each table
        self.embedding_bags = nn.ModuleDict()
        self._feature_to_table: Dict[str, str] = {}
        self._table_configs: Dict[str, EmbeddingBagConfig] = {}

        for config in tables:
            # Map pooling type to PyTorch mode
            if config.pooling == PoolingType.SUM:
                mode = "sum"
            elif config.pooling == PoolingType.MEAN:
                mode = "mean"
            elif config.pooling == PoolingType.MAX:
                mode = "max"
            else:
                mode = "mean"

            # Create embedding bag
            self.embedding_bags[config.name] = nn.EmbeddingBag(
                num_embeddings=config.num_embeddings,
                embedding_dim=config.embedding_dim,
                mode=mode,
                device=device,
            )

            # Map feature names to table
            self._table_configs[config.name] = config
            for feature_name in config.feature_names:
                self._feature_to_table[feature_name] = config.name

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        """
        Forward pass: look up and pool embeddings for each feature.

        Args:
            features: KeyedJaggedTensor containing sparse feature values

        Returns:
            KeyedTensor containing pooled embeddings for each feature
        """
        keys = []
        lengths = []
        embeddings_list = []

        for feature_name in features.keys():
            if feature_name not in self._feature_to_table:
                raise ValueError(f"Feature '{feature_name}' not found in any table")

            table_name = self._feature_to_table[feature_name]
            embedding_bag = self.embedding_bags[table_name]
            config = self._table_configs[table_name]

            # Get values and offsets for this feature
            values = features.values()
            offsets = features.offsets()

            # Look up embeddings with pooling
            pooled = embedding_bag(values, offsets[:-1])

            keys.append(feature_name)
            lengths.append(config.embedding_dim)
            embeddings_list.append(pooled)

        # Concatenate all embeddings along the feature dimension
        if embeddings_list:
            combined = torch.cat(embeddings_list, dim=1)
        else:
            combined = torch.empty(0)

        return KeyedTensor(
            keys=keys,
            length_per_key=lengths,
            values=combined,
        )


class MLP(nn.Module):
    """
    Multi-layer perceptron.

    Args:
        in_size: Input dimension
        layer_sizes: List of hidden layer sizes
        bias: Whether to use bias in linear layers
        activation: Activation function ("relu", "sigmoid", "tanh", or None)
        device: Device to place layers on
    """

    def __init__(
        self,
        in_size: int,
        layer_sizes: List[int],
        bias: bool = True,
        activation: str = "relu",
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        # Build layers
        layers: List[nn.Module] = []
        prev_size = in_size

        for i, layer_size in enumerate(layer_sizes):
            layers.append(nn.Linear(prev_size, layer_size, bias=bias, device=device))

            # Add activation after each layer except the last
            if i < len(layer_sizes) - 1:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif activation == "tanh":
                    layers.append(nn.Tanh())

            prev_size = layer_size

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self.mlp(x)
