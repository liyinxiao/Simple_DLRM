# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Simple DLRM Model Example with LambdaMART for Place Search Ranking

This example demonstrates how to create and train a DLRM model with:
- A few dense features (e.g., price, rating, distance)
- One categorical feature (category_ids) that can have 1 or multiple IDs
- Mean pooling for the embedding aggregation
- LambdaMART loss to optimize for NDCG (listwise ranking)
- Session-based batching for search ranking

The model learns to rank places for a given search session.
"""

import random
from typing import Any, Iterator, Optional

import pandas as pd
import torch

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Sampler
from ndcg_loss import LambdaRankNdcgLoss
from torchrec_compat import (
    EmbeddingBagConfig,
    EmbeddingBagCollection,
    KeyedJaggedTensor,
    MLP,
    PoolingType,
)


class SimpleDLRM(nn.Module):
    """
    A simple DLRM model with dense features and one categorical feature.

    This model follows the DLRM architecture:
    1. Dense features are processed through a bottom MLP to produce dense embeddings
    2. Categorical features are looked up in embedding tables and pooled (using mean)
    3. Dense and sparse embeddings interact via dot products
    4. The interaction output goes through a top MLP to produce final predictions

    Args:
        num_dense_features (int): Number of dense/continuous input features
        embedding_dim (int): Dimension of the embeddings (must be same for dense and sparse)
        num_embeddings (int): Number of unique category IDs for category_ids
        dense_hidden_layers (list[int]): Hidden layer sizes for the bottom MLP
        top_hidden_layers (list[int]): Hidden layer sizes for the top MLP
    """

    def __init__(
        self,
        num_dense_features: int,
        embedding_dim: int,
        num_embeddings: int,
        dense_hidden_layers: list[int],
        top_hidden_layers: list[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        # Bottom MLP: transforms dense features to embedding_dim
        self.dense_arch: nn.Module = MLP(
            in_size=num_dense_features,
            layer_sizes=[*dense_hidden_layers, embedding_dim],
            bias=True,
            activation="relu",
            device=device,
        )

        # Embedding table for category_ids with mean pooling
        # Mean pooling will average embeddings when multiple category IDs are present
        self.embedding_bag_collection: EmbeddingBagCollection = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="category_table",
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embeddings,
                    feature_names=["category_ids"],
                    pooling=PoolingType.MEAN,  # Use mean pooling for multi-hot features
                ),
            ],
            is_weighted=False,
            device=device,
        )

        # Top MLP: processes interaction output to produce final prediction
        # Input size = embedding_dim (dense) + 1 (dot product between dense and sparse)
        interaction_output_dim = (
            embedding_dim + 1
        )  # dense + (dense, sparse) interaction
        self.top_arch: nn.Module = nn.Sequential(
            MLP(
                in_size=interaction_output_dim,
                layer_sizes=top_hidden_layers,
                bias=True,
                device=device,
            ),
            nn.Linear(top_hidden_layers[-1], 1, bias=True, device=device),
        )

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Forward pass of the SimpleDLRM model.

        Args:
            dense_features (torch.Tensor): Dense input features of shape (batch_size, num_dense_features)
            sparse_features (KeyedJaggedTensor): Sparse features containing category_ids

        Returns:
            torch.Tensor: Predictions of shape (batch_size,)
        """
        # Process dense features through bottom MLP
        dense_embeddings = self.dense_arch(dense_features)  # (B, D)

        # Look up and pool sparse embeddings
        sparse_embeddings_kt = self.embedding_bag_collection(sparse_features)
        sparse_embeddings = sparse_embeddings_kt["category_ids"]  # (B, D)

        # Compute interaction: dot product between dense and sparse embeddings
        interaction = torch.sum(
            dense_embeddings * sparse_embeddings, dim=1, keepdim=True
        )  # (B, 1)

        # Concatenate dense embeddings with interaction
        combined = torch.cat([dense_embeddings, interaction], dim=1)  # (B, D + 1)

        # Final prediction through top MLP
        logits = self.top_arch(combined).squeeze(-1)  # (B,)

        return logits


def create_sample_dataframe(
    num_sessions: int,
    min_places_per_session: int,
    max_places_per_session: int,
    num_categories: int,
    num_dense_features: int,
    max_categories_per_place: int = 4,
    train_ratio: float = 0.8,
) -> pd.DataFrame:
    """
    Create a sample pandas DataFrame with dummy data for DLRM place search ranking.

    Each session represents a user search query, and each row represents a place
    that could be shown to the user. The label represents the relevance grade
    (0-4, where 4 is most relevant) for NDCG optimization.

    The DataFrame has the following columns:
    - session_id: ID grouping places shown for the same search query
    - feature_1, feature_2, ..., feature_n: Dense features (float)
    - category_ids: List of category IDs (variable length list of ints)
    - label: Relevance grade (0-4) for ranking
    - type: "train" or "test" indicating the data split

    Args:
        num_sessions: Number of search sessions to generate
        min_places_per_session: Minimum number of places per session
        max_places_per_session: Maximum number of places per session
        num_categories: Maximum category ID value (exclusive)
        num_dense_features: Number of dense features (default 3)
        max_categories_per_place: Maximum number of categories per place
        train_ratio: Ratio of sessions to use for training (default 0.8)

    Returns:
        pd.DataFrame with the sample data
    """
    random.seed(42)

    # Determine train/test split at session level
    num_train_sessions = int(num_sessions * train_ratio)
    session_types = ["train"] * num_train_sessions + ["test"] * (
        num_sessions - num_train_sessions
    )
    random.shuffle(session_types)

    # Initialize data dictionary with session_id and dense feature columns
    data: dict[str, list[Any]] = {
        "session_id": [],
    }
    # Add dense feature columns: feature_1, feature_2, ..., feature_n
    dense_feature_cols = [f"feature_{i+1}" for i in range(num_dense_features)]
    for col in dense_feature_cols:
        data[col] = []
    data["category_ids"] = []
    data["label"] = []
    data["type"] = []

    for session_idx in range(num_sessions):
        # Random number of places per session
        num_places = random.randint(min_places_per_session, max_places_per_session)
        session_type = session_types[session_idx]

        for _ in range(num_places):
            data["session_id"].append(session_idx)
            # Generate random values for each dense feature
            for col in dense_feature_cols:
                data[col].append(random.uniform(0, 100))
            data["category_ids"].append(
                [
                    random.randint(0, num_categories - 1)
                    for _ in range(random.randint(1, max_categories_per_place))
                ]
            )
            # Relevance grade 0-4 (for NDCG: 0=irrelevant, 4=highly relevant)
            data["label"].append(random.randint(0, 4))
            data["type"].append(session_type)

    return pd.DataFrame(data)


class LambdaRankDataset(Dataset[dict[str, torch.Tensor | list[int] | int]]):
    """
    PyTorch Dataset for DLRM with LambdaRank that wraps a pandas DataFrame.

    This dataset handles:
    - Dense features: Normalized continuous values
    - Sparse features: Variable-length lists of category IDs
    - Labels: Relevance grades for ranking (0-4)
    - Session IDs: Grouping identifier for listwise ranking

    Args:
        df: pandas DataFrame containing the training data
        dense_feature_cols: List of column names for dense features
        sparse_feature_col: Column name for sparse features (list of IDs)
        label_col: Column name for the label (relevance grade)
        session_id_col: Column name for session ID
        dense_min: Optional pre-computed min values for normalization (use training stats)
        dense_range: Optional pre-computed range values for normalization (use training stats)
    """

    df: pd.DataFrame
    dense_feature_cols: list[str]
    sparse_feature_col: str
    label_col: str
    session_id_col: str
    dense_min_tensor: torch.Tensor
    dense_range_tensor: torch.Tensor

    def __init__(
        self,
        df: pd.DataFrame,
        dense_feature_cols: list[str],
        sparse_feature_col: str,
        label_col: str,
        session_id_col: str,
        dense_min: Optional[torch.Tensor] = None,
        dense_range: Optional[torch.Tensor] = None,
    ) -> None:
        # Sort by session_id to ensure examples from same session are contiguous
        # This is required by LambdaRankNdcgLoss
        self.df = df.sort_values(by=session_id_col).reset_index(drop=True)
        self.dense_feature_cols = dense_feature_cols
        self.sparse_feature_col = sparse_feature_col
        self.label_col = label_col
        self.session_id_col = session_id_col

        # Use provided normalization stats or compute from this dataset
        if dense_min is not None and dense_range is not None:
            # Use provided stats (e.g., from training set)
            self.dense_min_tensor = dense_min
            self.dense_range_tensor = dense_range
        else:
            # Compute normalization stats from this dataset
            dense_data = self.df[dense_feature_cols].values.astype("float32")
            computed_min = dense_data.min(axis=0)
            computed_max = dense_data.max(axis=0)
            computed_range = computed_max - computed_min
            # Avoid division by zero
            computed_range[computed_range == 0] = 1.0
            # Pre-convert to tensors for efficient __getitem__
            self.dense_min_tensor = torch.tensor(computed_min, dtype=torch.float32)
            self.dense_range_tensor = torch.tensor(computed_range, dtype=torch.float32)

    def get_normalization_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalization stats for use with other datasets (e.g., test set)."""
        return self.dense_min_tensor, self.dense_range_tensor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | list[int] | int]:
        row = self.df.iloc[idx]

        # Get dense features and normalize using pre-computed tensors
        dense_values = [row[col] for col in self.dense_feature_cols]
        dense_array = torch.tensor(dense_values, dtype=torch.float32)
        dense_normalized = (dense_array - self.dense_min_tensor) / self.dense_range_tensor

        # Get sparse features (list of category IDs)
        sparse_ids: list[int] = row[self.sparse_feature_col]

        # Get label (relevance grade 0-4)
        label = torch.tensor(row[self.label_col], dtype=torch.float32)

        # Get session ID
        session_id: int = row[self.session_id_col]

        return {
            "dense_features": dense_normalized,
            "sparse_ids": sparse_ids,
            "label": label,
            "session_id": session_id,
        }


class SessionBatchSampler(Sampler[list[int]]):
    """
    A batch sampler that groups training samples from the same session together.

    This sampler ensures that all samples from a session are included in the same batch,
    which is required for listwise ranking losses like LambdaRank. It packs as many
    complete sessions as possible into each batch without exceeding the max batch size.

    Args:
        session_ids: List of session IDs for each sample (must be sorted by session)
        max_batch_size: Maximum number of samples per batch.
        drop_last: If True, drop the last batch (which likely has fewer samples than other batches)
    """

    def __init__(
        self,
        session_ids: list[int],
        max_batch_size: Optional[int] = None,
        drop_last: bool = False,
    ) -> None:
        self.session_ids = session_ids
        self.max_batch_size = max_batch_size
        self.drop_last = drop_last

        # Precompute session boundaries and batches
        self.session_batches: list[list[int]] = []
        self._compute_session_batches()

    def _compute_session_batches(self) -> None:
        """Compute batches where each batch contains complete sessions."""
        if len(self.session_ids) == 0:
            return

        current_batch: list[int] = []
        current_session = self.session_ids[0]
        session_start = 0

        def should_start_new_batch(
            batch_size: int, session_size: int, max_size: Optional[int]
        ) -> bool:
            """Check if we should start a new batch given the constraints."""
            if max_size is None:
                return False
            return batch_size > 0 and batch_size + session_size > max_size

        for i, session_id in enumerate(self.session_ids):
            if session_id != current_session:
                # Session boundary found - process the completed session
                session_indices = list(range(session_start, i))

                if should_start_new_batch(
                    len(current_batch), len(session_indices), self.max_batch_size
                ):
                    # Save current batch and start new one with this session
                    self.session_batches.append(current_batch)
                    current_batch = session_indices
                else:
                    current_batch.extend(session_indices)

                session_start = i
                current_session = session_id

        # Handle the last session
        session_indices = list(range(session_start, len(self.session_ids)))
        if should_start_new_batch(
            len(current_batch), len(session_indices), self.max_batch_size
        ):
            self.session_batches.append(current_batch)
            current_batch = session_indices
        else:
            current_batch.extend(session_indices)

        # Add the final batch if not empty
        if current_batch:
            self.session_batches.append(current_batch)

    def __iter__(self) -> Iterator[list[int]]:
        if self.drop_last and len(self.session_batches) > 0:
            yield from self.session_batches[:-1]
        else:
            yield from self.session_batches

    def __len__(self) -> int:
        if self.drop_last and len(self.session_batches) > 0:
            return len(self.session_batches) - 1
        return len(self.session_batches)


def lambdarank_collate_fn(
    batch: list[dict[str, torch.Tensor | list[int] | int]],
) -> tuple[torch.Tensor, KeyedJaggedTensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for LambdaRank DLRM DataLoader.

    This function:
    1. Stacks dense features into a batch tensor
    2. Converts variable-length sparse ID lists into a KeyedJaggedTensor
    3. Stacks labels into a batch tensor
    4. Stacks session IDs into a batch tensor (for LambdaRank grouping)

    Args:
        batch: List of samples from LambdaRankDataset.__getitem__

    Returns:
        Tuple of (dense_features, sparse_features, labels, session_ids)
    """
    # Stack dense features: (B, num_dense_features)
    dense_features = torch.stack(
        [sample["dense_features"] for sample in batch]  # pyre-ignore[6]
    )

    # Build KeyedJaggedTensor from variable-length sparse ID lists
    all_values: list[int] = []
    lengths: list[int] = []

    for sample in batch:
        sparse_ids: list[int] = sample["sparse_ids"]  # pyre-ignore[9]
        all_values.extend(sparse_ids)
        lengths.append(len(sparse_ids))

    sparse_features = KeyedJaggedTensor.from_lengths_sync(
        keys=["category_ids"],
        values=torch.tensor(all_values, dtype=torch.int64),
        lengths=torch.tensor(lengths, dtype=torch.int32),
    )

    # Stack labels: (B,)
    labels = torch.stack([sample["label"] for sample in batch])  # pyre-ignore[6]

    # Stack session IDs: (B,) - needed for LambdaRank loss
    session_ids = torch.tensor(
        [sample["session_id"] for sample in batch], dtype=torch.int32
    )

    return dense_features, sparse_features, labels, session_ids


def compute_ndcg(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    session_ids: torch.Tensor,
    k_values: list[int],
) -> dict[int, float]:
    """
    Compute NDCG@k for multiple k values in a single pass.

    Args:
        predictions: Model predictions (scores)
        labels: Relevance labels (grades 0-4)
        session_ids: Session IDs for grouping
        k_values: List of k values to compute NDCG for

    Returns:
        Dictionary mapping k values to average NDCG@k across all sessions
    """
    unique_sessions = torch.unique(session_ids)
    # Initialize accumulators for each k
    ndcg_sums: dict[int, float] = {k: 0.0 for k in k_values}
    ndcg_counts: dict[int, int] = {k: 0 for k in k_values}

    for session_id in unique_sessions:
        mask = session_ids == session_id
        session_preds = predictions[mask]
        session_labels = labels[mask]

        if len(session_labels) == 0:
            continue

        # Sort by predictions (descending)
        _, pred_order = torch.sort(session_preds, descending=True)
        sorted_labels = session_labels[pred_order]

        # Sort labels for ideal ranking
        ideal_labels, _ = torch.sort(session_labels, descending=True)

        # Compute NDCG for each k value
        session_len = len(sorted_labels)
        for k in k_values:
            k_actual = min(k, session_len)
            positions = torch.arange(1, k_actual + 1, dtype=torch.float32)
            discounts = torch.log2(positions + 1)

            # DCG@k
            gains = (torch.pow(2, sorted_labels[:k_actual]) - 1) / discounts
            dcg = torch.sum(gains).item()

            # IDCG@k
            ideal_gains = (torch.pow(2, ideal_labels[:k_actual]) - 1) / discounts
            idcg = torch.sum(ideal_gains).item()

            # NDCG@k
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_sums[k] += ndcg
            ndcg_counts[k] += 1

    # Compute averages
    return {
        k: ndcg_sums[k] / ndcg_counts[k] if ndcg_counts[k] > 0 else 0.0
        for k in k_values
    }


def evaluate_model_ndcg(
    model: SimpleDLRM,
    dataloader: DataLoader[dict[str, torch.Tensor | list[int] | int]],
    k_values: list[int],
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """
    Evaluate the DLRM model using NDCG for ranking.

    Args:
        model: The trained SimpleDLRM model
        dataloader: DataLoader for the evaluation dataset
        k_values: List of k values to compute NDCG for
        device: Device to run evaluation on (defaults to CPU if not specified)

    Returns:
        Dictionary containing evaluation metrics (ndcg@k for each k in k_values)
    """
    if device is None:
        device = torch.device("cpu")

    model.eval()

    all_predictions = []
    all_labels = []
    all_session_ids = []

    with torch.no_grad():
        for dense_features, sparse_features, labels, session_ids in dataloader:
            predictions = model(dense_features, sparse_features)
            # Move predictions back to CPU for NDCG computation
            all_predictions.append(predictions.cpu())
            all_labels.append(labels)
            all_session_ids.append(session_ids)

    # Concatenate all batches
    all_predictions_tensor = torch.cat(all_predictions)
    all_labels_tensor = torch.cat(all_labels)
    all_session_ids_tensor = torch.cat(all_session_ids)

    # Compute NDCG for all k values in a single pass
    ndcg_results = compute_ndcg(
        all_predictions_tensor, all_labels_tensor, all_session_ids_tensor, k_values=k_values
    )

    return {f"ndcg@{k}": v for k, v in ndcg_results.items()}


def train_simple_dlrm() -> SimpleDLRM:
    """
    Train a simple DLRM model using LambdaMART loss for NDCG optimization.

    Uses train/test split based on the 'type' column in the DataFrame.
    Sessions are kept together (no session spans train and test).

    Returns:
        The trained SimpleDLRM model
    """
    # Model configuration
    num_dense_features = 217  # e.g., price, rating, distance
    embedding_dim = 16
    num_embeddings = 1850  # Number of unique place categories
    dense_hidden_layers = [32, 16]
    top_hidden_layers = [16, 8]

    # Training configuration
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Create sample DataFrame with train/test split
    print("\nCreating sample DataFrame for place search ranking...")
    num_sessions = 200  # Number of search sessions
    min_places_per_session = 5
    max_places_per_session = 15
    df = create_sample_dataframe(
        num_sessions=num_sessions,
        min_places_per_session=min_places_per_session,
        max_places_per_session=max_places_per_session,
        num_categories=num_embeddings,
        num_dense_features=num_dense_features,
        train_ratio=0.8,
    )

    # Generate dense feature column names
    dense_feature_cols = [f"feature_{i+1}" for i in range(num_dense_features)]

    # Split into train and test DataFrames
    train_df = df[df["type"] == "train"].copy()
    test_df = df[df["type"] == "test"].copy()

    num_train_sessions = train_df["session_id"].nunique()
    num_test_sessions = test_df["session_id"].nunique()

    print(f"  Total samples: {len(df)}")
    print(f"  Total sessions: {num_sessions}")
    print(f"  Training samples: {len(train_df)} ({num_train_sessions} sessions)")
    print(f"  Test samples: {len(test_df)} ({num_test_sessions} sessions)")
    print("  Sample row:")
    print(f"    session_id: {df['session_id'].iloc[0]}")
    for col in dense_feature_cols[:3]:  # Show first 3 features as sample
        print(f"    {col}: {df[col].iloc[0]:.2f}")
    if num_dense_features > 3:
        print(f"    ... ({num_dense_features - 3} more dense features)")
    print(f"    category_ids: {df['category_ids'].iloc[0]}")
    print(f"    label (relevance): {df['label'].iloc[0]}")
    print(f"    type: {df['type'].iloc[0]}")

    # Create Training Dataset and DataLoader with dynamic session-based batching
    train_dataset = LambdaRankDataset(
        df=train_df,
        dense_feature_cols=dense_feature_cols,
        sparse_feature_col="category_ids",
        label_col="label",
        session_id_col="session_id",
    )

    # Get session IDs from the sorted dataset for the batch sampler
    train_session_ids = train_dataset.df["session_id"].tolist()

    # Get normalization stats from training set for use with test set
    train_dense_min, train_dense_range = train_dataset.get_normalization_stats()

    # Create SessionBatchSampler to group samples from the same session together
    # This ensures LambdaRank loss can properly compute pairwise gradients within sessions
    train_batch_sampler = SessionBatchSampler(
        session_ids=train_session_ids,
        max_batch_size=batch_size,
        drop_last=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,  # Use session-based batch sampler
        collate_fn=lambdarank_collate_fn,
        num_workers=0,
    )

    # Create Test Dataset and DataLoader with dynamic session-based batching
    # Use training set normalization stats to avoid data leakage
    test_dataset = LambdaRankDataset(
        df=test_df,
        dense_feature_cols=dense_feature_cols,
        sparse_feature_col="category_ids",
        label_col="label",
        session_id_col="session_id",
        dense_min=train_dense_min,
        dense_range=train_dense_range,
    )

    # Get session IDs from the sorted test dataset
    test_session_ids = test_dataset.df["session_id"].tolist()

    # Create SessionBatchSampler for test data
    test_batch_sampler = SessionBatchSampler(
        session_ids=test_session_ids,
        max_batch_size=batch_size,
        drop_last=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_sampler=test_batch_sampler,  # Use session-based batch sampler
        collate_fn=lambdarank_collate_fn,
        num_workers=0,
    )

    print(f"  Training batches per epoch: {len(train_dataloader)}")
    print(f"  Test batches: {len(test_dataloader)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = SimpleDLRM(
        num_dense_features=num_dense_features,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        dense_hidden_layers=dense_hidden_layers,
        top_hidden_layers=top_hidden_layers,
        device=device,
    )

    # LambdaMART Loss for NDCG optimization
    criterion = LambdaRankNdcgLoss(
        use_ndcg_as_loss=True,
        use_exp_gain=True,
        use_idcg_normalization=True,
        reduction="mean",
    )
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop using DataLoader
    print("\nStarting LambdaMART training for NDCG optimization...")
    print("=" * 60)
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        num_batches = 0

        for dense_features, sparse_features, labels, session_ids in train_dataloader:
            # Forward pass
            optimizer.zero_grad()
            predictions = model(dense_features, sparse_features)

            # Compute LambdaRank NDCG loss
            loss = criterion(
                logit=predictions,
                label=labels,
                session_ids=session_ids,
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Evaluation phase (every epoch) - compute all NDCG@k values in single pass
        test_metrics = evaluate_model_ndcg(
            model, test_dataloader, k_values=[5, 10, 30], device=device
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Test NDCG@5: {test_metrics['ndcg@5']:.4f} "
            f"Test NDCG@10: {test_metrics['ndcg@10']:.4f} "
            f"Test NDCG@30: {test_metrics['ndcg@30']:.4f}"
        )

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set:")
    print("=" * 60)
    final_metrics = evaluate_model_ndcg(
        model, test_dataloader, k_values=[5, 10], device=device
    )
    print(f"  NDCG@5:  {final_metrics['ndcg@5']:.4f}")
    print(f"  NDCG@10: {final_metrics['ndcg@10']:.4f}")

    return model


def main() -> None:
    """Main function to demonstrate the SimpleDLRM model with LambdaMART."""
    print("Training Simple DLRM Model with LambdaMART for Place Search")
    print("=" * 60)
    print("Configuration:")
    print("  - Dense features: N (feature_1, feature_2, ..., feature_N)")
    print("  - Categorical feature: category_ids (multi-hot)")
    print("  - Pooling: MEAN (averages embeddings for multiple categories)")
    print("  - Loss: LambdaRank NDCG (listwise ranking)")
    print("  - Optimization target: NDCG")
    print("=" * 60)

    # Train the model
    model = train_simple_dlrm()

    # Demonstrate inference
    print("\nInference Example (Ranking places for a search session):")
    print("-" * 60)

    model.eval()
    num_dense_features = 217  # Must match model configuration
    with torch.no_grad():
        # Create 3 candidate places for ranking
        # Dense features: feature_1, feature_2, ..., feature_n (normalized)
        dense = torch.rand(3, num_dense_features)  # Random normalized features for demo

        # Sparse features for 3 places
        sparse = KeyedJaggedTensor.from_lengths_sync(
            keys=["category_ids"],
            values=torch.tensor([5, 23, 12, 45, 7]),  # Categories for all places
            lengths=torch.tensor([2, 1, 2]),  # Place 1: 2, Place 2: 1, Place 3: 2
        )

        predictions = model(dense, sparse)

        # Rank places by predicted score
        ranking_order = torch.argsort(predictions, descending=True)

        print("  Candidate Places (with predicted ranking scores):")
        for idx, score in enumerate(predictions.tolist()):
            print(f"    Place {idx + 1}: Score = {score:.4f}")

        print("\n  Recommended Ranking (best to worst):")
        for rank, place_idx in enumerate(ranking_order.tolist()):
            print(
                f"    Rank {rank + 1}: Place {place_idx + 1} "
                f"(score: {predictions[place_idx].item():.4f})"
            )


if __name__ == "__main__":
    main()