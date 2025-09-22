"""
PyTorch Dataset classes for C5Q package.

Provides efficient data loading and preprocessing for quantum state
prediction with sliding window contexts and dual representation handling.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class C5QuantumDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for C5 Quantum Logic Matrix data.

    Supports sliding window contexts for sequence modeling with both
    QS (compact) and QV (binary) representations.
    """

    def __init__(
        self,
        csv_path: str,
        window_size: int = 128,
        stride: int = 1,
        prediction_horizon: int = 1
    ):
        """
        Initialize the dataset.

        Args:
            csv_path: Path to the primary dataset CSV
            window_size: Size of the context window
            stride: Stride for creating windows
            prediction_horizon: Number of future events to predict
        """
        self.window_size = window_size
        self.stride = stride
        self.prediction_horizon = prediction_horizon

        logger.info(f"Loading dataset from {csv_path}")
        self.df = pd.read_csv(csv_path)

        # Extract QS and QV columns
        self.qs_cols = [f"QS_{i}" for i in range(1, 6)]
        self.qv_cols = [f"QV_{i}" for i in range(1, 40)]

        # Convert to numpy arrays for efficiency
        self.qs_data = self.df[self.qs_cols].values.astype(np.int32)
        self.qv_data = self.df[self.qv_cols].values.astype(np.float32)

        # Create sliding windows
        self._create_windows()

        logger.info(f"Created {len(self.windows)} training samples with window size {window_size}")

    def _create_windows(self) -> None:
        """Create sliding windows for training."""
        self.windows = []
        self.targets = []

        n_events = len(self.df)
        max_start = n_events - self.window_size - self.prediction_horizon + 1

        for start_idx in range(0, max_start, self.stride):
            end_idx = start_idx + self.window_size
            target_idx = end_idx + self.prediction_horizon - 1

            # Context window (QV binary representation)
            context = self.qv_data[start_idx:end_idx]

            # Target (QS compact representation)
            target = self.qs_data[target_idx]

            self.windows.append(context)
            self.targets.append(target)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.int32)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (context_window, target_qs_values)
            - context_window: [window_size, 39] binary matrix
            - target_qs_values: [5] integer array of QS values
        """
        context = torch.from_numpy(self.windows[idx])  # [window_size, 39]
        target = torch.from_numpy(self.targets[idx])   # [5]

        return context, target

    def get_latest_context(self) -> torch.Tensor:
        """
        Get the latest context window for prediction.

        Returns:
            Latest context window as tensor [window_size, 39]
        """
        if len(self.df) < self.window_size:
            raise ValueError(f"Dataset has only {len(self.df)} events, need at least {self.window_size}")

        latest_context = self.qv_data[-self.window_size:]
        return torch.from_numpy(latest_context.astype(np.float32))

    def get_statistics(self) -> dict:
        """Get dataset statistics."""
        return {
            "total_events": len(self.df),
            "training_samples": len(self.windows),
            "window_size": self.window_size,
            "stride": self.stride,
            "prediction_horizon": self.prediction_horizon,
            "qs_value_ranges": {
                f"QS_{i+1}": (int(self.qs_data[:, i].min()), int(self.qs_data[:, i].max()))
                for i in range(5)
            },
            "qv_sparsity": float(self.qv_data.mean())  # Should be 5/39 ≈ 0.128
        }


class C5ValidationDataset(C5QuantumDataset):
    """
    Validation dataset with additional utilities for evaluation.
    """

    def __init__(
        self,
        csv_path: str,
        train_end_idx: int,
        window_size: int = 128,
        stride: int = 1
    ):
        """
        Initialize validation dataset.

        Args:
            csv_path: Path to the primary dataset CSV
            train_end_idx: Index where training data ends
            window_size: Size of the context window
            stride: Stride for creating windows
        """
        super().__init__(csv_path, window_size, stride)

        # Keep only validation samples
        self.train_end_idx = train_end_idx
        self._filter_validation_samples()

    def _filter_validation_samples(self) -> None:
        """Filter to keep only validation samples."""
        # Find windows that start after training cutoff
        valid_indices = []
        n_events = len(self.df)

        for i, start_idx in enumerate(range(0, len(self.windows) * self.stride, self.stride)):
            if start_idx >= self.train_end_idx:
                valid_indices.append(i)

        if valid_indices:
            self.windows = self.windows[valid_indices]
            self.targets = self.targets[valid_indices]
        else:
            self.windows = np.array([])
            self.targets = np.array([])

        logger.info(f"Validation dataset: {len(self.windows)} samples")


def create_train_val_split(
    csv_path: str,
    train_ratio: float = 0.8,
    window_size: int = 128,
    stride: int = 1,
    val_stride: int = 10
) -> Tuple[C5QuantumDataset, C5ValidationDataset]:
    """
    Create train/validation split of the dataset.

    Args:
        csv_path: Path to the primary dataset CSV
        train_ratio: Ratio of data to use for training
        window_size: Size of context windows
        stride: Stride for training samples
        val_stride: Stride for validation samples (usually larger)

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Load data to determine split point
    df = pd.read_csv(csv_path)
    n_events = len(df)
    train_end_idx = int(n_events * train_ratio)

    logger.info(f"Creating train/val split at event {train_end_idx}/{n_events}")

    # Create training dataset (full data with stride filtering)
    train_dataset = C5QuantumDataset(csv_path, window_size, stride)

    # Filter training samples to only include those before cutoff
    train_indices = []
    for i, start_idx in enumerate(range(0, len(train_dataset.windows) * stride, stride)):
        if start_idx < train_end_idx:
            train_indices.append(i)

    if train_indices:
        train_dataset.windows = train_dataset.windows[train_indices]
        train_dataset.targets = train_dataset.targets[train_indices]

    # Create validation dataset
    val_dataset = C5ValidationDataset(csv_path, train_end_idx, window_size, val_stride)

    logger.info(f"Split created: {len(train_dataset)} train, {len(val_dataset)} val samples")

    return train_dataset, val_dataset


def create_dataloader(
    dataset: C5QuantumDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader from a C5QuantumDataset.

    Args:
        dataset: C5QuantumDataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes

    Returns:
        PyTorch DataLoader
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )