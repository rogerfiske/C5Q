"""
Data I/O module for C5Q package.

Handles loading and validation of the C5 Quantum Logic Matrix dataset
with comprehensive integrity checks and error reporting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_primary_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load and validate the primary C5 dataset.

    Args:
        csv_path: Path to c5_Matrix_binary.csv file

    Returns:
        Validated DataFrame with QS and QV columns

    Raises:
        ValueError: If dataset validation fails
        FileNotFoundError: If dataset file not found
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)

    # Validate dataset structure
    _validate_dataset_structure(df)
    _validate_dataset_integrity(df)

    logger.info(f"Successfully loaded dataset with {len(df)} events")
    return df


def _validate_dataset_structure(df: pd.DataFrame) -> None:
    """Validate that dataset has required columns."""
    expected_columns = (
        ["event-ID"] +
        [f"QS_{i}" for i in range(1, 6)] +
        [f"QV_{i}" for i in range(1, 40)]
    )

    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    logger.debug("Dataset structure validation passed")


def _validate_dataset_integrity(df: pd.DataFrame) -> None:
    """Validate QS/QV consistency and constraints."""
    qs_cols = [f"QS_{i}" for i in range(1, 6)]
    qv_cols = [f"QV_{i}" for i in range(1, 40)]

    qs_data = df[qs_cols].to_numpy()
    qv_data = df[qv_cols].to_numpy()

    errors = []

    for idx, (qs_row, qv_row) in enumerate(zip(qs_data, qv_data)):
        # Check ascending order
        if not all(qs_row[i] < qs_row[i+1] for i in range(4)):
            errors.append(f"Row {idx}: QS values not strictly ascending")

        # Check range bounds (1-39)
        if not all(1 <= val <= 39 for val in qs_row):
            errors.append(f"Row {idx}: QS values out of range [1-39]")

        # Check QV has exactly 5 ones
        if qv_row.sum() != 5:
            errors.append(f"Row {idx}: QV has {qv_row.sum()} ones, expected 5")

        # Check QS matches QV positions
        qv_positions = set(np.where(qv_row == 1)[0] + 1)  # Convert to 1-indexed
        qs_positions = set(qs_row)
        if qv_positions != qs_positions:
            errors.append(f"Row {idx}: QS positions {qs_positions} don't match QV positions {qv_positions}")

    if errors:
        error_msg = "\n".join(errors[:10])  # Show first 10 errors
        if len(errors) > 10:
            error_msg += f"\n... and {len(errors) - 10} more errors"
        raise ValueError(f"Dataset integrity validation failed:\n{error_msg}")

    logger.debug("Dataset integrity validation passed")


def load_qs_binary_files(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load the QS-specific binary files if available.

    Args:
        data_dir: Directory containing QS binary files

    Returns:
        Dictionary mapping QS position to DataFrame
    """
    data_path = Path(data_dir)
    qs_files = {}

    for i in range(1, 6):
        file_path = data_path / f"QS{i}_binary_matrix.csv"
        if file_path.exists():
            logger.info(f"Loading {file_path}")
            qs_files[f"QS_{i}"] = pd.read_csv(file_path)

    return qs_files


def validate_feasibility_constraints(qs_values: np.ndarray) -> bool:
    """
    Validate that QS values respect positional feasibility constraints.

    Args:
        qs_values: Array of QS values [QS_1, QS_2, QS_3, QS_4, QS_5]

    Returns:
        True if constraints are satisfied

    Feasibility ranges:
    - QS_1: 1-35
    - QS_2: 2-36
    - QS_3: 3-37
    - QS_4: 4-38
    - QS_5: 5-39
    """
    constraints = [
        (1, 35),   # QS_1
        (2, 36),   # QS_2
        (3, 37),   # QS_3
        (4, 38),   # QS_4
        (5, 39),   # QS_5
    ]

    for i, (min_val, max_val) in enumerate(constraints):
        if not (min_val <= qs_values[i] <= max_val):
            return False

    return True