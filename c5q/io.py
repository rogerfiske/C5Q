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
import hashlib
import json
from datetime import datetime

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


def validate_cylindrical_adjacency(qs_values: np.ndarray) -> Tuple[bool, List[str]]:
    """
    Validate cylindrical adjacency patterns in QS values.

    Tests for quantum logic consistency in positional relationships
    where adjacent values should follow certain patterns.

    Args:
        qs_values: Array of QS values [QS_1, QS_2, QS_3, QS_4, QS_5]

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Cylindrical adjacency: positions wrap around (39 -> 1)
    def is_adjacent(a: int, b: int) -> bool:
        """Check if two positions are adjacent in cylindrical space."""
        return abs(a - b) == 1 or abs(a - b) == 38  # 39-1=38 for wrap-around

    # Check for invalid clustered patterns
    for i in range(len(qs_values) - 1):
        current = qs_values[i]
        next_val = qs_values[i + 1]

        # Check for over-clustering (too many adjacent values)
        adjacent_count = 0
        for j in range(len(qs_values)):
            if j != i and is_adjacent(current, qs_values[j]):
                adjacent_count += 1

        if adjacent_count > 2:  # Maximum 2 adjacent values per position
            errors.append(f"Position {current} has {adjacent_count} adjacent neighbors (max 2)")

    # Check for minimum spread requirement
    min_spread = 3  # Minimum distance between consecutive QS values
    for i in range(len(qs_values) - 1):
        current = qs_values[i]
        next_val = qs_values[i + 1]

        # Calculate minimum circular distance
        linear_dist = next_val - current
        circular_dist = min(linear_dist, 39 - linear_dist)

        if circular_dist < min_spread:
            errors.append(f"QS values {current} and {next_val} too close (distance {circular_dist}, min {min_spread})")

    return len(errors) == 0, errors


def generate_dataset_checksum(csv_path: str) -> Dict[str, Any]:
    """
    Generate SHA256 checksum and metadata for dataset integrity monitoring.

    Args:
        csv_path: Path to dataset CSV file

    Returns:
        Dictionary containing checksum and metadata
    """
    path = Path(csv_path)

    # Calculate file checksum
    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    # Get file stats
    stat = path.stat()

    checksum_data = {
        "file_path": str(path.absolute()),
        "file_name": path.name,
        "sha256": sha256_hash.hexdigest(),
        "file_size_bytes": stat.st_size,
        "modification_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "generated_at": datetime.now().isoformat(),
        "validation_version": "1.0"
    }

    return checksum_data


def comprehensive_dataset_validation(csv_path: str, save_checksum: bool = True) -> Dict[str, Any]:
    """
    Perform comprehensive validation of the C5 dataset with detailed reporting.

    Args:
        csv_path: Path to c5_Matrix_binary.csv file
        save_checksum: Whether to save checksum data to file

    Returns:
        Dictionary containing validation results and statistics
    """
    logger.info(f"Starting comprehensive validation of {csv_path}")

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    # Generate checksum
    checksum_data = generate_dataset_checksum(csv_path)

    # Load dataset
    df = pd.read_csv(csv_path)

    # Validation results
    validation_results = {
        "checksum": checksum_data,
        "dataset_info": {
            "total_events": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        },
        "validation_tests": {},
        "error_summary": {"total_errors": 0, "error_details": []},
        "validation_passed": True
    }

    try:
        # Test 1: Structure validation
        logger.info("Validating dataset structure...")
        _validate_dataset_structure(df)
        validation_results["validation_tests"]["structure"] = {"status": "PASSED", "errors": []}
        logger.info("Structure validation passed")

    except ValueError as e:
        validation_results["validation_tests"]["structure"] = {"status": "FAILED", "errors": [str(e)]}
        validation_results["error_summary"]["error_details"].append(f"Structure: {e}")
        validation_results["validation_passed"] = False
        logger.error(f"Structure validation failed: {e}")

    # Test 2: Basic integrity validation
    try:
        logger.info("Validating basic dataset integrity...")
        _validate_dataset_integrity(df)
        validation_results["validation_tests"]["basic_integrity"] = {"status": "PASSED", "errors": []}
        logger.info("Basic integrity validation passed")

    except ValueError as e:
        validation_results["validation_tests"]["basic_integrity"] = {"status": "FAILED", "errors": [str(e)]}
        validation_results["error_summary"]["error_details"].append(f"Basic Integrity: {e}")
        validation_results["validation_passed"] = False
        logger.error(f"Basic integrity validation failed: {e}")

    # Test 3: Advanced cylindrical adjacency validation
    logger.info("Validating cylindrical adjacency patterns...")
    qs_cols = [f"QS_{i}" for i in range(1, 6)]
    adjacency_errors = []

    for idx, row in df.iterrows():
        qs_values = row[qs_cols].values
        is_valid, errors = validate_cylindrical_adjacency(qs_values)
        if not is_valid:
            for error in errors:
                adjacency_errors.append(f"Event {row['event-ID']} (row {idx}): {error}")

    if adjacency_errors:
        validation_results["validation_tests"]["cylindrical_adjacency"] = {
            "status": "FAILED",
            "errors": adjacency_errors[:50]  # Limit to first 50 errors
        }
        validation_results["error_summary"]["error_details"].extend(adjacency_errors[:10])
        validation_results["validation_passed"] = False
        logger.warning(f"Cylindrical adjacency validation found {len(adjacency_errors)} issues")
    else:
        validation_results["validation_tests"]["cylindrical_adjacency"] = {"status": "PASSED", "errors": []}
        logger.info("Cylindrical adjacency validation passed")

    # Test 4: Feasibility constraints
    logger.info("Validating feasibility constraints...")
    feasibility_errors = []

    for idx, row in df.iterrows():
        qs_values = row[qs_cols].values
        if not validate_feasibility_constraints(qs_values):
            feasibility_errors.append(f"Event {row['event-ID']} (row {idx}): QS values {qs_values} violate feasibility constraints")

    if feasibility_errors:
        validation_results["validation_tests"]["feasibility"] = {
            "status": "FAILED",
            "errors": feasibility_errors[:50]
        }
        validation_results["error_summary"]["error_details"].extend(feasibility_errors[:10])
        validation_results["validation_passed"] = False
        logger.warning(f"Feasibility validation found {len(feasibility_errors)} issues")
    else:
        validation_results["validation_tests"]["feasibility"] = {"status": "PASSED", "errors": []}
        logger.info("Feasibility validation passed")

    # Update error summary
    validation_results["error_summary"]["total_errors"] = sum(
        len(test["errors"]) for test in validation_results["validation_tests"].values()
    )

    # Save checksum if requested
    if save_checksum:
        checksum_path = path.parent / f"{path.stem}_checksum.json"
        with open(checksum_path, "w") as f:
            json.dump(checksum_data, f, indent=2)
        logger.info(f"Checksum saved to {checksum_path}")

    # Final status
    if validation_results["validation_passed"]:
        logger.info("All dataset validations passed!")
    else:
        logger.error(f"Dataset validation failed with {validation_results['error_summary']['total_errors']} errors")

    return validation_results