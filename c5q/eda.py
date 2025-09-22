"""
Exploratory Data Analysis module for C5Q package.

Provides comprehensive analysis of the quantum logic matrix dataset
including entropy analysis, bucket clustering, and pattern detection.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def run_comprehensive_eda(
    df: pd.DataFrame,
    output_dir: str,
    k_values: List[int] = [4, 5, 6]
) -> Dict[str, Any]:
    """
    Run complete exploratory data analysis on C5 dataset.

    Args:
        df: Primary dataset DataFrame
        output_dir: Directory to save analysis artifacts
        k_values: List of k values for clustering analysis

    Returns:
        Dictionary containing all analysis results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting comprehensive EDA analysis")

    results = {
        "dataset_info": _analyze_dataset_info(df),
        "position_analysis": _analyze_position_entropy(df),
        "temporal_analysis": _analyze_temporal_patterns(df),
        "clustering_analysis": _analyze_clustering(df, k_values),
        "baseline_predictions": _generate_baseline_predictions(df)
    }

    # Save analysis results
    _save_eda_artifacts(results, output_path)

    logger.info(f"EDA analysis complete. Results saved to {output_path}")
    return results


def _analyze_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze basic dataset information."""
    qs_cols = [f"QS_{i}" for i in range(1, 6)]
    qv_cols = [f"QV_{i}" for i in range(1, 40)]

    return {
        "total_events": len(df),
        "date_range": {
            "start_event": int(df["event-ID"].min()),
            "end_event": int(df["event-ID"].max())
        },
        "columns": {
            "total": len(df.columns),
            "qs_columns": len(qs_cols),
            "qv_columns": len(qv_cols)
        },
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
    }


def _analyze_position_entropy(df: pd.DataFrame) -> Dict[str, float]:
    """Analyze entropy for each QS position."""
    qs_cols = [f"QS_{i}" for i in range(1, 6)]
    entropies = {}

    for col in qs_cols:
        values = df[col].values
        counts = np.bincount(values, minlength=40)[1:40]  # Exclude 0, include 1-39
        probabilities = counts / counts.sum()
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        entropy = -(probabilities * np.log2(probabilities)).sum()
        entropies[col] = float(entropy)

    return entropies


def _analyze_temporal_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze temporal patterns in the dataset."""
    qs_cols = [f"QS_{i}" for i in range(1, 6)]
    n_events = len(df)

    # Divide into 5 temporal segments
    segment_size = n_events // 5
    segments = []

    for i in range(5):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < 4 else n_events
        segment_data = df.iloc[start_idx:end_idx]

        # Get bottom 20 for this segment
        all_values = segment_data[qs_cols].values.flatten()
        bottom_20 = _get_bottom_20(all_values)

        segments.append({
            "segment": i + 1,
            "start_event": int(segment_data["event-ID"].iloc[0]),
            "end_event": int(segment_data["event-ID"].iloc[-1]),
            "event_count": len(segment_data),
            "bottom_20": bottom_20
        })

    return {
        "segments": segments,
        "global_bottom_20": _get_bottom_20(df[qs_cols].values.flatten())
    }


def _analyze_clustering(df: pd.DataFrame, k_values: List[int]) -> Dict[str, Any]:
    """Analyze state clustering using co-occurrence patterns."""
    qv_cols = [f"QV_{i}" for i in range(1, 40)]
    qv_data = df[qv_cols].values

    # Compute Jaccard similarity matrix
    jaccard_matrix = _compute_jaccard_similarity(qv_data)

    # Perform clustering for each k value
    clustering_results = {}
    for k in k_values:
        clusters = _kmeans_clustering(jaccard_matrix, k)
        clustering_results[f"k_{k}"] = {
            "clusters": clusters,
            "cluster_sizes": [len(cluster) for cluster in clusters]
        }

    return {
        "jaccard_similarity": jaccard_matrix.tolist(),
        "clustering": clustering_results
    }


def _generate_baseline_predictions(df: pd.DataFrame) -> Dict[str, List[int]]:
    """Generate baseline least-20 predictions using frequency analysis."""
    qs_cols = [f"QS_{i}" for i in range(1, 6)]
    all_values = df[qs_cols].values.flatten()

    # Simple frequency-based baseline
    frequency_bottom_20 = _get_bottom_20(all_values)

    # Position-aware baseline
    position_baselines = {}
    for i, col in enumerate(qs_cols, 1):
        position_values = df[col].values
        position_bottom_20 = _get_bottom_20(position_values)
        position_baselines[f"position_{i}"] = position_bottom_20

    return {
        "frequency_baseline": frequency_bottom_20,
        "position_baselines": position_baselines
    }


def _get_bottom_20(values: np.ndarray) -> List[int]:
    """Get the 20 least frequent values."""
    counter = Counter(values)
    # Get all possible values 1-39
    all_values = list(range(1, 40))
    # Sort by frequency (ascending), then by value (ascending) for ties
    sorted_values = sorted(all_values, key=lambda x: (counter.get(x, 0), x))
    return sorted_values[:20]


def _compute_jaccard_similarity(qv_data: np.ndarray) -> np.ndarray:
    """Compute Jaccard similarity matrix for QV columns."""
    n_states = qv_data.shape[1]
    similarity_matrix = np.zeros((n_states, n_states))

    for i in range(n_states):
        for j in range(n_states):
            vec_i = qv_data[:, i].astype(bool)
            vec_j = qv_data[:, j].astype(bool)

            intersection = np.logical_and(vec_i, vec_j).sum()
            union = np.logical_or(vec_i, vec_j).sum()

            if union > 0:
                similarity_matrix[i, j] = intersection / union
            else:
                similarity_matrix[i, j] = 0.0

    return similarity_matrix


def _kmeans_clustering(similarity_matrix: np.ndarray, k: int, max_iters: int = 100) -> List[List[int]]:
    """Perform k-means clustering on similarity matrix."""
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    n_points = distance_matrix.shape[0]

    # Initialize random centroids
    np.random.seed(42)  # For reproducibility
    centroids = np.random.choice(n_points, k, replace=False)

    for _ in range(max_iters):
        # Assign points to closest centroid
        assignments = []
        for i in range(n_points):
            distances = [distance_matrix[i, c] for c in centroids]
            assignments.append(np.argmin(distances))

        # Update centroids
        new_centroids = []
        for cluster_id in range(k):
            cluster_points = [i for i, assignment in enumerate(assignments) if assignment == cluster_id]
            if cluster_points:
                # Find point with minimum average distance to all points in cluster
                best_centroid = cluster_points[0]
                best_distance = float('inf')
                for candidate in cluster_points:
                    avg_distance = np.mean([distance_matrix[candidate, p] for p in cluster_points])
                    if avg_distance < best_distance:
                        best_distance = avg_distance
                        best_centroid = candidate
                new_centroids.append(best_centroid)
            else:
                # Keep old centroid if cluster is empty
                new_centroids.append(centroids[cluster_id])

        # Check for convergence
        if set(new_centroids) == set(centroids):
            break

        centroids = new_centroids

    # Group points by cluster (convert to 1-indexed)
    clusters = [[] for _ in range(k)]
    for i, assignment in enumerate(assignments):
        clusters[assignment].append(i + 1)

    return clusters


def _save_eda_artifacts(results: Dict[str, Any], output_path: Path) -> None:
    """Save EDA results to files."""
    # Save main summary
    with open(output_path / "eda_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save global bottom 20
    global_bottom_20 = results["temporal_analysis"]["global_bottom_20"]
    pd.DataFrame({"global_bottom20": global_bottom_20}).to_csv(
        output_path / "global_bottom20.csv", index=False
    )

    # Save segmented bottom 20
    for segment in results["temporal_analysis"]["segments"]:
        segment_id = segment["segment"]
        bottom_20 = segment["bottom_20"]
        pd.DataFrame({f"segment_{segment_id}_bottom20": bottom_20}).to_csv(
            output_path / f"seg{segment_id}_bottom20.csv", index=False
        )

    # Save clustering results
    for k_name, clustering_data in results["clustering_analysis"]["clustering"].items():
        clusters = clustering_data["clusters"]
        cluster_assignments = []
        states = []
        for cluster_idx, cluster in enumerate(clusters):
            for state in cluster:
                cluster_assignments.append(cluster_idx + 1)
                states.append(state)

        pd.DataFrame({
            "cluster": cluster_assignments,
            "state": states
        }).to_csv(output_path / f"buckets_{k_name}.csv", index=False)

    logger.info(f"EDA artifacts saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run C5Q EDA analysis")
    parser.add_argument("--csv", required=True, help="Path to c5_Matrix_binary.csv")
    parser.add_argument("--out", default="artifacts/eda", help="Output directory")
    parser.add_argument("--k", type=int, nargs="+", default=[4, 5, 6], help="K values for clustering")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Load dataset and run analysis
    from .io import load_primary_dataset
    df = load_primary_dataset(args.csv)
    results = run_comprehensive_eda(df, args.out, args.k)

    print(f"EDA analysis complete. Results saved to {args.out}")
    print(f"Total events analyzed: {results['dataset_info']['total_events']}")
    print(f"Global bottom-20: {results['temporal_analysis']['global_bottom_20'][:10]}...")