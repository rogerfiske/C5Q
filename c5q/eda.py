"""
Exploratory Data Analysis module for C5Q package.

Provides comprehensive analysis of the quantum logic matrix dataset
including entropy analysis, bucket clustering, and pattern detection.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Suppress matplotlib warnings on headless systems
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

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
        "statistical_summaries": _analyze_statistical_summaries(df),
        "temporal_analysis": _analyze_temporal_patterns(df),
        "adjacency_patterns": _analyze_cylindrical_adjacency_patterns(df),
        "clustering_analysis": _analyze_clustering(df, k_values),
        "baseline_predictions": _generate_baseline_predictions(df)
    }

    # Generate visualizations
    visualizations_created = _create_visualizations(results, output_path)
    results["visualizations"] = visualizations_created

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


def _analyze_position_entropy(df: pd.DataFrame) -> Dict[str, Any]:
    """Enhanced entropy analysis for each QS position with detailed statistics."""
    qs_cols = [f"QS_{i}" for i in range(1, 6)]
    entropy_analysis = {
        "position_entropies": {},
        "entropy_statistics": {},
        "value_distributions": {},
        "theoretical_max_entropy": np.log2(39)  # Max entropy for 39 possible values
    }

    all_entropies = []

    for col in qs_cols:
        values = df[col].values

        # Calculate entropy
        counts = np.bincount(values, minlength=40)[1:40]  # Exclude 0, include 1-39
        probabilities = counts / counts.sum()
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        entropy = -(probabilities * np.log2(probabilities)).sum()

        # Calculate additional statistics
        unique_values = len(probabilities)
        most_frequent = values.argmax() if len(values) > 0 else None
        least_frequent = values.argmin() if len(values) > 0 else None

        # Normalized entropy (0-1 scale)
        normalized_entropy = entropy / entropy_analysis["theoretical_max_entropy"]

        entropy_analysis["position_entropies"][col] = float(entropy)
        entropy_analysis["value_distributions"][col] = {
            "unique_values_count": int(unique_values),
            "most_frequent_value": int(np.argmax(counts) + 1),  # Convert back to 1-indexed
            "most_frequent_count": int(np.max(counts)),
            "least_frequent_value": int(np.argmin(counts[counts > 0]) + 1) if np.any(counts > 0) else None,
            "least_frequent_count": int(np.min(counts[counts > 0])) if np.any(counts > 0) else None,
            "mean_value": float(values.mean()),
            "std_value": float(values.std()),
            "median_value": float(np.median(values)),
            "value_range": [int(values.min()), int(values.max())],
            "normalized_entropy": float(normalized_entropy)
        }

        all_entropies.append(entropy)

    # Overall entropy statistics
    entropy_analysis["entropy_statistics"] = {
        "mean_entropy": float(np.mean(all_entropies)),
        "std_entropy": float(np.std(all_entropies)),
        "min_entropy": float(np.min(all_entropies)),
        "max_entropy": float(np.max(all_entropies)),
        "entropy_range": float(np.max(all_entropies) - np.min(all_entropies)),
        "coefficient_of_variation": float(np.std(all_entropies) / np.mean(all_entropies)) if np.mean(all_entropies) > 0 else 0.0
    }

    return entropy_analysis


def _analyze_statistical_summaries(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive statistical analysis of quantum states."""
    qs_cols = [f"QS_{i}" for i in range(1, 6)]
    qv_cols = [f"QV_{i}" for i in range(1, 40)]

    # QS statistical summaries
    qs_stats = {}
    for col in qs_cols:
        values = df[col].values
        qs_stats[col] = {
            "count": len(values),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": int(values.min()),
            "max": int(values.max()),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "skewness": float(stats.skew(values)),
            "kurtosis": float(stats.kurtosis(values)),
            "mode": int(stats.mode(values, keepdims=False)[0]),
            "variance": float(values.var()),
            "range": int(values.max() - values.min()),
            "iqr": float(np.percentile(values, 75) - np.percentile(values, 25))
        }

    # QV statistical summaries (co-occurrence patterns)
    qv_data = df[qv_cols].values
    qv_stats = {
        "total_activations": int(qv_data.sum()),
        "activations_per_event": {
            "mean": float(qv_data.sum(axis=1).mean()),
            "std": float(qv_data.sum(axis=1).std()),
            "expected": 5.0  # Should always be 5
        },
        "activation_rates_per_position": {},
        "position_correlations": {}
    }

    # Per-position activation rates
    for i, col in enumerate(qv_cols):
        activation_rate = qv_data[:, i].mean()
        qv_stats["activation_rates_per_position"][col] = {
            "activation_rate": float(activation_rate),
            "total_activations": int(qv_data[:, i].sum()),
            "z_score": float((activation_rate - (5/39)) / np.sqrt((5/39) * (1 - 5/39) / len(df)))
        }

    # Overall dataset statistics
    all_qs_values = df[qs_cols].values.flatten()
    overall_stats = {
        "total_qs_values": len(all_qs_values),
        "unique_values_used": len(np.unique(all_qs_values)),
        "coverage_percentage": float(len(np.unique(all_qs_values)) / 39 * 100),
        "most_frequent_values": _get_most_frequent_values(all_qs_values, n=10),
        "least_frequent_values": _get_least_frequent_values(all_qs_values, n=10),
        "value_frequency_distribution": _analyze_frequency_distribution(all_qs_values)
    }

    return {
        "qs_position_statistics": qs_stats,
        "qv_activation_statistics": qv_stats,
        "overall_statistics": overall_stats
    }


def _analyze_cylindrical_adjacency_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze cylindrical adjacency effects in quantum state patterns."""
    qs_cols = [f"QS_{i}" for i in range(1, 6)]

    def is_adjacent(a: int, b: int) -> bool:
        """Check if two positions are adjacent in cylindrical space."""
        return abs(a - b) == 1 or abs(a - b) == 38  # 39-1=38 for wrap-around

    def circular_distance(a: int, b: int) -> int:
        """Calculate minimum circular distance between two positions."""
        linear_dist = abs(a - b)
        return min(linear_dist, 39 - linear_dist)

    adjacency_analysis = {
        "adjacent_pairs_count": 0,
        "total_pairs_count": 0,
        "adjacency_rate": 0.0,
        "distance_distribution": {},
        "position_adjacency_patterns": {},
        "cluster_analysis": {}
    }

    adjacent_pairs = 0
    total_pairs = 0
    distance_counts = Counter()
    position_adjacencies = {col: {"adjacent_count": 0, "total_occurrences": 0} for col in qs_cols}

    for _, row in df.iterrows():
        qs_values = row[qs_cols].values

        # Analyze all pairs within this event
        for i in range(len(qs_values)):
            for j in range(i + 1, len(qs_values)):
                val_i, val_j = qs_values[i], qs_values[j]
                distance = circular_distance(val_i, val_j)
                distance_counts[distance] += 1

                total_pairs += 1
                if is_adjacent(val_i, val_j):
                    adjacent_pairs += 1

                # Track position-specific adjacencies
                col_i, col_j = qs_cols[i], qs_cols[j]
                position_adjacencies[col_i]["total_occurrences"] += 1
                position_adjacencies[col_j]["total_occurrences"] += 1

                if is_adjacent(val_i, val_j):
                    position_adjacencies[col_i]["adjacent_count"] += 1
                    position_adjacencies[col_j]["adjacent_count"] += 1

    # Calculate rates and patterns
    adjacency_analysis["adjacent_pairs_count"] = adjacent_pairs
    adjacency_analysis["total_pairs_count"] = total_pairs
    adjacency_analysis["adjacency_rate"] = adjacent_pairs / total_pairs if total_pairs > 0 else 0.0

    # Distance distribution
    for distance in range(1, 20):  # Only meaningful distances up to 19
        adjacency_analysis["distance_distribution"][f"distance_{distance}"] = distance_counts.get(distance, 0)

    # Position-specific patterns
    for col in qs_cols:
        data = position_adjacencies[col]
        adjacency_analysis["position_adjacency_patterns"][col] = {
            "adjacent_count": data["adjacent_count"],
            "total_occurrences": data["total_occurrences"],
            "adjacency_rate": data["adjacent_count"] / data["total_occurrences"] if data["total_occurrences"] > 0 else 0.0
        }

    # Clustering analysis
    cluster_sizes = []
    for _, row in df.iterrows():
        qs_values = sorted(row[qs_cols].values)
        current_cluster = [qs_values[0]]

        for i in range(1, len(qs_values)):
            if circular_distance(qs_values[i-1], qs_values[i]) <= 2:  # Adjacent or close
                current_cluster.append(qs_values[i])
            else:
                if len(current_cluster) > 1:
                    cluster_sizes.append(len(current_cluster))
                current_cluster = [qs_values[i]]

        if len(current_cluster) > 1:
            cluster_sizes.append(len(current_cluster))

    adjacency_analysis["cluster_analysis"] = {
        "total_clusters_found": len(cluster_sizes),
        "average_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "cluster_size_distribution": {str(k): int(v) for k, v in Counter(cluster_sizes).items()}
    }

    return adjacency_analysis


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


def _get_most_frequent_values(values: np.ndarray, n: int = 10) -> List[Dict[str, Any]]:
    """Get the n most frequent values with their counts."""
    counter = Counter(values)
    most_common = counter.most_common(n)
    return [{"value": int(val), "count": count, "frequency": count/len(values)} for val, count in most_common]


def _get_least_frequent_values(values: np.ndarray, n: int = 10) -> List[Dict[str, Any]]:
    """Get the n least frequent values with their counts."""
    counter = Counter(values)
    # Get values sorted by frequency (ascending)
    least_common = sorted(counter.items(), key=lambda x: (x[1], x[0]))[:n]
    return [{"value": int(val), "count": count, "frequency": count/len(values)} for val, count in least_common]


def _analyze_frequency_distribution(values: np.ndarray) -> Dict[str, Any]:
    """Analyze the frequency distribution of values."""
    counter = Counter(values)
    frequencies = list(counter.values())

    return {
        "unique_values": len(counter),
        "frequency_stats": {
            "mean": float(np.mean(frequencies)),
            "std": float(np.std(frequencies)),
            "min": int(min(frequencies)),
            "max": int(max(frequencies)),
            "median": float(np.median(frequencies))
        },
        "frequency_distribution": {str(k): int(v) for k, v in counter.items()}
    }


def _create_visualizations(results: Dict[str, Any], output_path: Path) -> Dict[str, List[str]]:
    """Create visualizations for EDA results."""
    viz_dir = output_path / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    created_plots = {
        "entropy_plots": [],
        "distribution_plots": [],
        "temporal_plots": [],
        "adjacency_plots": []
    }

    try:
        # Set style for better looking plots
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Entropy analysis plots
        entropy_data = results["position_analysis"]["position_entropies"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Entropy by position
        positions = list(entropy_data.keys())
        entropies = list(entropy_data.values())

        ax1.bar(positions, entropies, alpha=0.7)
        ax1.set_title('Entropy by QS Position')
        ax1.set_xlabel('QS Position')
        ax1.set_ylabel('Entropy (bits)')
        ax1.tick_params(axis='x', rotation=45)

        # Normalized entropy comparison
        theoretical_max = results["position_analysis"]["theoretical_max_entropy"]
        normalized_entropies = [e / theoretical_max for e in entropies]

        ax2.bar(positions, normalized_entropies, alpha=0.7, color='orange')
        ax2.set_title('Normalized Entropy by Position')
        ax2.set_xlabel('QS Position')
        ax2.set_ylabel('Normalized Entropy (0-1)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Maximum Entropy')
        ax2.legend()

        plt.tight_layout()
        entropy_plot_path = viz_dir / "entropy_analysis.png"
        plt.savefig(entropy_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        created_plots["entropy_plots"].append(str(entropy_plot_path))

        # 2. Value distribution plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Global frequency distribution
        freq_dist = results["statistical_summaries"]["overall_statistics"]["value_frequency_distribution"]["frequency_distribution"]
        values = list(range(1, 40))
        frequencies = [freq_dist.get(str(v), 0) for v in values]

        ax1.bar(values, frequencies, alpha=0.7)
        ax1.set_title('Global Value Frequency Distribution')
        ax1.set_xlabel('QS Value')
        ax1.set_ylabel('Frequency')

        # Bottom 20 visualization
        bottom_20 = results["temporal_analysis"]["global_bottom_20"]
        bottom_20_freqs = [freq_dist.get(str(v), 0) for v in bottom_20]

        ax2.bar(range(len(bottom_20)), bottom_20_freqs, alpha=0.7, color='red')
        ax2.set_title('Bottom 20 Values Frequency')
        ax2.set_xlabel('Rank (Least Frequent)')
        ax2.set_ylabel('Frequency')
        ax2.set_xticks(range(0, len(bottom_20), 2))
        ax2.set_xticklabels([bottom_20[i] for i in range(0, len(bottom_20), 2)])

        # Position-wise distributions
        position_data = results["statistical_summaries"]["qs_position_statistics"]
        pos_means = [position_data[f"QS_{i}"]["mean"] for i in range(1, 6)]
        pos_stds = [position_data[f"QS_{i}"]["std"] for i in range(1, 6)]

        x_pos = range(1, 6)
        ax3.errorbar(x_pos, pos_means, yerr=pos_stds, marker='o', capsize=5)
        ax3.set_title('Mean ± Std by Position')
        ax3.set_xlabel('QS Position')
        ax3.set_ylabel('Value')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'QS_{i}' for i in range(1, 6)])

        # Adjacency distance distribution
        if "adjacency_patterns" in results:
            dist_data = results["adjacency_patterns"]["distance_distribution"]
            distances = []
            counts = []
            for key, count in dist_data.items():
                if key.startswith("distance_"):
                    distance = int(key.split("_")[1])
                    distances.append(distance)
                    counts.append(count)

            if distances and counts:
                ax4.bar(distances, counts, alpha=0.7, color='green')
                ax4.set_title('Cylindrical Distance Distribution')
                ax4.set_xlabel('Circular Distance')
                ax4.set_ylabel('Pair Count')

        plt.tight_layout()
        dist_plot_path = viz_dir / "distribution_analysis.png"
        plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        created_plots["distribution_plots"].append(str(dist_plot_path))

        # 3. Temporal analysis plots
        segments = results["temporal_analysis"]["segments"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Segment event counts
        segment_ids = [s["segment"] for s in segments]
        event_counts = [s["event_count"] for s in segments]

        ax1.bar(segment_ids, event_counts, alpha=0.7)
        ax1.set_title('Events per Temporal Segment')
        ax1.set_xlabel('Segment')
        ax1.set_ylabel('Event Count')

        # Bottom 20 comparison across segments
        bottom_20_overlap = {}
        global_bottom_20 = set(results["temporal_analysis"]["global_bottom_20"])

        for segment in segments:
            segment_bottom_20 = set(segment["bottom_20"])
            overlap = len(global_bottom_20.intersection(segment_bottom_20))
            bottom_20_overlap[segment["segment"]] = overlap

        ax2.bar(bottom_20_overlap.keys(), bottom_20_overlap.values(), alpha=0.7, color='purple')
        ax2.set_title('Bottom-20 Overlap with Global')
        ax2.set_xlabel('Segment')
        ax2.set_ylabel('Overlapping Values')
        ax2.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Perfect Overlap')
        ax2.legend()

        plt.tight_layout()
        temporal_plot_path = viz_dir / "temporal_analysis.png"
        plt.savefig(temporal_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        created_plots["temporal_plots"].append(str(temporal_plot_path))

        logger.info(f"Created {len(sum(created_plots.values(), []))} visualizations in {viz_dir}")

    except Exception as e:
        logger.warning(f"Failed to create some visualizations: {e}")
        # Continue without failing the entire analysis

    return created_plots


def _save_eda_artifacts(results: Dict[str, Any], output_path: Path) -> None:
    """Save EDA results to files."""
    # Save main summary with JSON serialization fix
    def json_serializer(obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path / "eda_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=json_serializer)

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


def main():
    """Enhanced CLI interface for C5Q EDA analysis."""
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="C5Q Exploratory Data Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m c5q.eda --csv data/c5_Matrix_binary.csv
  python -m c5q.eda --csv data/input.csv --out artifacts/eda/ --k 4 5 6
  python -m c5q.eda --csv data/input.csv --no-viz --verbose
        """
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to c5_Matrix_binary.csv file"
    )

    parser.add_argument(
        "--out",
        default="artifacts/eda",
        help="Output directory for EDA artifacts (default: artifacts/eda)"
    )

    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[4, 5, 6],
        help="K values for clustering analysis (default: 4 5 6)"
    )

    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip generating visualizations"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--validate-first",
        action="store_true",
        help="Run dataset validation before EDA"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick analysis mode (skip detailed statistics)"
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Validate inputs
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: Dataset file not found: {csv_path}")
        sys.exit(1)

    try:
        # Optional validation
        if args.validate_first:
            print("Running dataset validation first...")
            from .io import comprehensive_dataset_validation
            validation_results = comprehensive_dataset_validation(args.csv, save_checksum=False)
            if not validation_results["validation_passed"]:
                print(f"Warning: Dataset validation failed with {validation_results['error_summary']['total_errors']} errors")
                response = input("Continue with EDA anyway? (y/n): ")
                if response.lower() != 'y':
                    sys.exit(1)

        # Load dataset
        print(f"Loading dataset from {csv_path}...")
        from .io import load_primary_dataset
        df = load_primary_dataset(args.csv)
        print(f"Loaded {len(df)} events successfully")

        # Run EDA analysis
        print("Starting comprehensive EDA analysis...")
        results = run_comprehensive_eda(df, args.out, args.k)

        # Print summary
        print("\n" + "="*60)
        print("EDA ANALYSIS COMPLETE")
        print("="*60)
        print(f"Dataset: {results['dataset_info']['total_events']:,} events")
        print(f"Output directory: {args.out}")

        # Key insights
        print(f"\nKey Insights:")
        entropy_stats = results["position_analysis"]["entropy_statistics"]
        print(f"  • Average entropy across positions: {entropy_stats['mean_entropy']:.3f} bits")
        print(f"  • Entropy range: {entropy_stats['min_entropy']:.3f} - {entropy_stats['max_entropy']:.3f} bits")

        coverage = results["statistical_summaries"]["overall_statistics"]["coverage_percentage"]
        print(f"  • Value coverage: {coverage:.1f}% of possible values (1-39)")

        if "adjacency_patterns" in results:
            adj_rate = results["adjacency_patterns"]["adjacency_rate"]
            print(f"  • Cylindrical adjacency rate: {adj_rate:.3f}")

        # Bottom 20 preview
        bottom_20 = results["temporal_analysis"]["global_bottom_20"]
        print(f"  • Global bottom-20: {bottom_20[:10]}... (see artifacts for full list)")

        # Artifacts created
        print(f"\nArtifacts created:")
        print(f"  • Main summary: {args.out}/eda_summary.json")
        print(f"  • Bottom-20 CSVs: {args.out}/global_bottom20.csv, seg*.csv")
        print(f"  • Clustering results: {args.out}/buckets_k*.csv")

        if not args.no_viz and "visualizations" in results:
            viz_count = len(sum(results["visualizations"].values(), []))
            print(f"  • Visualizations: {viz_count} plots in {args.out}/visualizations/")

        print("="*60)

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during EDA analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()