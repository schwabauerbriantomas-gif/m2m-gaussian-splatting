#!/usr/bin/env python
"""
Run Benchmarks - M2M Gaussian Splatting

Compares HRM2 hierarchical search against brute-force linear search.
"""

import sys
import os
import json
import platform
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from time import time
from typing import List, Tuple

from m2m_gaussian_splatting.core.splat_types import GaussianSplat
from m2m_gaussian_splatting.core.hrm2_engine import HRM2Engine, generate_test_splats


def linear_search(
    splats: List[GaussianSplat], embeddings: np.ndarray, query: np.ndarray, k: int = 10
) -> List[Tuple[GaussianSplat, float]]:
    """Brute-force linear search."""
    distances = np.linalg.norm(embeddings - query, axis=1)
    indices = np.argsort(distances)[:k]
    return [(splats[i], float(distances[i])) for i in indices]


def run_benchmark(n_splats: int, n_queries: int = 50) -> dict:
    """
    Run benchmark for a given dataset size.

    Args:
        n_splats: Number of splats
        n_queries: Number of queries to average

    Returns:
        Dictionary with benchmark results
    """
    # Generate data
    splats = generate_test_splats(n_splats, seed=42)

    # Build embeddings
    positions = np.array([s.position for s in splats])
    colors = np.array([s.color for s in splats])
    opacities = np.array([s.opacity for s in splats])
    scales = np.array([s.scale for s in splats])
    rotations = np.array([s.rotation for s in splats])

    from m2m_gaussian_splatting.core.encoding import FullEmbeddingBuilder

    encoder = FullEmbeddingBuilder()
    embeddings = encoder.build(positions, colors, opacities, scales, rotations)

    # Build HRM2 index
    n_coarse = max(10, n_splats // 500)
    n_fine = max(50, n_splats // 50)

    engine = HRM2Engine(n_coarse=n_coarse, n_fine=n_fine, n_probe=3)
    engine.add_splats(splats)
    build_time = engine.index()

    # Benchmark queries
    linear_times = []
    hrm2_times = []
    recalls = []

    np_random = np.random.RandomState(123)
    query_indices = np_random.choice(n_splats, n_queries, replace=False)

    for idx in query_indices:
        query = embeddings[idx]

        # Linear search
        start = time()
        linear_results = linear_search(splats, embeddings, query, k=10)
        linear_times.append(time() - start)

        # HRM2 search
        start = time()
        hrm2_results = engine.query(query, k=10)
        hrm2_times.append(time() - start)

        # Compute recall
        linear_ids = set(s.id for s, _ in linear_results)
        hrm2_ids = set(s.id for s, _ in hrm2_results)
        recall = len(linear_ids & hrm2_ids) / 10
        recalls.append(recall)

    return {
        "n_splats": n_splats,
        "build_time": build_time,
        "linear_avg_ms": np.mean(linear_times) * 1000,
        "linear_std_ms": np.std(linear_times) * 1000,
        "hrm2_avg_ms": np.mean(hrm2_times) * 1000,
        "hrm2_std_ms": np.std(hrm2_times) * 1000,
        "speedup": np.mean(linear_times) / max(np.mean(hrm2_times), 1e-9),
        "recall": np.mean(recalls),
    }


def main():
    print("=" * 70)
    print("M2M GAUSSIAN SPLATTING - BENCHMARKS")
    print("=" * 70)
    print()

    DATASET_SIZES = [1000, 5000, 10000, 25000, 50000]
    N_QUERIES = 50

    print(f"Configuration:")
    print(f"  Queries per dataset: {N_QUERIES}")
    print(f"  K (results): 10")
    print()

    print("-" * 70)
    print(
        f"{'Splats':>10} | {'Build (s)':>10} | {'Linear (ms)':>12} | {'HRM2 (ms)':>10} | {'Speedup':>8} | {'Recall':>6}"
    )
    print("-" * 70)

    results = []

    for n in DATASET_SIZES:
        print(f"Running benchmark with {n:,} splats...", end="\r")
        result = run_benchmark(n, N_QUERIES)
        results.append(result)

        print(
            f"{result['n_splats']:>10,} | "
            f"{result['build_time']:>10.2f} | "
            f"{result['linear_avg_ms']:>12.2f} | "
            f"{result['hrm2_avg_ms']:>10.2f} | "
            f"{result['speedup']:>8.1f}x | "
            f"{result['recall']:>6.2%}"
        )

    print("-" * 70)
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    print("Key observations (computed from the measured results above):")
    print()

    ns = np.array([r["n_splats"] for r in results])
    builds = np.array([r["build_time"] for r in results])
    hrm2_ms = np.array([r["hrm2_avg_ms"] for r in results])
    recalls_arr = np.array([r["recall"] for r in results])

    # Build scaling: slope of log-log fit (≈1 linear, ≈2 quadratic)
    if len(results) >= 2:
        slope = np.polyfit(np.log(ns), np.log(builds), 1)[0]
        scaling = (
            "linear"
            if slope < 1.3
            else "quadratic" if slope > 1.7 else f"superlinear (exponent {slope:.2f})"
        )
        print(f"  - Build time scales {scaling} (log-log slope {slope:.2f})")

        q_growth = hrm2_ms[-1] / hrm2_ms[0]
        print(
            f"  - Query time grew {q_growth:.1f}x from {ns[0]:,} to {ns[-1]:,} splats "
            f"({hrm2_ms[0]:.2f}ms → {hrm2_ms[-1]:.2f}ms)"
        )

        sp = np.array([r["speedup"] for r in results])
        trend = "increases" if sp[-1] > sp[0] else "decreases" if sp[-1] < sp[0] else "is flat"
        print(f"  - Speedup {trend} with dataset size ({sp[0]:.1f}x → {sp[-1]:.1f}x)")

    min_recall = recalls_arr.min()
    if min_recall > 0.9:
        print(f"  - Recall stays above 90% at all sizes (min {min_recall:.2%})")
    else:
        print(f"  - WARNING: recall dropped below 90% (min {min_recall:.2%})")
    print()

    max_speedup = max(r["speedup"] for r in results)
    avg_recall = np.mean([r["recall"] for r in results])

    print(f"Maximum speedup: {max_speedup:.1f}x")
    print(f"Average recall: {avg_recall:.2%}")
    print()

    # Persist results with reproducibility metadata
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "config": {"n_queries": N_QUERIES, "k": 10, "dataset_sizes": DATASET_SIZES},
        "results": results,
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "benchmark_results_cpu.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"Results saved to {os.path.normpath(out_path)}")
    print()

    print("=" * 70)
    print("BENCHMARKS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
