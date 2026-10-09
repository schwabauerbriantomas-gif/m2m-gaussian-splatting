#!/usr/bin/env python
"""
Benchmark GPU vs CPU: m2m-gaussian-splatting

Runs the HRM2 engine on CPU (and GPU when CUDA is available) and verifies
query recall against a NumPy brute-force ground truth.

Note: the recall check intentionally uses only NumPy on the same data —
no torch — so it validates the GPU path against an independent CPU reference.
"""

import json
import os
from time import time

import numpy as np

from m2m_gaussian_splatting.core.hrm2_engine import HRM2Engine, generate_test_splats
from m2m_gaussian_splatting.gpu import HAS_CUDA, get_gpu_info

SIZES = [1000, 10000, 50000]


def numpy_topk(distances: np.ndarray, k: int) -> np.ndarray:
    """CPU ground truth: indices of the k smallest distances (NumPy only)."""
    return np.argsort(distances)[:k]


def compute_recall(
    embeddings: np.ndarray, query: np.ndarray, result_ids: list, k: int = 10
) -> float:
    """
    Recall@k of `result_ids` vs NumPy brute-force top-k on the same embeddings.

    Pure NumPy — deliberately independent of torch/GPU code paths.
    """
    distances = np.linalg.norm(embeddings - query, axis=1)
    gt_ids = numpy_topk(distances, k)
    return len(set(gt_ids) & set(result_ids)) / k


def run_bench(n_splats, n_queries=50, use_gpu=False):
    splats = generate_test_splats(n_splats, seed=42)
    n_coarse = max(10, n_splats // 500)
    n_fine = max(50, n_splats // 50)

    engine = HRM2Engine(n_coarse=n_coarse, n_fine=n_fine, n_probe=5, use_gpu=use_gpu)
    engine.add_splats(splats)
    build_time = engine.index()
    device = engine.get_stats().device

    np.random.seed(123)
    q_indices = np.random.choice(n_splats, n_queries, replace=False)

    # Warmup
    for i in range(3):
        engine.query(engine.embeddings[q_indices[0]], k=10)

    # Benchmark
    times = []
    recalls = []
    for idx in q_indices:
        q = engine.embeddings[idx]
        t0 = time()
        results = engine.query(q, k=10)
        times.append(time() - t0)

        # Recall vs NumPy CPU ground truth (first 5 queries only)
        if len(recalls) < 5:
            result_ids = [s.id for s, _ in results]
            recalls.append(compute_recall(engine.embeddings, q, result_ids, k=10))

    recall = float(np.mean(recalls)) if recalls else 1.0
    if recall < 1.0:
        print(
            f"  WARNING: recall vs NumPy argsort ground truth is {recall:.2f} (<1.0) "
            f"for n={n_splats} on {device}"
        )

    return {
        "n": n_splats,
        "device": device,
        "build_s": round(build_time, 3),
        "p50_ms": round(float(np.percentile(times, 50) * 1000), 3),
        "p95_ms": round(float(np.percentile(times, 95) * 1000), 3),
        "qps": round(len(times) / sum(times), 1),
        "recall": round(recall, 3),
    }


def print_gpu_status():
    print(f"GPU available: {HAS_CUDA}")
    if HAS_CUDA:
        info = get_gpu_info()
        print(f"  Device: {info.get('name', 'N/A')}")
        print(f"  VRAM:   {info.get('vram_total_gb', 0):.1f} GB")
    print()


def main():
    print_gpu_status()

    print(
        f"{'N':>10} | {'Device':>6} | {'Build(s)':>8} | {'p50(ms)':>8} | {'p95(ms)':>8} | "
        f"{'QPS':>10} | {'Recall':>6}"
    )
    print("-" * 80)

    results = []
    for n in SIZES:
        r_cpu = run_bench(n, use_gpu=False)
        print(
            f"{r_cpu['n']:>10,} | {r_cpu['device']:>6} | {r_cpu['build_s']:>8.2f} | "
            f"{r_cpu['p50_ms']:>8.2f} | {r_cpu['p95_ms']:>8.2f} | {r_cpu['qps']:>10.1f} | "
            f"{r_cpu['recall']:>6.2f}"
        )
        results.append(r_cpu)

        if HAS_CUDA:
            r_gpu = run_bench(n, use_gpu=True)
            speedup = r_cpu["p50_ms"] / r_gpu["p50_ms"] if r_gpu["p50_ms"] > 0 else 0
            print(
                f"{r_gpu['n']:>10,} | {r_gpu['device']:>6} | {r_gpu['build_s']:>8.2f} | "
                f"{r_gpu['p50_ms']:>8.2f} | {r_gpu['p95_ms']:>8.2f} | {r_gpu['qps']:>10.1f} | "
                f"{r_gpu['recall']:>6.2f}  ({speedup:.1f}x)"
            )
            results.append(r_gpu)
        print()

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "benchmark_results.json",
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
