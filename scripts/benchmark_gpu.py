#!/usr/bin/env python
"""Benchmark GPU vs CPU: m2m-gaussian-splatting v2.0.0"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from time import time
from src.core.hrm2_engine import HRM2Engine, generate_test_splats
from src.gpu import HAS_CUDA, get_gpu_info

print(f"GPU available: {HAS_CUDA}")
if HAS_CUDA:
    info = get_gpu_info()
    print(f"  Device: {info.get('name', 'N/A')}")
    print(f"  VRAM:   {info.get('vram_total_gb', 0):.1f} GB")
print()


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
    for idx in q_indices:
        q = engine.embeddings[idx]
        t0 = time()
        results = engine.query(q, k=10)
        times.append(time() - t0)

    return {
        "n": n_splats,
        "device": device,
        "build_s": round(build_time, 3),
        "p50_ms": round(float(np.percentile(times, 50) * 1000), 3),
        "p95_ms": round(float(np.percentile(times, 95) * 1000), 3),
        "qps": round(len(times) / sum(times), 1),
    }


SIZES = [1000, 10000, 50000]

print(
    f"{'N':>10} | {'Device':>6} | {'Build(s)':>8} | {'p50(ms)':>8} | {'p95(ms)':>8} | {'QPS':>10}"
)
print("-" * 70)

results = []
for n in SIZES:
    r_cpu = run_bench(n, use_gpu=False)
    print(
        f"{r_cpu['n']:>10,} | {r_cpu['device']:>6} | {r_cpu['build_s']:>8.2f} | {r_cpu['p50_ms']:>8.2f} | {r_cpu['p95_ms']:>8.2f} | {r_cpu['qps']:>10.1f}"
    )
    results.append(r_cpu)

    if HAS_CUDA:
        r_gpu = run_bench(n, use_gpu=True)
        speedup = r_cpu["p50_ms"] / r_gpu["p50_ms"] if r_gpu["p50_ms"] > 0 else 0
        print(
            f"{r_gpu['n']:>10,} | {r_gpu['device']:>6} | {r_gpu['build_s']:>8.2f} | {r_gpu['p50_ms']:>8.2f} | {r_gpu['p95_ms']:>8.2f} | {r_gpu['qps']:>10.1f}  ({speedup:.1f}x)"
        )
        results.append(r_gpu)
    print()

import json

with open(os.path.join(os.path.dirname(__file__), "..", "benchmark_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to benchmark_results.json")
