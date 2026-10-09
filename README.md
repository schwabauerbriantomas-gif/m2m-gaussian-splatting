# M2M Gaussian Splatting

[![CI](https://github.com/schwabauerbriantomas-gif/m2m-gaussian-splatting/actions/workflows/ci.yml/badge.svg)](https://github.com/schwabauerbriantomas-gif/m2m-gaussian-splatting/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/m2m-gaussian-splatting.svg)](https://pypi.org/project/m2m-gaussian-splatting/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Hierarchical retrieval and memory management for 3D Gaussian Splatting, with GPU acceleration (CUDA/PyTorch) and CPU fallback (Numba JIT).**

---

## Overview

M2M Gaussian Splatting converts 3D Gaussian splats into 640-dimensional embeddings and indexes them using a two-level hierarchical clustering (HRM2) for fast approximate nearest neighbor search. It automatically uses CUDA when available and falls back to CPU.

**Key components:**

- **640D embeddings** — position (64D sinusoidal), color (512D histogram), attributes (64D)
- **HRM2 Engine** — two-level K-Means index (coarse + fine) with IVF-style candidate pruning
- **GPU search** — brute-force L2 k-NN on CUDA via PyTorch `torch.topk`, auto-detected
- **Memory manager** — three-tier LRU cache (VRAM → RAM → cold) with thread-safe eviction

---

## Installation

```bash
pip install m2m-gaussian-splatting
```

**With GPU support (optional):**

```bash
pip install m2m-gaussian-splatting[gpu]
```

This installs PyTorch with CUDA. The library works on CPU alone (`numpy` + `numba`); GPU is optional and auto-detected at runtime.

**From source:**

```bash
git clone https://github.com/schwabauerbriantomas-gif/m2m-gaussian-splatting.git
cd m2m-gaussian-splatting
pip install -e ".[dev]"
```

---

## Quick Start

```python
from m2m_gaussian_splatting import HRM2Engine, generate_test_splats

# Generate synthetic test data
splats = generate_test_splats(n_splats=10000)

# Build index (auto-detects CUDA, falls back to CPU)
engine = HRM2Engine(n_coarse=50, n_fine=200, use_gpu=True)
engine.add_splats(splats)
engine.index()

# Query for similar splats
query = engine.embeddings[0]
results = engine.query(query, k=10)

for splat, distance in results:
    print(f"Splat {splat.id}: distance={distance:.4f}")

print(f"Device: {engine.get_stats().device}")  # "cuda" or "cpu"
```

---

## Performance

### CPU (hierarchical IVF path) — v2.1

**Benchmark environment:** AMD Ryzen 5 3400G (8 threads, CPU-only), Python 3.12.3, NumPy 2.5.3, Numba 0.68. 640D embeddings, k=10, 50 queries, mean latency. Measured with `m2m-benchmark` (`m2m_gaussian_splatting/scripts/run_benchmarks.py`); raw JSON with full metadata: `benchmark_results_cpu.json`.

| Splats | HRM2 query | vs linear | Build |
|--------|-----------|-----------|-------|
| 1,000 | 0.54 ms | 2.7x | 0.22 s |
| 10,000 | 2.30 ms | 9.0x | 2.00 s |
| 50,000 | 7.01 ms | 13.5x | 4.25 s |

Recall@10 = 100% at every size. Compared to v2.0.0 on the same machine and seeds: build **3.8x-35.4x faster** (50k: 150.5s → 4.25s), query **2.4x-6.7x faster** (50k: 47.0ms → 7.0ms) — lazy fine clustering, BLAS-GEMM cluster assignment, parallel numba kernels, CSR query layout.

### GPU (brute-force path) — v2.0.0 measurements

**Environment:** NVIDIA RTX 3090 (24GB), Python 3.12, PyTorch 2.6 + CUDA 12.4. Not re-measured in v2.1 (no CUDA in the audit environment); the GPU K-Means update loop was vectorized since then, so GPU build times below are expected to improve.

| Splats | CPU p50 | GPU p50 | Speedup | CPU Build | GPU Build | GPU QPS |
|--------|---------|---------|---------|-----------|-----------|---------|
| 10,000 | 8.22 ms | 1.12 ms | **7.3x** | 28.0 s | 2.1 s | 831 |
| 50,000 | 49.76 ms | 3.64 ms | **13.7x** | 156.7 s | 7.4 s | 236 |

GPU brute-force search dominates at scale: single-tensor upload, batched `torch.topk`, no clustering overhead.

---

## Architecture

### Embedding Pipeline (640D)

```
┌──────────────────────────────────────────────────────────┐
│                    640D Embedding                         │
├────────────────┬──────────────────┬───────────────────────┤
│ Position (64D) │  Color (512D)    │  Attributes (64D)     │
│ Sinusoidal PE  │  Histogram +     │  Opacity, Scale,      │
│ (NeRF-style)   │  Gaussian smooth │  Rotation features    │
└────────────────┴──────────────────┴───────────────────────┘
```

### HRM2 Two-Level Index (CPU path)

```
Query Vector (640D)
    │
    ▼
┌──────────────────────┐
│ Level 1: Coarse      │  K-Means (n_coarse clusters)
│ Find n_probe nearest │  Gram-matrix L2: ||q||² + ||c||² - 2·q·c
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│ Level 2: Candidates  │  Gather splats from probed clusters
│ Rank by L2 distance  │  argpartition top-k (O(N) not O(N log N))
└──────────────────────┘
    │
    ▼
  Top-K Results
```

### GPU Path

```
Query Batch (B × 640D)
    │
    ▼
┌──────────────────────────────┐
│ GPU Brute-Force L2           │  torch.matmul: B×N distance matrix
│ Gram trick on CUDA tensors   │  ||q||² + ||m||² - 2·q·mᵀ
└──────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│ torch.topk                   │  Batched top-k selection on GPU
│ Sorted distances + indices   │  Chunked by max_batch_size
└──────────────────────────────┘
    │
    ▼
  Top-K Results per query
```

---

## Project Structure

```
m2m-gaussian-splatting/
├── m2m_gaussian_splatting/
│   ├── core/
│   │   ├── splat_types.py       # GaussianSplat, SplatEmbedding dataclasses
│   │   ├── encoding.py          # Numba JIT encoders (position, color, attribute)
│   │   ├── clustering.py        # K-Means++ + Mini-batch (BLAS GEMM + Numba)
│   │   └── hrm2_engine.py       # HRM2 hierarchical index + GPU dispatch
│   ├── gpu/
│   │   ├── backend.py           # CUDA detection, device info
│   │   ├── gpu_kmeans.py        # GPU K-Means via PyTorch tensors
│   │   └── gpu_search.py        # GPU brute-force k-NN search
│   ├── memory/
│   │   └── manager.py           # Three-tier LRU memory (thread-safe)
│   └── scripts/
│       ├── quick_demo.py        # Interactive demo (entry point: m2m-demo)
│       ├── run_benchmarks.py    # CPU HRM2 vs linear (entry point: m2m-benchmark)
│       └── benchmark_gpu.py     # GPU vs CPU comparison + recall check
├── tests/
│   ├── test_core.py             # 70 tests (CPU + GPU + v2.1/v2.1.1 regression)
│   ├── test_persistence.py      # save/load round-trips (4 tests)
│   └── test_edges.py            # edge-case contracts (8 tests)
├── docs/
│   └── ARCHITECTURE.md          # Architecture document
├── benchmark_results.json       # Measured results
├── pyproject.toml
└── README.md
```

---

## API Reference

### HRM2Engine

```python
engine = HRM2Engine(
    n_coarse=100,     # Coarse clusters
    n_fine=1000,      # Fine clusters per coarse
    n_probe=5,        # Clusters to probe at query time
    use_gpu=True,     # Auto-detect CUDA (default), set False to force CPU
)

engine.add_splats(splats)
engine.index()

# Single query → List[(GaussianSplat, distance)]
results = engine.query(query_vector, k=10)

# Batch query (GPU-parallelized when available)
results = engine.batch_query(query_batch, k=10)

# Detailed results with cluster IDs
details = engine.query_with_details(query_vector, k=10)  # → List[SearchResult]
```

### GaussianSplat

```python
splat = GaussianSplat(
    id=0,
    position=[x, y, z],        # 3D position
    color=[r, g, b],           # RGB color [0,1] or [0,255]
    opacity=0.9,               # Transparency [0,1]
    scale=[sx, sy, sz],        # Ellipsoid scale factors
    rotation=[w, x, y, z],     # Quaternion (auto-normalized)
)
```

### SplatMemoryManager

```python
memory = SplatMemoryManager(
    vram_limit=100000,         # Hot tier (max splats)
    ram_limit=1000000,         # Warm tier
    eviction_threshold=0.8,    # Evict at 80% capacity
    access_threshold=10,       # Promote to hot tier after N accesses
)

memory.add_splats(splats)
splat = memory.get_splat(splat_id)  # Thread-safe, auto-promotes
```

Note: tiers are recency tiers over in-process dicts — there is no CUDA or disk backing. Eviction never loses data (warm evictions return to cold storage) and tiers are exclusive (each splat lives in exactly one tier).

---

## Use Cases

- **3D scene retrieval** — find similar regions in large-scale 3D scans or reconstructions
- **Asset search** — query through millions of 3D Gaussian splats by appearance similarity
- **Point cloud processing** — efficient k-NN on LiDAR and photogrammetry data
- **Memory-tiered caching** — hot/warm/cold splat storage with LRU eviction for VRAM-constrained rendering

---

## Testing

```bash
# Run full test suite (70 tests)
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=m2m_gaussian_splatting

# Run benchmarks (after `pip install -e .`)
m2m-benchmark                      # HRM2 vs linear
m2m-demo --n-splats 300            # Quick demo with a small dataset
python -m m2m_gaussian_splatting.scripts.benchmark_gpu    # GPU vs CPU
```

Tests cover: GPU recall against CPU ground truth, K-Means convergence, encoding edge cases (single point, empty input, NaN rejection, color spaces), thread safety (8-thread concurrent access), LRU eviction with no data loss, index save/load round-trip, float64 norm overflow, and global RNG isolation.

---

## Technical Details

### Position Encoding (64D)

NeRF-style multi-frequency sinusoidal encoding applied per-axis (x, y, z):

```
PE(axis, freq_i) = [sin(π · 2^i · axis_norm), cos(π · 2^i · axis_norm)]
```

where `axis_norm` normalizes each coordinate to [0, 1] using scene bounds. Output is `n_freq × 6` columns (sin/cos for x, y, z per frequency), zero-padded to exactly 64D. Pass explicit `bounds` (or call `fit()`) for batch-independent, query-consistent encodings — by default normalization uses each batch's min/max, which makes single-point queries inconsistent with a batch-indexed base (legacy behavior kept for compatibility).

### Color Encoding (512D)

Histogram-based with Gaussian kernel smoothing:

- 8 bins per RGB channel → 8³ = 512 total dimensions
- Only evaluates bins within radius 2 of the target (125 vs 512 iterations per splat)
- Gaussian kernel `exp(-d²/4)` decays below 0.37 at d=2
- Output rows are L2-normalized (corner colors otherwise get a smaller norm and bias L2 search)
- `color_space='01' | '255'` overrides the legacy batch-max auto-detection, which misclassifies dark [0,255] scenes

### K-Means

- **K-Means++ initialization** with running min-distance array (O(N·K·D), not O(N·K²·D))
- **Assignment via BLAS GEMM** (Gram trick `‖x‖² - 2x·c + ‖c‖²`) instead of scalar loops
- **Mini-batch updates** with learning-rate decay `η = 1/count`, `tol` early stopping, and dead-cluster reseeding
- Numba `@njit(parallel=True)` kernels on CPU, PyTorch CUDA (fully vectorized, no per-cluster syncs) on GPU
- NaN/Inf input rejected explicitly (previously collapsed the clustering silently)
- OOM recovery: GPU init frees the cache and falls back to CPU automatically

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

---

## Author

Brian Schwabauer — schwabauerbriantomas@gmail.com

Project: https://github.com/schwabauerbriantomas-gif/m2m-gaussian-splatting

---

## Keywords

`gaussian-splatting` `3dgs` `hierarchical-retrieval` `vector-search` `numba` `kmeans` `similarity-search` `point-cloud` `embedding` `cpu-optimized` `gpu` `cuda` `pytorch` `memory-management` `hrm2` `clustering` `jit-compilation`
