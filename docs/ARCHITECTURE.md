# M2M Gaussian Splatting - Architecture

This document describes the technical architecture of M2M Gaussian Splatting.

## Overview

M2M provides hierarchical retrieval and memory management for large-scale 3D Gaussian splat datasets.

```
┌─────────────────────────────────────────────────────────────┐
│                     M2M Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Splats     │───▶│  Encodings   │───▶│    HRM2      │  │
│  │  (640 bytes) │    │   (640D)     │    │    Index     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                 │           │
│                                                 ▼           │
│                    ┌────────────────────────────────┐      │
│                    │      Memory Manager            │      │
│                    │    Hot → Warm → Cold (host)    │      │
│                    └────────────────────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Splat Types (`splat_types.py`)

**GaussianSplat:**
- Position (3D): x, y, z coordinates
- Color (RGB): r, g, b values
- Opacity: transparency value
- Scale (3D): sx, sy, sz scaling factors
- Rotation (4D): quaternion w, x, y, z

**SplatEmbedding:**
- Position encoding (64D): Sinusoidal encoding
- Color encoding (512D): Histogram representation
- Attribute encoding (64D): Opacity/scale features

### 2. Encoding (`encoding.py`)

**Position Encoding (64D):**
```
PE(pos, 2i)   = sin(pi · 2^i · pos)     i = 0..n_freq-1 (per axis)
PE(pos, 2i+1) = cos(pi · 2^i · pos)
```
The `pi` factor (NeRF-style) makes the lowest band a true half-wave over
[0, 1]; without it the low bands are near-linear and waste dimensions.
Positions are normalized to [0, 1] with scene bounds fixed once via
`encoder.fit(positions)` (or an explicit `bounds` argument) so that
indexing and querying agree — batch-wise normalization is the legacy
default and is NOT recommended for vector search.
Dimensions beyond 3 axes × 2·n_freq bands (e.g. 60-63 at dim=64) carry
the top frequency band on the remaining axes instead of staying zero.

**Color Encoding (512D):**
- 8 bins per channel
- Gaussian-smoothed histogram
- 8³ = 512 dimensions

**Attribute Encoding (64D):**
- Opacity features: raw, squared, cubic, sqrt, log
- Scale features: raw, ratios, logs
- Rotation features: quaternion components, products

### 3. Clustering (`clustering.py`)

**KMeansJIT:**
- K-Means++ initialization
- Numba JIT compilation
- Parallel distance computation

**Algorithm:**
```
1. Initialize centroids with K-Means++
2. Repeat until convergence:
   a. Assign points to nearest centroid
   b. Update centroids as cluster means
   c. Check for shift < tolerance
```

### 4. HRM2 Engine (`hrm2_engine.py`)

**Hierarchical Retrieval Model 2:**

```
Level 1 (Coarse):
- K-Means with K = n_coarse clusters
- Fast pruning of search space

Level 2 (Fine):
- K-Means within each coarse cluster
- Precise local search

Query Process:
1. Find n_probe nearest coarse clusters
2. Search within selected clusters
3. Return top-k results
```

### 5. Memory Manager (`manager.py`)

**Three-Tier Memory (all host-side in v2.1.x):**

| Tier | Storage | Access Time | Capacity |
|------|---------|-------------|----------|
| Hot | host dict (hottest) | <1μs | ~100K |
| Warm | host dict | <1ms | ~1M |
| Cold | host dict (coldest) | <1ms | RAM-bound |

Note: tiers are logical (dicts in host RAM); they are NOT CUDA/disk tiers.
The guarantee that matters: eviction never loses data — splats evicted
from the hot tier move to cold, and every splat lives in exactly one tier.

**Eviction Policy:**
- LRU (Least Recently Used) with second chance: victims at ≥70% of
  `access_threshold` are promoted instead of evicted
- Promotion based on access count

## Performance

### Time Complexity

| Operation | Complexity |
|-----------|------------|
| Build Index | O(N · I · K · D) |
| Query | O(n_probe · cluster_size · D) |

Where:
- N = number of splats
- I = K-Means iterations
- K = number of clusters
- D = embedding dimension (640)
- n_probe = clusters searched

### Space Complexity

| Component | Size |
|-----------|------|
| Splats | O(N · 56) bytes |
| Embeddings | O(N · 640 · 4) bytes |
| Coarse centroids | O(K_coarse · 640 · 4) bytes |
| Fine centroids | O(K_fine · 640 · 4) bytes |

## Optimization Techniques

### 1. Numba JIT

```python
@njit(fastmath=True, cache=True, parallel=True)
def compute_distances(data, centroids):
    # Parallel loop
    for i in prange(N):
        # Fast math operations
        ...
```

Note: `parallel=True` is required for `prange` to actually run in
parallel (it silently runs serial otherwise). Distance computations on
the assignment step use BLAS GEMM via the Gram trick
(`‖x‖² − 2x·c + ‖c‖²`) rather than scalar loops.

### 2. Dynamic Clustering

```python
n_fine = min(default_n_fine, cluster_size // 5)
```

### 3. Memory Hierarchy

- Frequently accessed: promoted to VRAM
- Recently accessed: kept in RAM
- Infrequently accessed: stored on disk

## Limitations

1. **Approximate Search**: Not guaranteed to find exact nearest neighbors
2. **Memory Bound**: Embeddings must fit in host RAM
3. **CPU-First**: GPU acceleration exists (`TorchKMeans`, `GPUSearcher`
   via torch) but CUDA paths are not re-benchmarked as of v2.1.x
4. **Single Machine**: No distributed processing
5. **Memory Tiers Are Logical**: hot/warm/cold are host-side dicts, not
   CUDA/disk tiers
6. **Incremental Inserts Are Approximate**: `add_splats_incremental()`
   does not re-cluster; use `index()` for canonical geometry

## Future Improvements

1. **Real GPU/Disk Tiers**: move the memory hierarchy to CUDA/disk storage
2. **Product Quantization**: reduce memory for embeddings (not worth it
   below ~1M vectors per external benchmarking)
3. **CUDA Re-Benchmark**: measure GPU build/query and fp16 index on real
   hardware
4. **Incremental Re-Clustering Policy**: periodic rebalance of clusters
   after incremental inserts
5. **Distributed**: Shard across multiple machines

## References

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Kerbl et al., 2023
- [Numba](https://numba.pydata.org/) - JIT compilation
- [IVF-Flat](https://github.com/facebookresearch/faiss) - Inverted file index
