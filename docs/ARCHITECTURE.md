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
│                    │  VRAM → RAM → Disk             │      │
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
PE(pos, 2i) = sin(pos / 10000^(2i/dim))
PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
```

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

**Three-Tier Memory:**

| Tier | Storage | Access Time | Capacity |
|------|---------|-------------|----------|
| Hot | VRAM | <10μs | ~100K |
| Warm | RAM | <1ms | ~1M |
| Cold | Disk | <10ms | Unlimited |

**Eviction Policy:**
- LRU (Least Recently Used)
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
@njit(fastmath=True, cache=True)
def compute_distances(data, centroids):
    # Parallel loop
    for i in prange(N):
        # Fast math operations
        ...
```

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
2. **Memory Bound**: Embeddings must fit in RAM
3. **CPU Only**: No GPU acceleration (yet)
4. **Single Machine**: No distributed processing

## Future Improvements

1. **HNSW Integration**: Replace fine clustering with HNSW graphs
2. **Product Quantization**: Reduce memory for embeddings
3. **GPU Support**: CUDA kernels for encoding and clustering
4. **Distributed**: Shard across multiple machines

## References

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Kerbl et al., 2023
- [Numba](https://numba.pydata.org/) - JIT compilation
- [IVF-Flat](https://github.com/facebookresearch/faiss) - Inverted file index
