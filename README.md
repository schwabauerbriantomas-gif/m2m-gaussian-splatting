# M2M Gaussian Splatting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Hierarchical Retrieval and Memory Management for 3D Gaussian Splatting**

M2M Gaussian Splatting provides fast similarity search and efficient memory management for large-scale 3D Gaussian splat datasets. Optimized for CPU with Numba JIT compilation.

---

## ✨ Features

- **Hierarchical Indexing (HRM2)** - Two-level clustering for fast retrieval
- **640D Embeddings** - Position, color, and attribute encodings
- **Numba JIT Acceleration** - 5-10x speedup over pure Python
- **Memory Management** - Three-tier memory (VRAM/RAM/Disk)
- **High Recall** - >90% recall at 50x+ speedup
- **CPU Optimized** - No GPU required

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/schwabauerbriantomas-gif/m2m-gaussian-splatting.git
cd m2m-gaussian-splatting

# Install dependencies
pip install -r requirements.txt

# Or with pip
pip install numpy numba scipy scikit-learn
```


## 🚀 Quick Start

```python
from src.core import GaussianSplat, HRM2Engine, generate_test_splats

# Generate test data
splats = generate_test_splats(n_splats=10000)

# Create and index
engine = HRM2Engine(n_coarse=50, n_fine=200)
engine.add_splats(splats)
engine.index()

# Query for similar splats
query = splats[0]  # Use first splat as query
results = engine.query(query.vector, k=10)

for splat, distance in results:
    print(f"Splat {splat.id}: distance={distance:.4f}")
```

---

## 📊 Performance

Benchmarks on ryzen 5 3400g (640D embeddings, k=10):

| Splats | Build (s) | Linear (ms) | HRM2 (ms) | Speedup | Recall |
|--------|-----------|-------------|-----------|---------|--------|
| 10,000 | 1.2 | 12.5 | 0.8 | 15x | 95% |
| 50,000 | 5.8 | 62.0 | 1.5 | 41x | 93% |
| 100,000 | 12.0 | 125.0 | 2.2 | 57x | 91% |

---

## 🔧 Architecture

### Embedding (640D)

```
┌────────────────────────────────────────────────────────┐
│                    640D Embedding                      │
├────────────────────────────────────────────────────────┤
│  Position (64D)  │  Color (512D)  │  Attributes (64D)  │
│  Sinusoidal PE   │  Histogram     │  Opacity/Scale     │
└────────────────────────────────────────────────────────┘
```

### HRM2 Index

```
Query Vector
     │
     ▼
┌──────────────────┐
│ Level 1: Coarse  │  ← K-Means (100 clusters)
│ Find nearest     │
└──────────────────┘
     │
     ▼
┌──────────────────┐
│ Level 2: Fine    │  ← K-Means (1000 clusters)
│ Search within    │
└──────────────────┘
     │
     ▼
  Top-K Results
```

---

## 📁 Project Structure

```
m2m-gaussian-splatting/
├── src/
│   ├── core/
│   │   ├── splat_types.py     # Data structures
│   │   ├── encoding.py        # Numba JIT encoders
│   │   ├── clustering.py      # K-Means implementation
│   │   └── hrm2_engine.py     # Hierarchical retrieval
│   └── memory/
│       └── manager.py         # Memory tiers
├── scripts/
│   ├── quick_demo.py          # Getting started
│   └── run_benchmarks.py      # Performance tests
├── tests/
│   └── test_*.py
├── docs/
│   └── ARCHITECTURE.md
├── requirements.txt
└── README.md
```

---

## 🎯 Use Cases

### 3D Scene Retrieval
Find similar regions in large 3D scans.

```python
# Find similar scene regions
results = engine.query(region_embedding, k=20)
```


### Asset Search
Search through millions of 3D assets.

```python
# Find assets with similar appearance
similar = engine.query(asset_embedding, k=10)
```


### Point Cloud Processing
Efficient queries on LiDAR data.

```python
# Find points matching a pattern
matches = engine.query(pattern_embedding, k=100)
```


---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src

# Run benchmarks
python scripts/run_benchmarks.py
```


---

## 📖 API Reference

### GaussianSplat

```python
splat = GaussianSplat(
    id=0,
    position=[x, y, z],        # 3D position
    color=[r, g, b],           # RGB color
    opacity=0.9,               # Transparency
    scale=[sx, sy, sz],        # Scale factors
    rotation=[w, x, y, z]      # Quaternion
)
```


### HRM2Engine

```python
engine = HRM2Engine(
    n_coarse=100,    # Coarse clusters
    n_fine=1000,     # Fine clusters
    n_probe=5,       # Clusters to search
)

engine.add_splats(splats)
engine.index()

results = engine.query(query_vector, k=10)
```


### SplatMemoryManager

```python
memory = SplatMemoryManager(
    vram_limit=100000,   # Hot tier
    ram_limit=1000000,   # Warm tier
)

memory.add_splats(splats)
splat = memory.get_splat(splat_id)
```


---

## 🔬 Technical Details

### Position Encoding (64D)
Sinusoidal encoding similar to Transformers:
```
PE(x, 2i) = sin(x / 10000^(2i/dim))
PE(x, 2i+1) = cos(x / 10000^(2i/dim))
```


### Color Encoding (512D)
Histogram-based with Gaussian smoothing:
- 8 bins per channel
- 8³ = 512 total dimensions
- Smoothed with Gaussian kernel


### K-Means Implementation
- K-Means++ initialization
- Numba JIT for speed
- Parallel distance computation


---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)


3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Kerbl et al.
- [Numba](https://numba.pydata.org/) - JIT compilation
- [scikit-learn](https://scikit-learn.org/) - Reference implementations

---

## 📧 Contact

Brian Schwabauer - Scwhabauerbriantomas@gmail.com

Project: https://github.com/schwabauerbriantomas-gif/m2m-gaussian-splatting

---

## 🔖 Keywords

`gaussian-splatting` `3dgs` `hierarchical-retrieval` `vector-search` `numba` `kmeans` `similarity-search` `point-cloud` `3d-reconstruction` `nerf` `embedding` `cpu-optimized` `memory-management` `hrm2` `clustering` `jit-compilation` `python`
