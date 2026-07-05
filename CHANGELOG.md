# Changelog

## [v2.0.0] - 2026-07-04

### Added — GPU Acceleration
- **CUDA support via PyTorch** with automatic detection and CPU fallback
- New `src/gpu/` module: `backend.py` (device detection), `gpu_kmeans.py`
  (GPU K-Means), `gpu_search.py` (brute-force L2 k-NN on GPU)
- `HRM2Engine(use_gpu=True)` routes queries through GPU searcher
- `batch_query()` parallelized on GPU (single upload, batched top-k)
- K-Means++ init runs on GPU when available
- `detect_device()` and `HAS_CUDA` exported at package level
- `pyproject.toml`: optional `[gpu]` dependency group (`torch>=2.0.0`)

### Fixed
- `generate_test_splats()` no longer pollutes global `np.random` state
  (uses local `RandomState`)
- `ColorHistogramEncoder` normalization threshold (>1.5 instead of >1.0)
  prevents misinterpreting float [0,1] data as [0,255]
- Thread safety in `SplatMemoryManager` via `threading.RLock`
- `GPUSearcher.search()` returns 1D arrays (was returning 2D batch)
- K-Means++ GPU init handles degenerate distributions (NaN/inf guard)

### Optimized
- Fine clustering is now lazy on GPU (skipped entirely — search is
  brute-force) and deferred on CPU until `query_with_details()` is called
- Vectorized attribute extraction in `index()` (preallocated arrays
  instead of list comprehensions)
- `batch_query` GPU path: single tensor upload, batched `torch.topk`
- Build time at 50K splats: **156s → 7.4s** (21x faster, GPU)

### Benchmark Results (RTX 3090, measured)
- 10K splats: CPU 8.2ms → GPU 1.1ms (**7.3x** speedup
- 50K splats: CPU 49.8ms → GPU 3.6ms (**13.7x** speedup)

### Tests
- 12 → 36 tests (24 new tests covering GPU, encoding edge cases,
  thread safety, serialization, and regression tests)

## [v1.1.0] - 2026-07-04

### Added
- GitHub Actions CI workflow (test + lint)
- PyPI Trusted Publisher auto-release workflow
- Full test suite

### Fixed
- Invalid build-backend (setuptools.backends._legacy:_Backend)
