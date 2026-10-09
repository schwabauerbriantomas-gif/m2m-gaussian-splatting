# Changelog

## [v2.1.0] - 2026-10-09

### Fixed — audit-driven correctness release
- **Packaging**: the wheel shipped a top-level `src` package (import
  `m2m_gaussian_splatting` failed; the PyPI 1.1.0 wheel had the same defect).
  Package renamed to `m2m_gaussian_splatting` and verified by installing the
  wheel and importing from outside the repo.
- **Memory manager data loss**: `_evict_from_ram` dropped evicted splats
  permanently (any splat living only in RAM was unretrievable). Eviction now
  returns splats to cold storage; tiers are exclusive (no double-counted
  `total_splats`); the RAM limit is enforced on all insert paths;
  `get_stats` returns a defensive copy; `clear` takes the lock; unknown IDs
  no longer create phantom access-count entries.
- **Encoding N=1 crash**: `FullEmbeddingBuilder.build()` raised ValueError
  for a single splat (1D/2D inconsistency between encoders). All encoders
  now return 2D and handle N=0 and NaN input explicitly (NaN → ValueError,
  empty → (0, 640)).
- **Per-batch position normalization**: the embedding of a point depended
  on the rest of the batch (single points encoded to a constant vector).
  `SinusoidalPositionEncoder.fit()` / `bounds` allow fixed scene bounds for
  query-consistent encodings; legacy per-batch behavior stays the default.
- **Color edge cases**: empty color array crashed (`max()` on N=0); the
  `[0,255]` auto-detection misclassified dark scenes; corner colors got a
  smaller histogram norm biasing L2 search (rows now L2-normalized);
  explicit `color_space='01'|'255'` overrides auto-detection.
- **Silent NaN collapse in K-Means**: one NaN in the embeddings collapsed
  every point into cluster 0 without error. `fit()` now rejects non-finite
  input; attribute encoder clips opacity to [0,1] and scales to ≥1e-6
  (log/ratio features previously produced NaN/~1e8 magnitudes).
- **K-Means++ degenerate guard** ported to both CPU variants (duplicated
  data previously produced duplicate centroids / a ValueError).
- **k-means convergence test** used `assert A or B` (could not fail);
  brute-force recall test asserted ≥0.9 instead of ==1.0 on CPU;
  GPU asserts skipped silently without CUDA.
- Quaternion with zero/NaN norm silently produced identity covariance
  (falls back to identity rotation); `SplatEmbedding.full_embedding`
  forces float32; `SplatCluster.bounds` default is float32.
- Benchmark "Key observations" were hardcoded strings unrelated to the
  measured data — now computed from the results (log-log build scaling,
  query growth, recall check), and results persist to
  `benchmark_results_cpu.json` with timestamp/platform/versions.
- Publish workflow built and shipped without running tests (it released
  the broken 1.1.0 wheel); publish now gates on test + build + wheel
  metadata check. CI installs the `[gpu]` extra (torch paths had 0%
  coverage in CI) and reports coverage.

### Added
- `HRM2Engine.save_index()` / `load_index()` — persist and restore the
  built index (npz) so expensive builds survive process restarts.
- Explicit `ValueError` for wrong-dimension queries and `n_probe < 1`.
- 22 regression tests covering every fix above (36 → 58 tests).

### Optimized — measured on Ryzen 5 3400G (8 threads), CPU-only
- Fine (level-2) clustering is now truly lazy: `index()` deferred it but
  still built it eagerly on CPU where the plain search path never uses it.
  It accounted for ~80% of CPU build time.
- Cluster assignment switched from a scalar O(N·K·D) numba loop to BLAS
  GEMM (Gram trick); numba kernels with `prange` now declare
  `parallel=True` (previously prange silently degraded to serial).
- Full sweep (1k-50k splats, `scripts/run_benchmarks.py`, same machine,
  same seeds): build **3.8x-35.4x faster** (50k: 150.5s → 4.25s), query
  **2.4x-6.7x faster** (50k: 47.0ms → 7.0ms), recall@10 = 100% at every
  size. Query path uses a CSR cluster layout (views instead of per-query
  fancy-index copies) and float64 norms. See `benchmark_results_cpu.json`.
- GPU K-Means: per-cluster Python update loop (~6 kernel launches + a
  device sync per cluster per iteration) replaced with vectorized
  `index_add_`/`bincount` (~4 kernels/iteration, 1 sync); `tol` honored
  (accepted-but-never-used before); dead clusters reseeded from farthest
  points; `torch.inference_mode()`; `empty_cache()` before CPU fallback.
- `batch_query` CPU: one coarse GEMM for the whole batch + one candidate
  GEMM per probed cluster (was a full single-query pipeline per row).
- Attribute extraction at index time via `np.stack`; module-level
  precomputed histogram kernel weights.

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
- 10K splats: CPU 8.2ms → GPU 1.1ms (**7.3x** speedup)
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
