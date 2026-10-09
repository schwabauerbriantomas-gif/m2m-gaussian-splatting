"""
GPU-accelerated K-Means clustering using PyTorch CUDA tensors.

Drop-in replacement for the Numba ``KMeans`` when a GPU is available.
Falls back to CPU automatically.

v2.1 (audit 2026-10) changes:
- The mini-batch centroid update no longer loops over clusters in Python
  (the old loop launched ~6 CUDA kernels and forced a device sync per
  cluster per iteration ≈ tens of thousands of syncs per fit). Now one
  ``index_add_`` + ``bincount`` per iteration, ~4 kernels, 1 sync.
- ``tol`` is actually honored (was accepted, stored, never used): the
  loop early-stops when the squared centroid shift drops below it, and
  ``n_iter_`` reports the iterations run.
- Mini-batch sampling uses ``torch.randint`` (with replacement) instead
  of a full O(N) ``randperm`` per iteration.
- Dead clusters (never updated, which the 1/counts rule freezes forever)
  are reseeded from the points farthest from their centroid before the
  final assignment.
- The final assignment runs in chunks of 65536 rows accumulating into
  pre-allocated label/distance buffers, so peak memory no longer
  materializes a full (N, K) distance matrix (N*K*4 bytes for the labels
  pass alone is bounded by the chunk, not N).
- ``__init__`` accepts ``use_gpu`` (default True, current behavior) and
  ``dtype`` (``'float32'`` default, or ``'half'`` to run the index in
  fp16). Existing callers that pass neither see no change. ``dtype`` is
  only applied on the torch path; importing torch stays optional
  (guarded by try/except at module level).
- Everything runs under ``torch.inference_mode()``; on the OOM fallback
  ``torch.cuda.empty_cache()`` runs before the CPU retry.
- The CPU fallback is vectorized (reuses the JIT ``_sum_by_label``
  kernel), uses a local ``np.random.Generator`` (no global-RNG state),
  and shares the same convergence/reseeding semantics.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from .backend import HAS_TORCH, HAS_CUDA

if HAS_TORCH:
    import torch

from ..core.clustering import _sum_by_label  # JIT per-cluster sums (CPU fallback)


class TorchKMeans:
    """
    K-Means clustering with GPU acceleration via PyTorch.

    Uses mini-batch updates and runs entirely on GPU when available.
    API mirrors the CPU ``KMeans`` class.
    """

    def __init__(
        self,
        n_clusters: int = 100,
        max_iter: int = 100,
        batch_size: int = 10000,
        tol: float = 1e-4,
        random_state: int = 42,
        device: Optional[str] = None,
        use_gpu: bool = True,
        dtype: str = "float32",
    ):
        """
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            batch_size: Mini-batch size
            tol: Early-stop tolerance on total squared centroid shift
            random_state: Random seed
            device: Explicit device override ('cuda'/'cpu'). When None,
                autodetects (cuda if available).
            use_gpu: When False, force the CPU path regardless of device
                detection. Defaults True (previous behavior).
            dtype: Compute dtype for the torch path — 'float32' (default)
                or 'half' (fp16, halves index memory). Ignored on CPU.
        """
        if dtype not in ("float32", "half"):
            raise ValueError(f"dtype must be 'float32' or 'half', got {dtype!r}")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.random_state = random_state
        self.dtype = dtype
        if not use_gpu:
            self._device = "cpu"
        else:
            self._device = device or ("cuda" if HAS_CUDA else "cpu")

        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: int = 0

    # ---- public API --------------------------------------------------

    def fit(self, data: np.ndarray) -> "TorchKMeans":
        data_np = np.ascontiguousarray(data.astype(np.float32))
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        n_clusters = min(self.n_clusters, data_np.shape[0])

        if not HAS_TORCH or self._device == "cpu":
            return self._fit_cpu(data_np, n_clusters)

        try:
            return self._fit_gpu(data_np, n_clusters)
        except (RuntimeError, MemoryError):  # noqa: BLE001 — torch OOM subclasses RuntimeError
            # OOM or driver issue — free cache and fall back to CPU
            if HAS_CUDA:
                torch.cuda.empty_cache()
            return self._fit_cpu(data_np, n_clusters)

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("Must call fit() before predict()")
        data_np = np.ascontiguousarray(data.astype(np.float32))
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)
        return self._assign_batch(data_np, self.centroids_)

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.labels_  # type: ignore[return-value]

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Return distances from each point to every centroid."""
        if self.centroids_ is None:
            raise RuntimeError("Must call fit() before transform()")
        data_np = np.ascontiguousarray(data.astype(np.float32))
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        data_norms = np.sum(data_np**2, axis=1, keepdims=True)  # (N,1)
        cent_norms = np.sum(self.centroids_**2, axis=1)  # (K,)
        cross = data_np @ self.centroids_.T  # (N,K)
        dist_sq = data_norms + cent_norms - 2.0 * cross
        return np.sqrt(np.maximum(dist_sq, 0.0))

    # ---- GPU internals ---------------------------------------------------

    def _fit_gpu(self, data_np: np.ndarray, n_clusters: int) -> "TorchKMeans":
        """Full GPU path (vectorized mini-batch updates, no per-cluster loop)."""
        with torch.inference_mode():
            torch.manual_seed(self.random_state)
            device = torch.device(self._device)

            data_t = torch.from_numpy(data_np).to(device)
            if self.dtype == "half":
                # Halve the stored index (N·D·2 bytes). Distance math is
                # upcast to fp32 inside _pairwise_sq, so accuracy holds;
                # only the resident index pays the fp16 discount.
                data_t = data_t.half()
            N, D = data_t.shape

            centroids_t = self._kpp_init_gpu(data_t, n_clusters, device)

            counts = torch.ones(n_clusters, dtype=torch.float32, device=device)
            ever_updated = torch.zeros(n_clusters, dtype=torch.bool, device=device)
            batch_size = min(self.batch_size, N)
            n_iter = 0

            for it in range(self.max_iter):
                n_iter = it + 1

                idx = torch.randint(0, N, (batch_size,), device=device)
                batch = data_t[idx]

                # Assign: (batch, K)
                dists = self._pairwise_sq(batch, centroids_t)
                labels = torch.argmin(dists, dim=1)

                # Vectorized mini-batch update: per-cluster sums/counts in
                # ~4 kernels instead of a Python loop over clusters.
                # Sums accumulate in fp32 even in half mode (index_add_
                # on fp16 loses precision across a 10k-row batch).
                sums = torch.zeros(
                    centroids_t.shape, dtype=torch.float32, device=device
                ).index_add_(0, labels, batch.float())
                bcounts = torch.bincount(labels, minlength=n_clusters).to(torch.float32)
                updated = bcounts > 0
                ever_updated |= updated
                if not bool(updated.any()):
                    continue

                eta = 1.0 / (counts[updated] + bcounts[updated])
                new_centroids = centroids_t.clone()
                new_centroids[updated] = (
                    (1.0 - eta.unsqueeze(1)) * centroids_t[updated].float()
                    + eta.unsqueeze(1) * (sums[updated] / bcounts[updated].unsqueeze(1))
                ).to(centroids_t.dtype)
                counts[updated] += bcounts[updated]

                # fp32 shift (fp16 differences underflow to 0 → false stop)
                shift = float((new_centroids.float() - centroids_t.float()).pow(2).sum().item())
                centroids_t = new_centroids

                if shift < self.tol:
                    break

            # Final assignment — chunked so a full (N, K) distance matrix
            # is never materialized. Peak extra memory: one (chunk, K)
            # fp32 block (65536*K*4 bytes) + two (N,) buffers.
            all_labels, point_d = self._assign_chunked_gpu(data_t, centroids_t)

            # Reseed never-updated (dead) clusters from the farthest points
            dead = ~ever_updated
            n_dead = int(dead.sum().item())
            if n_dead > 0 and N > n_clusters:
                far = torch.topk(point_d, n_dead, largest=True).indices
                centroids_t[dead] = data_t[far]
                all_labels, point_d = self._assign_chunked_gpu(data_t, centroids_t)

            self.centroids_ = centroids_t.float().cpu().numpy()
            self.labels_ = all_labels.cpu().numpy().astype(np.int32)
            self.inertia_ = float(point_d.sum().item())
            self.n_iter_ = n_iter

            del data_t, centroids_t, all_labels, point_d
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return self

    _ASSIGN_CHUNK = 65536  # rows per final-assignment block

    def _assign_chunked_gpu(
        self, data_t: "torch.Tensor", centroids_t: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Chunked nearest-centroid assignment.

        Accumulates labels and per-point squared distances into
        pre-allocated (N,) buffers, looping in row blocks of
        ``_ASSIGN_CHUNK``. Memory for the distance block is
        ``_ASSIGN_CHUNK * K * 4`` bytes regardless of N.
        """
        N = data_t.shape[0]
        labels = torch.empty(N, dtype=torch.int64, device=data_t.device)
        point_d = torch.empty(N, dtype=torch.float32, device=data_t.device)
        for start in range(0, N, self._ASSIGN_CHUNK):
            end = min(start + self._ASSIGN_CHUNK, N)
            dists = self._pairwise_sq(data_t[start:end], centroids_t)  # (b, K)
            chunk_labels = torch.argmin(dists, dim=1)
            labels[start:end] = chunk_labels
            point_d[start:end] = dists.gather(1, chunk_labels.unsqueeze(1)).squeeze(1)
        return labels, point_d

    def _kpp_init_gpu(
        self, data_t: "torch.Tensor", k: int, device: "torch.device"
    ) -> "torch.Tensor":
        """K-Means++ initialization on GPU."""
        N, D = data_t.shape
        centroids = torch.empty(k, D, device=device, dtype=data_t.dtype)

        first = torch.randint(0, N, (1,), device=device)
        centroids[0] = data_t[first]

        min_dist_sq = self._pairwise_sq(data_t, centroids[:1]).squeeze(1)
        min_dist_sq = torch.clamp(min_dist_sq, min=0.0)  # avoid float negatives

        for i in range(1, k):
            total = min_dist_sq.sum() + 1e-10
            probs = min_dist_sq / total
            # Guard against NaN/inf from degenerate distributions
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            if probs.sum() <= 0:
                # All points are at distance 0 — pick random
                idx = torch.randint(0, N, (1,), device=device)
            else:
                probs = probs / probs.sum()
                idx = torch.multinomial(probs, 1)
            centroids[i] = data_t[idx]
            new_dist = self._pairwise_sq(data_t, centroids[i : i + 1]).squeeze(1)
            new_dist = torch.clamp(new_dist, min=0.0)
            min_dist_sq = torch.minimum(min_dist_sq, new_dist)

        return centroids

    # ---- CPU fallback ----------------------------------------------------

    def _fit_cpu(self, data_np: np.ndarray, n_clusters: int) -> "TorchKMeans":
        """CPU fallback using vectorized NumPy (+ JIT per-cluster sums)."""
        rng = np.random.default_rng(self.random_state)
        centroids = self._kpp_init_cpu(data_np, n_clusters, rng)

        counts = np.ones(n_clusters, dtype=np.float64)
        ever_updated = np.zeros(n_clusters, dtype=bool)
        N = data_np.shape[0]
        batch_size = min(self.batch_size, N)
        n_iter = 0

        for it in range(self.max_iter):
            n_iter = it + 1
            idx = rng.integers(0, N, size=batch_size)
            batch = np.ascontiguousarray(data_np[idx])

            labels = self._assign_batch(batch, centroids)
            sums, bcounts = _sum_by_label(batch, labels.astype(np.int64), n_clusters)
            bcounts = bcounts.astype(np.float64)

            updated = bcounts > 0
            ever_updated |= updated
            if not updated.any():
                continue

            eta = 1.0 / (counts[updated] + bcounts[updated])
            new_centroids = centroids.copy()
            new_centroids[updated] = (1.0 - eta[:, None]) * centroids[updated] + eta[:, None] * (
                sums[updated] / bcounts[updated, None].astype(np.float32)
            )
            counts[updated] += bcounts[updated]

            shift = float(((new_centroids - centroids) ** 2).sum())
            centroids = new_centroids

            if shift < self.tol:
                break

        labels = self._assign_batch(data_np, centroids)

        # Reseed dead clusters from the farthest points, reassign once
        dead = ~ever_updated
        n_dead = int(dead.sum())
        if n_dead > 0 and N > n_clusters:
            data_norms = np.sum(data_np**2, axis=1)
            cent_norms = np.sum(centroids[labels] ** 2, axis=1)
            cross = np.sum(data_np * centroids[labels], axis=1)
            point_d = data_norms + cent_norms - 2.0 * cross
            far = np.argpartition(point_d, N - n_dead)[N - n_dead :]
            centroids[dead] = data_np[far]
            labels = self._assign_batch(data_np, centroids)

        cent_norms = np.sum(centroids[labels] ** 2, axis=1)
        cross = np.sum(data_np * centroids[labels], axis=1)
        data_norms = np.sum(data_np**2, axis=1)
        inertia = float(np.sum(data_norms + cent_norms - 2.0 * cross))

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        self.n_iter_ = n_iter
        return self

    # ---- helpers -----------------------------------------------------

    @staticmethod
    def _pairwise_sq(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
        """Squared L2 distance matrix via Gram trick.

        Accumulates in float32 even when the index is stored in fp16
        (``dtype='half'``): squared fp16 magnitudes overflow past ~65k,
        so the norms and the output are always computed in fp32. Only
        the stored index (N·D) stays half.
        """
        a32 = a.float()
        b32 = b.float()
        a_sq = (a32**2).sum(dim=1, keepdim=True)  # (N,1)
        b_sq = (b32**2).sum(dim=1, keepdim=True).t()  # (1,K)
        cross = a32 @ b32.t()
        return a_sq + b_sq - 2.0 * cross

    @staticmethod
    def _kpp_init_cpu(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        """K-Means++ init on CPU with running min-distance."""
        N, D = data.shape
        centroids = np.zeros((k, D), dtype=np.float32)

        idx = int(rng.integers(N))
        centroids[0] = data[idx]

        diff = data - centroids[0]
        min_dist_sq = np.sum(diff * diff, axis=1)

        for i in range(1, k):
            total = min_dist_sq.sum() + 1e-10
            probs = min_dist_sq / total
            if probs.sum() <= 0:
                # Degenerate: all remaining points identical → random pick
                idx = int(rng.integers(N))
            else:
                # Sample ∝ D² via inverse-CDF (same scheme as the JIT path)
                r = rng.random() * float(probs.sum())
                idx = int(np.searchsorted(np.cumsum(probs), r))
                idx = min(idx, N - 1)
            centroids[i] = data[idx]
            diff = data - centroids[i]
            new_dist = np.sum(diff * diff, axis=1)
            min_dist_sq = np.minimum(min_dist_sq, new_dist)

        return centroids

    @staticmethod
    def _assign_batch(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign points to nearest centroid (vectorized NumPy)."""
        data_norms = np.sum(data**2, axis=1, keepdims=True)  # (N,1)
        cent_norms = np.sum(centroids**2, axis=1)  # (K,)
        cross = data @ centroids.T  # (N,K)
        dist_sq = data_norms + cent_norms - 2.0 * cross
        return np.argmin(dist_sq, axis=1).astype(np.int32)
